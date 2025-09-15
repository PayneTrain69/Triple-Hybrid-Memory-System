# thms/model_v4.py
from __future__ import annotations
from typing import Optional, Tuple
from collections import deque
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .store import IConsolidatedStore
from .nfm import EnhancedNeverForgettingMemory, NeverForgettingMemoryAdapter
from .subsystems import (
    EnhancedHyperGeometricMemoryWithTransformerV2,
    EnhancedCGMNMemoryWithTransformerV2,
    EnhancedCurvedMemoryWithTransformerV2,
)

logger = logging.getLogger("EnhancedTripleHybridMemoryV4")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

Tensor = torch.Tensor

class EnhancedTripleHybridMemoryV4(nn.Module):
    """
    V4 baseline:
      • 3 branches: HG / CGMN / Curved + consolidated branch for write/read
      • Cross-memory attention integration -> 4*D fusion
      • NFM used as read-side context (not concatenated into fusion vector)
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hg_dim: int = 24, hg_slots: int = 1028, hg_qubits: int = 8,
                 cgmn_dim: int = 16, cgmn_slots: int = 512, cgmn_slot_dim: int = 256,
                 curved_hidden: int = 256, curved_curvature: int = 8, curved_slots: int = 128,
                 n_transformer_layers: int = 3, n_heads: int = 8,
                 attention_type: str = 'multiscale',
                 nfm_path: str = "./enhanced_memory_db",
                 nfm_vector_dim: int = 512,
                 consolidated_store: Optional[IConsolidatedStore] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads

        # Subsystems (V4: no entanglement, no bank; Curved without TCN)
        self.hyper_geometric = EnhancedHyperGeometricMemoryWithTransformerV2(
            input_dim, hg_dim, hg_slots, hg_qubits, n_transformer_layers, n_heads, attention_type,
            enable_bank=False, use_entanglement=False
        )
        self.cgmn = EnhancedCGMNMemoryWithTransformerV2(
            input_dim, cgmn_dim, cgmn_slots, cgmn_slot_dim, n_transformer_layers, n_heads, attention_type
        )
        self.curved = EnhancedCurvedMemoryWithTransformerV2(
            input_dim, curved_hidden, curved_curvature, curved_slots, n_transformer_layers, n_heads, attention_type,
            use_tcn=False
        )
        self.consolidated_memory = EnhancedHyperGeometricMemoryWithTransformerV2(
            input_dim, hg_dim, 512, hg_qubits, n_transformer_layers, n_heads, attention_type,
            enable_bank=False, use_entanglement=False
        )

        # Router & fusion (4 tokens -> 4D)
        self.router = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, n_heads, dim_feedforward=1024, batch_first=True),
            num_layers=2
        )
        self.router_head = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 4), nn.Softmax(dim=-1)
        )
        self.cross_memory_attention = nn.MultiheadAttention(input_dim, num_heads=n_heads, batch_first=True)
        self.memory_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim * 4, n_heads, dim_feedforward=2048, batch_first=True),
            num_layers=2
        )
        self.fusion_projection = nn.Linear(input_dim * 4, output_dim)

        # Importance predictor
        self.importance_predictor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim * 2, n_heads, dim_feedforward=1024, batch_first=True),
            num_layers=1
        )
        self.importance_head = nn.Sequential(
            nn.Linear(input_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        # NFM (read-side context)
        self.nfm_core = EnhancedNeverForgettingMemory(
            base_path=nfm_path, enable_compression=True, enable_indexing=True,
            enable_relationships=True, vector_dim=nfm_vector_dim
        )
        self.nfm = NeverForgettingMemoryAdapter(self.nfm_core, input_dim, vector_dim=nfm_vector_dim)
        self._seq_pool = lambda x: x.mean(dim=1)

        # State
        self.consolidation_counter = 0
        self.consolidation_interval = 10
        self.register_buffer('consolidated_usage', torch.zeros(512))
        self.lightbulb_intensity = 0.0
        self.lightbulb_decay = 0.9

        # Consolidated store (NoOp default)
        if consolidated_store is None:
            class _NoOpStore(IConsolidatedStore):
                def write(self, *a, **k): return None
                def read_latest(self, *a, **k): return None
                def read_version(self, *a, **k): return None
                def snapshot(self, *a, **k): return "noop"
                def restore_snapshot(self, *a, **k): return False
                def flush_events(self): return []
            self.cons_store = _NoOpStore()
        else:
            self.cons_store = consolidated_store

    # ------------ helpers (trimmed vs V4.1) ------------
    def _cross_memory_integrate(self, hg_out, cgmn_out, curved_out, consolidated_out=None):
        outs = [hg_out, cgmn_out, curved_out]
        if consolidated_out is not None:
            outs.append(consolidated_out)
        mem = torch.stack(outs, dim=1)            # [B,M,T,D] (M=4)
        pooled = mem.mean(dim=2)                  # [B,M,D]
        attn, _ = self.cross_memory_attention(pooled, pooled, pooled)
        refined = []
        for i in range(attn.size(1)):
            token = attn[:, i:i+1, :]
            sa, _ = self.cross_memory_attention(token, token, token)
            refined.append(sa.squeeze(1))
        return torch.stack([attn[:, i, :] + refined[i] for i in range(attn.size(1))], dim=1)  # [B,4,D]

    # ------------ forward ------------
    def forward(self, x: Tensor, context: Optional[Tensor] = None, operation: str = 'read') -> Tensor:
        B, T, D = x.shape
        if context is None:
            context = x.mean(dim=1, keepdim=True).expand(-1, T, -1)

        # Importance
        contextual_input = torch.cat([x, context], dim=-1)
        importance_features = self.importance_predictor(contextual_input)
        importance = self.importance_head(importance_features.mean(dim=1))  # [B,1]

        # Branch outputs
        hg_out = self.hyper_geometric(x, operation, self.lightbulb_intensity, context)
        cgmn_out = self.cgmn(x, operation, importance, self.lightbulb_intensity, context)
        curved_out = self.curved(x, operation, importance, self.lightbulb_intensity, context)

        if operation == 'write':
            pooled = torch.stack([hg_out, cgmn_out, curved_out], dim=1).mean(dim=2)  # [B,3,D]
            if (self.consolidation_counter % self.consolidation_interval) == 0:
                _ = self.consolidated_memory(pooled.unsqueeze(1), operation='write', lightbulb_intensity=0.0)
            self.consolidation_counter += 1
            with torch.no_grad():
                try:
                    self.cons_store.write(
                        series_id="input_summary",
                        content=x.mean(dim=1),
                        tags=["write_operation"],
                        context={"op": "write"},
                        importance=float(importance.mean().item()),
                        confidence=0.9,
                    )
                except Exception:
                    pass
            return x

        # READ path
        consolidated_out = self.consolidated_memory(x, operation='read', lightbulb_intensity=self.lightbulb_intensity)
        integrated = self._cross_memory_integrate(hg_out, cgmn_out, curved_out, consolidated_out)  # [B,4,D]

        routing_weights = self.router(integrated).mean(dim=1)   # [B,D]
        routing_weights = self.router_head(routing_weights)     # [B,4]
        fused_vec = torch.cat([
            integrated[:, 0, :] * routing_weights[:, 0:1],
            integrated[:, 1, :] * routing_weights[:, 1:2],
            integrated[:, 2, :] * routing_weights[:, 2:3],
            integrated[:, 3, :] * routing_weights[:, 3:4],
        ], dim=-1)  # [B,4D]

        fused = self.memory_fusion(fused_vec.unsqueeze(1)).squeeze(1)  # [B,4D]

        # NFM context (not concatenated; side context if you want to use later)
        _nfm_ctx, _ = self.nfm.query_by_vectors(x.mean(dim=1), topk=6)

        output = self.fusion_projection(fused)
        return output

# Smoke test
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 2, 5, 32
    x = torch.randn(B, T, D)
    model = EnhancedTripleHybridMemoryV4(input_dim=D, output_dim=16, n_heads=4, n_transformer_layers=2)
    y_read = model(x, operation='read')
    y_write = model(x, operation='write')
    print("READ:", y_read.shape)
    print("WRITE:", y_write.shape)
