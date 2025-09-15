# thms/model_v4_1.py
from __future__ import annotations
from typing import Optional, Tuple
from collections import deque
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .store import IConsolidatedStore
from .nfm import EnhancedNeverForgettingMemory, NeverForgettingMemoryAdapter
from .router import HybridRouterV2
from .consolidation_v2 import MemoryConsolidationManagerV2
from .subsystems import (
    EnhancedHyperGeometricMemoryWithTransformerV2,
    EnhancedCGMNMemoryWithTransformerV2,
    EnhancedCurvedMemoryWithTransformerV2,
)

logger = logging.getLogger("EnhancedTripleHybridMemoryV4_1")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

Tensor = torch.Tensor

class EnhancedTripleHybridMemoryV4_1(nn.Module):
    """
    V4.1:
      • HG: entanglement + QIP + optional bank prior
      • Curved: optional dilated TCN
      • 4-way integration + topology token -> 5 tokens routed by HybridRouterV2
      • NFM token appended at fusion -> 6*D projection
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
                 base_nfm_topk: int = 6,
                 max_nfm_topk: int = 24,
                 nfm_gain_on_spike: float = 1.0,
                 consolidated_store: Optional[IConsolidatedStore] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads

        # Subsystems (V4.1 with extras)
        self.hyper_geometric = EnhancedHyperGeometricMemoryWithTransformerV2(
            input_dim, hg_dim, hg_slots, hg_qubits, n_transformer_layers, n_heads, attention_type,
            enable_bank=True, bank_size=1024, use_entanglement=True
        )
        self.cgmn = EnhancedCGMNMemoryWithTransformerV2(
            input_dim, cgmn_dim, cgmn_slots, cgmn_slot_dim, n_transformer_layers, n_heads, attention_type
        )
        self.curved = EnhancedCurvedMemoryWithTransformerV2(
            input_dim, curved_hidden, curved_curvature, curved_slots, n_transformer_layers, n_heads, attention_type,
            use_tcn=True
        )
        self.consolidated_memory = EnhancedHyperGeometricMemoryWithTransformerV2(
            input_dim, hg_dim, 512, hg_qubits, n_transformer_layers, n_heads, attention_type,
            enable_bank=False, use_entanglement=True
        )

        # Router over 5 tokens (HG, CGMN, Curved, Consolidated, TOPO)
        self.router = HybridRouterV2(input_dim=input_dim, n_subsystems=5, n_heads=n_heads)
        self.cross_memory_attention = nn.MultiheadAttention(input_dim, num_heads=n_heads, batch_first=True)

        fusion_in_dim = input_dim * 6  # 5 routed tokens + 1 NFM token
        self.memory_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(fusion_in_dim, n_heads, dim_feedforward=2048, batch_first=True),
            num_layers=2
        )
        self.fusion_projection = nn.Linear(fusion_in_dim, output_dim)

        # Importance predictor
        self.importance_predictor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim * 2, n_heads, dim_feedforward=1024, batch_first=True),
            num_layers=1
        )
        self.importance_head = nn.Sequential(
            nn.Linear(input_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        # NFM
        self.nfm_core = EnhancedNeverForgettingMemory(
            base_path=nfm_path, enable_compression=True, enable_indexing=True,
            enable_relationships=True, vector_dim=nfm_vector_dim
        )
        self.nfm = NeverForgettingMemoryAdapter(self.nfm_core, input_dim, vector_dim=nfm_vector_dim)
        self._seq_pool = lambda x: x.mean(dim=1)
        self.base_nfm_topk = int(base_nfm_topk)
        self.max_nfm_topk = int(max_nfm_topk)
        self.nfm_gain_on_spike = float(nfm_gain_on_spike)

        # Consolidation (simple gating backbone)
        self.consolidation_network = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim * 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim), nn.Tanh()
        )
        self.consolidation_gates = nn.Sequential(
            nn.Linear(input_dim * 3, 3), nn.Softmax(dim=-1)
        )

        # Topology consolidator -> token
        self.topo_consolidator = MemoryConsolidationManagerV2(embedding_dim=input_dim, topological_dim=128)
        self.topo_to_input = nn.Linear(128, input_dim)

        # State
        self.consolidation_counter = 0
        self.consolidation_interval = 10
        self.lightbulb_intensity = 0.0
        self.cons_novelty = 0.0
        self.lightbulb_decay = 0.9
        self.register_buffer('consolidated_usage', torch.zeros(512))

        # Consolidated store
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

    # ---------- helpers ----------
    def _choose_nfm_topk(self, base_k: int, intensity: float) -> int:
        extra = int(self.nfm_gain_on_spike * intensity * (self.max_nfm_topk - base_k))
        return max(1, min(self.max_nfm_topk, base_k + extra))

    def detect_cross_memory_lightbulb(self, hg_nov: Tensor, cg_sim: Tensor, cu_spread: Tensor) -> Tuple[bool, Tensor]:
        hg_norm = torch.sigmoid(hg_nov * 5)
        cg_norm = torch.sigmoid(cg_sim * 5)
        cu_norm = torch.sigmoid(cu_spread * 5)
        combined = (hg_norm + (1 - cg_norm) + cu_norm) / 3
        return (combined.mean() > 0.6, combined.mean())

    def coordinate_lightbulb_moments(self, hg_query: Tensor, cgmn_positions: Tensor, curved_activation: Tensor) -> float:
        hg_light, hg_nov = self.hyper_geometric.detect_lightbulb_moment(hg_query)
        cg_ins, cg_sim = self.cgmn.detect_geometric_insight(cgmn_positions)
        cu_chain, cu_spread = self.curved.detect_associative_chain(curved_activation)
        is_cross, intensity = self.detect_cross_memory_lightbulb(hg_nov.mean(), cg_sim.mean(), cu_spread.mean())
        if is_cross or hg_light or cg_ins or cu_chain:
            new_intensity = max(float(self.lightbulb_intensity), float(intensity.item()))
            self.lightbulb_intensity = 0.9 * self.lightbulb_intensity + 0.1 * new_intensity
        else:
            self.lightbulb_intensity *= self.lightbulb_decay
        return float(self.lightbulb_intensity)

    def _hns_detect_consolidated_lightbulb(self, query: Tensor, consolidated_output: Tensor) -> Tuple[bool, float]:
        if consolidated_output is None or (isinstance(consolidated_output, torch.Tensor) and consolidated_output.norm() == 0):
            return False, 0.0
        q = query.mean(dim=1) if query.dim() == 3 else query
        dif = (consolidated_output.mean(dim=1) - q)
        consolidated_novelty = torch.sigmoid(dif.norm(dim=-1) * 5).mean().item()
        return (consolidated_novelty > 0.7), float(consolidated_novelty)

    def _hns_cross_memory_integrate(self, hg_out, cgmn_out, curved_out, consolidated_out=None):
        outs = [hg_out, cgmn_out, curved_out]
        if consolidated_out is not None:
            outs.append(consolidated_out)
        mem = torch.stack(outs, dim=1)        # [B,M,T,D] (M=4)
        pooled = mem.mean(dim=2)              # [B,M,D]
        attn, _ = self.cross_memory_attention(pooled, pooled, pooled)
        refined = []
        for i in range(attn.size(1)):
            token = attn[:, i:i+1, :]
            sa, _ = self.cross_memory_attention(token, token, token)
            refined.append(sa.squeeze(1))
        return torch.stack([attn[:, i, :] + refined[i] for i in range(attn.size(1))], dim=1)  # [B,4,D]

    def _hns_adaptive_consolidation(self, pooled: Tensor, importance: Tensor) -> None:
        B, M, D = pooled.shape
        gates = self.consolidation_gates(pooled.reshape(B, -1))                # [B,3]
        consolidated = self.consolidation_network((gates.unsqueeze(-1) * pooled).reshape(B, -1))  # [B,D]
        if (self.consolidation_counter % self.consolidation_interval) == 0:
            _ = self.consolidated_memory(consolidated.unsqueeze(1), operation='write', lightbulb_intensity=0.0)
        self.consolidation_counter += 1

    def _hns_retrieve_consolidated(self, query: Tensor, lightbulb_intensity: float = 0.0) -> Tensor:
        if self.consolidation_counter == 0:
            self.cons_novelty = 0.0
            return torch.zeros_like(query)
        out = self.consolidated_memory(query, operation='read', lightbulb_intensity=lightbulb_intensity)
        _, nov = self._hns_detect_consolidated_lightbulb(query, out)
        self.cons_novelty = nov
        return out

    # ---------- forward ----------
    def forward(self, x: Tensor, context: Optional[Tensor] = None, operation: str = 'read') -> Tensor:
        B, T, D = x.shape
        if context is None:
            context = x.mean(dim=1, keepdim=True).expand(-1, T, -1)

        # Importance estimate
        contextual_input = torch.cat([x, context], dim=-1)
        importance_features = self.importance_predictor(contextual_input)
        importance = self.importance_head(importance_features.mean(dim=1))  # [B,1]

        # Lightbulb coordination
        if operation == 'read':
            hg_query = self.hyper_geometric.encode_to_manifold(x).mean(dim=2)
            cgmn_proj_flat = self.cgmn.manifold_projection(x)
            if cgmn_proj_flat.numel() == 0:
                cgmn_positions = x.new_zeros(x.size(0), x.size(1), 1, 3)
            else:
                cgmn_positions = cgmn_proj_flat.view(-1, x.size(1), -1, 3)
            curved_encoded = self.curved.encoder(x)
            curved_activation = F.relu(curved_encoded.mean(dim=1))
            cross_intensity = self.coordinate_lightbulb_moments(hg_query, cgmn_positions, curved_activation)
        else:
            cross_intensity = 0.0

        # Branch outputs
        cgmn_out = self.cgmn(x, operation, importance, cross_intensity, context)
        curved_out = self.curved(x, operation, importance, cross_intensity, context)

        if operation == 'write':
            hg_out_w = self.hyper_geometric(x, operation, cross_intensity, context)
            pooled = torch.stack([hg_out_w, cgmn_out, curved_out], dim=1).mean(dim=2)  # [B,3,D]
            self._hns_adaptive_consolidation(pooled, importance)
            with torch.no_grad():
                topo_proto = self.topo_consolidator.consolidate(pooled, importance)  # [B,128]
                try:
                    self.cons_store.write(
                        series_id="topo_prototype",
                        content=topo_proto,
                        tags=["topology", "consolidation_v2"],
                        context={"op": "write"},
                        importance=float(importance.mean().item()),
                        confidence=0.9,
                    )
                except Exception:
                    pass
            return x

        # READ path
        consolidated_out = self._hns_retrieve_consolidated(x, cross_intensity)       # [B,T,D]
        combined_intensity = float(0.7 * cross_intensity + 0.3 * self.cons_novelty)
        hg_out = self.hyper_geometric(x, operation, combined_intensity, context)

        integrated4 = self._hns_cross_memory_integrate(hg_out, cgmn_out, curved_out, consolidated_out)  # [B,4,D]

        topo_recalled = self.topo_consolidator.recall(x, topk=1)  # [B,1,128] or [B,0,128]
        if topo_recalled.size(1) == 0:
            topo_token = x.new_zeros(B, 1, self.input_dim)
        else:
            topo_token = self.topo_to_input(topo_recalled.squeeze(1)).unsqueeze(1)  # [B,1,D]

        tokens5 = torch.cat([integrated4, topo_token], dim=1)  # [B,5,D]

        # Route & weight -> 5*D
        seq_ctx = x.mean(dim=1)
        weights, refined = self.router(tokens5, context=seq_ctx)  # [B,5], [B,5,D]
        fused_tokens = refined * weights.unsqueeze(-1)            # [B,5,D]
        routed_vec = fused_tokens.reshape(B, -1)                  # [B,5D]

        # NFM token -> +D => 6D
        k = self._choose_nfm_topk(self.base_nfm_topk, cross_intensity)
        nfm_ctx, _ = self.nfm.query_by_vectors(x.mean(dim=1), topk=k)  # [B,k,D]
        nfm_token = nfm_ctx.mean(dim=1, keepdim=True)                  # [B,1,D]
        fused_vec = torch.cat([routed_vec, nfm_token.squeeze(1)], dim=-1)  # [B,6D]

        fused = self.memory_fusion(fused_vec.unsqueeze(1)).squeeze(1)       # [B,6D]
        output = self.fusion_projection(fused)                              # [B,out]
        return output

# Smoke test
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 2, 5, 32
    x = torch.randn(B, T, D)
    model = EnhancedTripleHybridMemoryV4_1(
        input_dim=D, output_dim=16, n_heads=4, n_transformer_layers=2,
        base_nfm_topk=4, max_nfm_topk=12
    )
    y_read = model(x, operation='read')
    y_write = model(x, operation='write')
    print("READ:", y_read.shape)
    print("WRITE:", y_write.shape)
