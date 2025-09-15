# thms/subsystems.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantum import QuantumEntanglementLayer, QuantumInterferenceProjection
Tensor = torch.Tensor

class EnhancedHyperGeometricMemoryWithTransformerV2(nn.Module):
    """
    V4.1 HG branch: Entanglement -> QIP -> optional bank prior (toggle via enable_bank).
    V4 HG branch: set enable_bank=False and skip entanglement in your V4 model.
    """
    def __init__(self,
                 input_dim: int,
                 hg_dim: int,
                 mem_slots: int,
                 quantum_qubits: int,
                 n_transformer_layers: int,
                 n_heads: int,
                 attention_type: str,
                 enable_bank: bool = True,
                 bank_size: int = 1024,
                 use_entanglement: bool = True):
        super().__init__()
        self.embed = nn.Linear(input_dim, input_dim)
        self.use_entanglement = bool(use_entanglement)
        if self.use_entanglement:
            self.entangle = QuantumEntanglementLayer(dim=input_dim, num_qubits=quantum_qubits)
        else:
            self.register_module("entangle", None)
        self.qip = QuantumInterferenceProjection(
            feature_dim=input_dim, num_qubits=quantum_qubits, phase_scale_init=0.7,
            entangle=True, hidden_mul=2, dropout=0.05
        )
        self.enable_bank = bool(enable_bank)
        if self.enable_bank:
            self.bank = nn.Parameter(torch.randn(bank_size, input_dim) * 0.02)
            self.bank_norm = nn.LayerNorm(input_dim)
            self.bank_mha = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
            self.bank_res_scale = nn.Parameter(torch.tensor(0.0))

    def _bank_prior(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = x.mean(dim=1, keepdim=True)                             # [B,1,D]
        bank_seq = self.bank.unsqueeze(0).expand(B, -1, -1)         # [B,M,D]
        attn_out, _ = self.bank_mha(q, bank_seq, bank_seq, need_weights=False)  # [B,1,D]
        prior = self.bank_norm(attn_out)
        return self.bank_res_scale.tanh() * prior                   # [B,1,D]

    def forward(self,
                x: Tensor,
                operation: str = 'read',
                lightbulb_intensity: float = 0.0,
                context: Optional[dict] = None) -> Tensor:
        z = self.embed(x)
        if self.use_entanglement:
            z = self.entangle(z)
        z = self.qip(z, intensity=lightbulb_intensity)
        if self.enable_bank:
            prior = self._bank_prior(z)               # [B,1,D]
            z = z + prior.expand(-1, z.size(1), -1)
        return z

    def encode_to_manifold(self, x: Tensor) -> Tensor:
        return x.unsqueeze(2)

    def detect_lightbulb_moment(self, query: Tensor) -> Tuple[bool, Tensor]:
        novelty = torch.sigmoid(query.mean(dim=-1, keepdim=True))
        return False, novelty

class EnhancedCGMNMemoryWithTransformerV2(nn.Module):
    def __init__(self, input_dim, cgmn_dim, mem_slots, slot_dim, n_transformer_layers, n_heads, attention_type):
        super().__init__()
        self.manifold_dim = cgmn_dim
        self.enc = nn.Linear(input_dim, input_dim)
    def forward(self, x, operation='read', importance=None, lightbulb_intensity=0.0, context=None):
        return x
    def manifold_projection(self, x):
        b, t, d = x.shape
        return x.new_zeros(b, t, self.manifold_dim * 3)
    def detect_geometric_insight(self, positions):
        similarity = torch.sigmoid(torch.randn(positions.shape[0], 1, device=positions.device))
        return False, similarity

class EnhancedCurvedMemoryWithTransformerV2(nn.Module):
    """
    Optional dilated TCN for V4.1; set use_tcn=False in V4 model.
    """
    def __init__(self, input_dim, hidden, curvature, slots, n_layers, n_heads, attention_type, use_tcn: bool = True):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden)
        self._act = nn.ReLU()
        self.back = nn.Linear(hidden, input_dim)
        self.use_tcn = bool(use_tcn)
        if self.use_tcn:
            self.dilated_convs = nn.ModuleList([
                nn.Conv1d(input_dim, hidden, kernel_size=3, dilation=2**i, padding=2**i)
                for i in range(3)
            ])
            self.temporal_attn = nn.MultiheadAttention(hidden, 4, batch_first=True)

    def forward(self, x, operation='read', importance=None, lightbulb_intensity=0.0, context=None):
        if not self.use_tcn:
            return x
        x_ch = x.permute(0, 2, 1)
        conv_outs = [conv(x_ch) for conv in self.dilated_convs]   # [B,H,T]
        combined = torch.stack(conv_outs, dim=-1).mean(dim=-1)    # [B,H,T]
        combined = combined.permute(0, 2, 1)                       # [B,T,H]
        attn_out, _ = self.temporal_attn(combined, combined, combined)  # [B,T,H]
        return self.back(self._act(attn_out))

    def detect_associative_chain(self, activation):
        spread = torch.sigmoid(activation.mean(dim=-1, keepdim=True))
        return False, spread
