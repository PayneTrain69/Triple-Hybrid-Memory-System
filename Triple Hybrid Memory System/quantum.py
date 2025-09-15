# thms/quantum.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor

class QuantumEntanglementLayer(nn.Module):
    """
    Real-valued, shape-stable entanglement pre-processor.
    x: [B,T,D] -> [B,T,D]
    """
    def __init__(self, dim: int, num_qubits: int = 8):
        super().__init__()
        self.qubits = nn.Parameter(torch.randn(num_qubits, dim) * 0.02)  # [Q,D]
        self.phase_mod = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.out_proj = nn.Linear(dim, dim)
        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        states = torch.einsum('btd,qd->bqt', x, self.qubits)     # [B,Q,T]
        states_mix = torch.einsum('bqt,qd->btd', states, self.qubits)  # [B,T,D]
        mod = self.phase_mod(states_mix)
        y = self.out_proj(states_mix * mod)
        return x + self.res_scale.tanh() * (y - x)

class QuantumInterferenceProjection(nn.Module):
    """
    QIP: lightbulb-intensity sensitive projection with optional entanglement.
    """
    def __init__(self, feature_dim: int, num_qubits: int = 8,
                 phase_scale_init: float = 0.7, entangle: bool = True,
                 hidden_mul: int = 2, dropout: float = 0.05):
        super().__init__()
        self.D = int(feature_dim)
        self.Q = int(num_qubits)

        self.to_qubits = nn.Linear(self.D, self.Q, bias=False)
        self.mix_to_features = nn.Linear(self.Q, self.D, bias=False)

        self.entangle = entangle
        if entangle:
            self.entanglement_layer = nn.Sequential(
                nn.Linear(self.D * 2, self.D * hidden_mul),
                nn.GELU(),
                nn.Linear(self.D * hidden_mul, self.D),
                nn.Tanh(),
            )
            self.dropout = nn.Dropout(dropout)
        else:
            self.register_module("entanglement_layer", None)
            self.dropout = nn.Identity()

        self.res_scale = nn.Parameter(torch.tensor(0.0))
        self.phase_scale = nn.Parameter(torch.tensor(float(phase_scale_init)))

    def forward(self, x: Tensor, intensity: float | Tensor = 0.0) -> Tensor:
        squeeze_time = False
        if x.dim() == 2:
            x = x.unsqueeze(1); squeeze_time = True

        if not torch.is_tensor(intensity):
            intensity = torch.tensor(float(intensity), device=x.device)

        x32 = x.float()
        alpha_eff = self.phase_scale * (1.0 + 0.75 * torch.sigmoid(3.0 * (intensity - 0.5)))

        phi = self.to_qubits(x32)                      # [B,T,Q]
        theta = self.mix_to_features(phi)              # [B,T,D]
        gate = torch.cos(alpha_eff * theta)            # [B,T,D]

        if self.entangle:
            concat = torch.cat([x32, x32 * gate], dim=-1)
            ent_gate = self.dropout(self.entanglement_layer(concat))
            y = x32 * (0.5 * (ent_gate + 1.0))
        else:
            y = x32 * gate

        out = x + self.res_scale.tanh() * (y.to(x.dtype) - x)
        if squeeze_time:
            out = out.squeeze(1)
        return out
