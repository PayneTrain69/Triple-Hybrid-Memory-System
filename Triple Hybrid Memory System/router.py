# thms/router.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
Tensor = torch.Tensor

class HybridRouterV2(nn.Module):
    """Adaptive Subsystem Router with Attention + Gate."""
    def __init__(self, input_dim: int, n_subsystems: int, n_heads: int = 8):
        super().__init__()
        self.n = n_subsystems
        self.mha = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
        self.ctx_proj = nn.Linear(input_dim, input_dim)
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, n_subsystems * 4),
            nn.GELU(),
            nn.Linear(n_subsystems * 4, n_subsystems),
            nn.Softmax(dim=-1)
        )
        self.skip_scaler = nn.Parameter(torch.ones(1))

    def forward(self, tokens: Tensor, context: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, N, D = tokens.shape
        assert N == self.n, f"Expected N={self.n} subsystems, got {N}"
        attn_out, _ = self.mha(tokens, tokens, tokens, need_weights=False)  # [B,N,D]
        ctx = tokens.mean(dim=1) if context is None else context
        ctx = torch.tanh(self.ctx_proj(ctx))
        weights = self.gate_network(ctx)          # [B,N]
        refined = tokens + self.skip_scaler * attn_out
        return weights, refined
