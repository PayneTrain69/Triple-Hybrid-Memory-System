# thms/consolidation_v2.py
from __future__ import annotations
from typing import Deque, Optional
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor

class MemoryConsolidationManagerV2(nn.Module):
    """
    Hierarchical/topology-aware memory consolidation.
    consolidate: [B,T,D_in] x [B,1] -> [B,D_topo]
    recall: [B,T,D_in]|[B,D_in] -> [B,K,D_topo]
    """
    def __init__(self, embedding_dim: int, topological_dim: int = 128, min_keep: int = 1, max_keep: Optional[int] = None):
        super().__init__()
        self.topo_dim = int(topological_dim)
        self.topological_map = nn.Sequential(
            nn.Linear(embedding_dim, self.topo_dim),
            nn.LeakyReLU(),
            nn.Linear(self.topo_dim, self.topo_dim),
            nn.Tanh(),
        )
        self.rehearsal_buffer: Deque[Tensor] = deque(maxlen=1000)
        self.min_keep = int(min_keep)
        self.max_keep = max_keep

    @torch.no_grad()
    def _batch_topk_mask(self, distances: Tensor, k_per_row: Tensor) -> Tensor:
        B, T = distances.shape
        vals, _ = torch.sort(distances, dim=1)
        idx_row = torch.arange(B, device=distances.device)
        kth_idx = (k_per_row.clamp(1, T) - 1).long()
        thresh = vals[idx_row, kth_idx]
        mask = distances <= thresh.unsqueeze(1)
        return mask

    def consolidate(self, memories: Tensor, importance: Tensor) -> Tensor:
        B, T, D = memories.shape
        topo_proj = self.topological_map(memories)                  # [B,T,D_topo]
        centroids = topo_proj.mean(dim=1, keepdim=True)             # [B,1,D_topo]
        dists = torch.cdist(topo_proj, centroids).squeeze(-1)       # [B,T]
        max_keep = self.max_keep or T
        k = (importance.squeeze(-1) * (max_keep - self.min_keep) + self.min_keep).round().long()
        k = k.clamp(min=self.min_keep, max=max_keep)
        mask = self._batch_topk_mask(dists, k)
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        retained = (topo_proj * mask.unsqueeze(-1)).sum(dim=1) / counts  # [B,D_topo]
        self.rehearsal_buffer.append(retained.detach())
        return retained

    def recall(self, query: Tensor, topk: int = 1) -> Tensor:
        if len(self.rehearsal_buffer) == 0:
            if query.dim() == 3: B, _, _ = query.shape
            else: B, _ = query.shape
            return query.new_zeros(B, 0, self.topo_dim)
        q = query.mean(dim=1) if query.dim() == 3 else query
        q_proj = self.topological_map(q)                             # [B,D_topo]
        bank = torch.stack(list(self.rehearsal_buffer), dim=1)       # [B,Kbuf,D_topo]
        Kbuf = bank.size(1)
        K = min(topk, Kbuf)
        sim = F.cosine_similarity(q_proj.unsqueeze(1), bank, dim=-1) # [B,Kbuf]
        _, top_idx = torch.topk(sim, k=K, dim=1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, bank.size(-1))
        selected = torch.gather(bank, dim=1, index=gather_idx)       # [B,K,D_topo]
        return selected
