# thms/nfm.py
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
Tensor = torch.Tensor

class EnhancedNeverForgettingMemory:
    def __init__(self, base_path="./enhanced_memory_db", enable_compression=True,
                 enable_indexing=True, enable_relationships=True, vector_dim=512):
        self.vector_dim = vector_dim

class NeverForgettingMemoryAdapter(nn.Module):
    def __init__(self, core: EnhancedNeverForgettingMemory, input_dim: int, vector_dim: int = 512):
        super().__init__()
        self.core = core
        self.proj = nn.Linear(input_dim, vector_dim)

    @torch.no_grad()
    def query_by_vectors(self, x: Tensor, topk: int = 6) -> Tuple[Tensor, Tensor]:
        b, d = x.shape
        return x.new_zeros(b, topk, d), x.new_zeros(b, topk, dtype=torch.long)
