# thms/store.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch

class IConsolidatedStore:
    """Interface for persistent memory storage supporting versioning and snapshots"""

    def write(self,
              series_id: str,
              content: torch.Tensor,
              tags: List[str],
              context: Dict[str, Any],
              importance: float = 0.6,
              confidence: float = 0.95,
              **kwargs) -> Optional[str]:
        raise NotImplementedError

    def read_latest(self, series_id: str) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def read_version(self, series_id: str, version: int) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def snapshot(self, label: str, metadata: Dict[str, Any]) -> str:
        raise NotImplementedError

    def restore_snapshot(self, snapshot_id: str) -> bool:
        raise NotImplementedError

    def flush_events(self) -> List[Any]:
        return []
