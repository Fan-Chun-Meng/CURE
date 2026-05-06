from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .utils import safe_normalize


@dataclass
class MemoryEntry:
    feature: torch.Tensor
    pseudo_label: int
    reweighted_entropy: float
    image: Optional[torch.Tensor] = None
    reliability: float = 1.0
    stability: float = 1.0
    long_score: float = float("inf")
    grad_vector: Optional[torch.Tensor] = None


class DualMemoryBuffer:
    def __init__(self, num_classes: int, feature_dim: int, short_capacity: int, long_capacity: int) -> None:
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.short_capacity = short_capacity
        self.long_capacity = long_capacity
        self.short_memory: Dict[int, List[MemoryEntry]] = defaultdict(list)
        self.long_memory: Dict[int, List[MemoryEntry]] = defaultdict(list)
        self.short_last_update: Dict[int, int] = defaultdict(int)
        self.long_last_update: Dict[int, int] = defaultdict(int)

    def push_short(self, entry: MemoryEntry, step: int) -> None:
        if self.short_capacity <= 0:
            return
        queue = self.short_memory[entry.pseudo_label]
        if len(queue) < self.short_capacity:
            queue.append(entry)
        else:
            worst_idx = max(range(len(queue)), key=lambda idx: queue[idx].reweighted_entropy)
            queue[worst_idx] = entry
        self.short_last_update[entry.pseudo_label] = step

    def push_long(self, entry: MemoryEntry, step: int) -> None:
        if self.long_capacity <= 0:
            return
        queue = self.long_memory[entry.pseudo_label]
        if len(queue) < self.long_capacity:
            queue.append(entry)
        else:
            worst_idx = max(range(len(queue)), key=lambda idx: queue[idx].long_score)
            queue[worst_idx] = entry
        self.long_last_update[entry.pseudo_label] = step

    def get_short_samples(self) -> List[MemoryEntry]:
        return [item for items in self.short_memory.values() for item in items]

    def select_recent_classes(self, memory_name: str, max_classes: int) -> List[int]:
        if memory_name == "short":
            memory_source = self.short_memory
            update_source = self.short_last_update
        elif memory_name == "long":
            memory_source = self.long_memory
            update_source = self.long_last_update
        else:
            raise ValueError(f"Unknown memory name: {memory_name}")

        active_classes = [cls_idx for cls_idx, items in memory_source.items() if items]
        active_classes.sort(key=lambda cls_idx: update_source.get(cls_idx, 0), reverse=True)
        if max_classes > 0:
            active_classes = active_classes[:max_classes]
        return active_classes

    def prototypes(self, memory_name: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if memory_name == "short":
            source = self.short_memory
        elif memory_name == "long":
            source = self.long_memory
        else:
            raise ValueError(f"Unknown memory name: {memory_name}")

        prototypes = []
        valid_mask = []
        for cls_idx in range(self.num_classes):
            items = source.get(cls_idx, [])
            if items:
                feats = torch.stack([item.feature for item in items], dim=0).to(device)
                prototype = safe_normalize(feats.mean(dim=0), dim=0)
                valid_mask.append(True)
            else:
                prototype = torch.zeros(self.feature_dim, device=device)
                valid_mask.append(False)
            prototypes.append(prototype)

        return torch.stack(prototypes, dim=0), torch.tensor(valid_mask, device=device, dtype=torch.bool)
