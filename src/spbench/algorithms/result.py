"""Result of a CPU schedule check."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ScheduleResult:
    output: torch.Tensor
    executed_pairs_per_head: list[int]
    heads_on_rank: list[int]
    query_positions: list[list[int]]

    def total_executed_pairs(self) -> int:
        return sum(
            pairs * heads for pairs, heads in zip(self.executed_pairs_per_head, self.heads_on_rank, strict=True)
        )
