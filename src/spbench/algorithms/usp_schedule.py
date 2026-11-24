"""CPU model of Unified Sequence Parallelism."""

from __future__ import annotations

import torch

from spbench.algorithms.collectives import all_to_all_heads_to_sequence, all_to_all_sequence_to_heads
from spbench.algorithms.result import ScheduleResult
from spbench.algorithms.ring_schedule import ring_attention
from spbench.errors import InvalidConfiguration
from spbench.layouts import assemble, build_group_maps, take_positions, usp_positions


def usp_attention_global(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ulysses_degree: int,
    ring_degree: int,
    variant: str,
    placement: str,
    causal: bool = True,
) -> ScheduleResult:
    world_size = ulysses_degree * ring_degree
    if query.shape[2] % ulysses_degree != 0:
        raise InvalidConfiguration(
            f"USP requires {query.shape[2]} heads to be divisible by Ulysses degree {ulysses_degree}"
        )
    maps = build_group_maps(world_size, ulysses_degree, ring_degree, placement)
    sequence_length = query.shape[1]
    positions = [
        usp_positions(sequence_length, rank, ulysses_degree, ring_degree, placement, variant)
        for rank in range(world_size)
    ]
    local = {
        "q": [take_positions(query, index) for index in positions],
        "k": [take_positions(key, index) for index in positions],
        "v": [take_positions(value, index) for index in positions],
    }
    gathered = {"q": [None] * world_size, "k": [None] * world_size, "v": [None] * world_size}
    for group in maps.ulysses_groups:
        for name in ("q", "k", "v"):
            exchanged = all_to_all_sequence_to_heads([local[name][rank] for rank in group])
            for index, rank in enumerate(group):
                gathered[name][rank] = exchanged[index]
    ring_output: list[torch.Tensor | None] = [None] * world_size
    executed = [0] * world_size
    for group in maps.ring_groups:
        ring_variant = variant
        if ring_degree == 1:
            # One ring rank has no KV circulation. Zigzag ownership, if any,
            # was applied by the extract; the single local attention is basic.
            ring_variant = "basic"
        outputs, pairs = ring_attention(
            [gathered["q"][rank] for rank in group],
            [gathered["k"][rank] for rank in group],
            [gathered["v"][rank] for rank in group],
            ring_variant,
            causal,
        )
        for index, rank in enumerate(group):
            ring_output[rank] = outputs[index]
            executed[rank] = pairs[index]
    restored: list[torch.Tensor | None] = [None] * world_size
    for group in maps.ulysses_groups:
        exchanged = all_to_all_heads_to_sequence([ring_output[rank] for rank in group])
        for index, rank in enumerate(group):
            restored[rank] = exchanged[index]
    if any(tensor is None for tensor in restored):
        raise InvalidConfiguration("USP schedule lost a rank output")
    heads_local = query.shape[2] // ulysses_degree
    return ScheduleResult(
        output=assemble([tensor for tensor in restored if tensor is not None], positions, sequence_length),
        executed_pairs_per_head=executed,
        heads_on_rank=[heads_local] * world_size,
        query_positions=positions,
    )
