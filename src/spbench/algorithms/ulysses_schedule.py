"""CPU model of Ulysses attention."""

from __future__ import annotations

import torch

from spbench.algorithms.collectives import all_to_all_heads_to_sequence, all_to_all_sequence_to_heads
from spbench.algorithms.result import ScheduleResult
from spbench.attention_math import geometric_pairs, reference_attention
from spbench.errors import InvalidConfiguration
from spbench.layouts import assemble, contiguous_positions, take_positions


def ulysses_attention_global(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    world_size: int,
    causal: bool = True,
) -> ScheduleResult:
    if query.shape[2] % world_size != 0:
        raise InvalidConfiguration(
            f"Ulysses requires {query.shape[2]} heads to be divisible by world size {world_size}"
        )
    sequence_length = query.shape[1]
    positions = [contiguous_positions(sequence_length, world_size, rank) for rank in range(world_size)]
    local_q = [take_positions(query, index) for index in positions]
    local_k = [take_positions(key, index) for index in positions]
    local_v = [take_positions(value, index) for index in positions]
    gathered_q = all_to_all_sequence_to_heads(local_q)
    gathered_k = all_to_all_sequence_to_heads(local_k)
    gathered_v = all_to_all_sequence_to_heads(local_v)
    local_attention = [
        reference_attention(q_shard, k_shard, v_shard, causal=causal)
        for q_shard, k_shard, v_shard in zip(gathered_q, gathered_k, gathered_v, strict=True)
    ]
    restored = all_to_all_heads_to_sequence(local_attention)
    heads_local = query.shape[2] // world_size
    # Every rank executes the full causal (or dense) triangle for each of its heads.
    per_head = geometric_pairs(sequence_length, sequence_length, causal)
    return ScheduleResult(
        output=assemble(restored, positions, sequence_length),
        executed_pairs_per_head=[per_head] * world_size,
        heads_on_rank=[heads_local] * world_size,
        query_positions=positions,
    )
