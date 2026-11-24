"""CPU model of ring-flash-attention's basic and zigzag schedules."""

from __future__ import annotations

import torch

from spbench.algorithms.result import ScheduleResult
from spbench.attention_math import block_attention, geometric_pairs, merge_query_slice, merge_statistics
from spbench.errors import InvalidConfiguration
from spbench.layouts import assemble, contiguous_positions, take_positions, zigzag_ring_positions


def ring_attention(
    query_shards: list[torch.Tensor],
    key_shards: list[torch.Tensor],
    value_shards: list[torch.Tensor],
    variant: str,
    causal: bool,
) -> tuple[list[torch.Tensor], list[int]]:
    world_size = len(query_shards)
    if world_size < 1 or len(key_shards) != world_size or len(value_shards) != world_size:
        raise InvalidConfiguration("ring shards must be one list per tensor, one entry per rank")
    if variant == "zigzag" and not causal:
        raise InvalidConfiguration(
            "zigzag_ring_flash_attn_forward asserts causal == True; noncausal zigzag is unavailable"
        )
    if variant not in {"basic", "zigzag"}:
        raise InvalidConfiguration(f"unknown ring variant {variant!r}")
    state: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [None] * world_size
    executed = [0] * world_size
    held_k = list(key_shards)
    held_v = list(value_shards)
    for step in range(world_size):
        next_k = [held_k[(rank - 1) % world_size] for rank in range(world_size)]
        next_v = [held_v[(rank - 1) % world_size] for rank in range(world_size)]
        for rank in range(world_size):
            query = query_shards[rank]
            key = held_k[rank]
            value = held_v[rank]
            if variant == "basic":
                if (not causal) or step <= rank:
                    block_causal = bool(causal and step == 0)
                    block = block_attention(query, key, value, causal=block_causal)
                    executed[rank] += geometric_pairs(query.shape[1], key.shape[1], block_causal)
                    state[rank] = block if state[rank] is None else merge_statistics(state[rank], block)
            else:
                half = query.shape[1] // 2
                if query.shape[1] % 2 != 0:
                    raise InvalidConfiguration("zigzag local sequence must be even")
                if step == 0:
                    block = block_attention(query, key, value, causal=True)
                    executed[rank] += geometric_pairs(query.shape[1], key.shape[1], True)
                    state[rank] = block
                elif step <= rank:
                    block = block_attention(query, key[:, :half], value[:, :half], causal=False)
                    executed[rank] += geometric_pairs(query.shape[1], half, False)
                    state[rank] = merge_statistics(state[rank], block)
                else:
                    block = block_attention(query[:, half:], key, value, causal=False)
                    executed[rank] += geometric_pairs(half, key.shape[1], False)
                    state[rank] = merge_query_slice(state[rank], block, half)
        held_k, held_v = next_k, next_v
    if any(item is None for item in state):
        raise InvalidConfiguration("ring schedule produced no output for a rank")
    return [item[0] for item in state], executed


def ring_attention_global(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    world_size: int,
    variant: str,
    causal: bool = True,
) -> ScheduleResult:
    sequence_length = query.shape[1]
    if variant == "basic":
        positions = [contiguous_positions(sequence_length, world_size, rank) for rank in range(world_size)]
    elif variant == "zigzag":
        positions = [zigzag_ring_positions(sequence_length, world_size, rank) for rank in range(world_size)]
    else:
        raise InvalidConfiguration(f"unknown ring variant {variant!r}")
    outputs, executed = ring_attention(
        [take_positions(query, index) for index in positions],
        [take_positions(key, index) for index in positions],
        [take_positions(value, index) for index in positions],
        variant,
        causal,
    )
    heads = query.shape[2]
    return ScheduleResult(
        output=assemble(outputs, positions, sequence_length),
        executed_pairs_per_head=executed,
        heads_on_rank=[heads] * world_size,
        query_positions=positions,
    )
