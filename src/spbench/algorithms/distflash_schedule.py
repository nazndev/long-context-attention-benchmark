"""CPU reconstruction of the DistFlashAttn forward schedule."""

from __future__ import annotations

import torch

from spbench.algorithms.result import ScheduleResult
from spbench.attention_math import block_attention, geometric_pairs, merge_statistics
from spbench.errors import InvalidConfiguration
from spbench.layouts import assemble, contiguous_positions, take_positions


def _is_local(rank: int, time_step: int, world_size: int) -> bool:
    return rank >= min(world_size // 2, time_step)


def _is_idle(rank: int, time_step: int, world_size: int) -> bool:
    return rank < (world_size // 2) and time_step == world_size // 2


def _is_sync(rank: int, time_step: int, world_size: int) -> bool:
    return rank > max(world_size // 2, world_size - time_step)


def distflash_attention_global(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    world_size: int,
    causal: bool = True,
) -> ScheduleResult:
    if not causal:
        raise InvalidConfiguration(
            "The audited DistFlashAttn loop is a causal load-balanced schedule. "
            "Noncausal mode is not available from this control flow."
        )
    if world_size < 2:
        raise InvalidConfiguration("DistFlashAttn is a distributed method and needs at least 2 ranks")
    if world_size % 2 != 0:
        raise InvalidConfiguration(
            "LightSeq maybe_send_recv_fwd_qkvo at commit 8f486dad says it only supports an even "
            "number of machines. The P//2 loop does not cover every causal KV block when P is odd, "
            "so odd world sizes are rejected instead of being relabeled as a working schedule."
        )
    sequence_length = query.shape[1]
    positions = [contiguous_positions(sequence_length, world_size, rank) for rank in range(world_size)]
    query_shards = [take_positions(query, index) for index in positions]
    key_shards = [take_positions(key, index) for index in positions]
    value_shards = [take_positions(value, index) for index in positions]
    peer_q: list[list[torch.Tensor | None]] = [[None, None] for _ in range(world_size)]
    peer_k: list[list[torch.Tensor | None]] = [[None, None] for _ in range(world_size)]
    peer_v: list[list[torch.Tensor | None]] = [[None, None] for _ in range(world_size)]
    peer_o: list[list[torch.Tensor | None]] = [[None, None] for _ in range(world_size)]
    peer_m: list[list[torch.Tensor | None]] = [[None, None] for _ in range(world_size)]
    peer_l: list[list[torch.Tensor | None]] = [[None, None] for _ in range(world_size)]
    state: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [None] * world_size
    executed = [0] * world_size

    for time_step in range(world_size // 2 + 1):
        current = time_step % 2
        previous = (time_step - 1) % 2
        incoming_q: list[torch.Tensor | None] = [None] * world_size
        incoming_k: list[torch.Tensor | None] = [None] * world_size
        incoming_v: list[torch.Tensor | None] = [None] * world_size
        for rank in range(world_size):
            source = rank - (time_step + 1)
            if source < 0:
                if time_step < (world_size // 2 - 1):
                    incoming_q[rank] = query_shards[source % world_size]
            else:
                incoming_k[rank] = key_shards[source]
                incoming_v[rank] = value_shards[source]
        for rank in range(world_size):
            peer_q[rank][current] = incoming_q[rank]
            peer_k[rank][current] = incoming_k[rank]
            peer_v[rank][current] = incoming_v[rank]

        outgoing = []
        for rank in range(world_size):
            destination_offset = rank - (time_step - 1)
            if destination_offset < 0 and time_step > 1:
                output = peer_o[rank][current]
                row_max = peer_m[rank][current]
                row_sum = peer_l[rank][current]
                if output is None or row_max is None or row_sum is None:
                    raise InvalidConfiguration(
                        f"DistFlashAttn rank {rank} would send empty partials at step {time_step}"
                    )
                outgoing.append((destination_offset % world_size, output, row_max, row_sum))
        received_o: list[torch.Tensor | None] = [None] * world_size
        received_m: list[torch.Tensor | None] = [None] * world_size
        received_l: list[torch.Tensor | None] = [None] * world_size
        for destination, output, row_max, row_sum in outgoing:
            received_o[destination] = output
            received_m[destination] = row_max
            received_l[destination] = row_sum
        for rank in range(world_size):
            if received_o[rank] is not None:
                peer_o[rank][current] = received_o[rank]
                peer_m[rank][current] = received_m[rank]
                peer_l[rank][current] = received_l[rank]

        for rank in range(world_size):
            if _is_local(rank, time_step, world_size):
                if time_step == 0:
                    block = block_attention(query_shards[rank], key_shards[rank], value_shards[rank], causal=True)
                    executed[rank] += geometric_pairs(query_shards[rank].shape[1], key_shards[rank].shape[1], True)
                    state[rank] = block
                else:
                    remote_k = peer_k[rank][previous]
                    remote_v = peer_v[rank][previous]
                    if remote_k is None or remote_v is None:
                        raise InvalidConfiguration(
                            f"DistFlashAttn rank {rank} is missing KV at step {time_step}"
                        )
                    block = block_attention(query_shards[rank], remote_k, remote_v, causal=False)
                    executed[rank] += geometric_pairs(query_shards[rank].shape[1], remote_k.shape[1], False)
                    if state[rank] is None:
                        raise InvalidConfiguration(f"DistFlashAttn rank {rank} has no local state to update")
                    state[rank] = merge_statistics(state[rank], block)
            elif _is_idle(rank, time_step, world_size):
                continue
            else:
                remote_q = peer_q[rank][previous]
                if remote_q is None:
                    raise InvalidConfiguration(
                        f"DistFlashAttn helper rank {rank} is missing a remote query at step {time_step}"
                    )
                block_output, block_max, block_sum = block_attention(
                    remote_q, key_shards[rank], value_shards[rank], causal=False
                )
                executed[rank] += geometric_pairs(remote_q.shape[1], key_shards[rank].shape[1], False)
                peer_o[rank][previous] = block_output
                peer_m[rank][previous] = block_max
                peer_l[rank][previous] = block_sum

        for rank in range(world_size):
            if _is_sync(rank, time_step, world_size):
                partial_o = peer_o[rank][current]
                partial_m = peer_m[rank][current]
                partial_l = peer_l[rank][current]
                if partial_o is None or partial_m is None or partial_l is None or state[rank] is None:
                    raise InvalidConfiguration(
                        f"DistFlashAttn rank {rank} cannot rescale at step {time_step}"
                    )
                state[rank] = merge_statistics(state[rank], (partial_o, partial_m, partial_l))

    if any(item is None for item in state):
        raise InvalidConfiguration("DistFlashAttn schedule left a rank without an output")
    heads = query.shape[2]
    return ScheduleResult(
        output=assemble([item[0] for item in state if item is not None], positions, sequence_length),
        executed_pairs_per_head=executed,
        heads_on_rank=[heads] * world_size,
        query_positions=positions,
    )
