"""Token ownership for contiguous, zigzag, and USP placements."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spbench.errors import InvalidConfiguration


@dataclass(frozen=True)
class GroupMaps:
    world_size: int
    ulysses_degree: int
    ring_degree: int
    placement: str
    ulysses_groups: tuple[tuple[int, ...], ...]
    ring_groups: tuple[tuple[int, ...], ...]
    ulysses_rank: dict[int, int]
    ring_rank: dict[int, int]

    def groups_identical_to(self, other: GroupMaps) -> bool:
        return self.ulysses_groups == other.ulysses_groups and self.ring_groups == other.ring_groups


def _rank_index(groups: tuple[tuple[int, ...], ...]) -> dict[int, int]:
    index: dict[int, int] = {}
    for group in groups:
        for position, rank in enumerate(group):
            if rank in index:
                raise InvalidConfiguration(f"rank {rank} appears in more than one process group")
            index[rank] = position
    return index


def build_group_maps(world_size: int, ulysses_degree: int, ring_degree: int, placement: str) -> GroupMaps:
    """Reproduce YunChang sequence-parallel groups for one sequence-parallel job.

    Data-parallel degree is fixed at 1. A second data-parallel axis would change
    the comparison the assignment asks for, so it is rejected rather than implied.
    """

    if ulysses_degree < 1 or ring_degree < 1:
        raise InvalidConfiguration("Ulysses and ring degrees must be positive")
    if world_size != ulysses_degree * ring_degree:
        raise InvalidConfiguration(
            f"world_size {world_size} must equal ulysses_degree {ulysses_degree} * ring_degree {ring_degree}"
        )
    if placement == "ulysses_low":
        ulysses_groups = tuple(
            tuple(range(i * ulysses_degree, (i + 1) * ulysses_degree)) for i in range(ring_degree)
        )
        ring_groups = tuple(tuple(range(i, world_size, ulysses_degree)) for i in range(ulysses_degree))
    elif placement == "ring_low":
        ring_groups = tuple(tuple(range(i * ring_degree, (i + 1) * ring_degree)) for i in range(ulysses_degree))
        ulysses_groups = tuple(tuple(range(i, world_size, ring_degree)) for i in range(ring_degree))
    else:
        raise InvalidConfiguration(f"unknown USP placement {placement!r}")
    ulysses_rank = _rank_index(ulysses_groups)
    ring_rank = _rank_index(ring_groups)
    expected = set(range(world_size))
    if set(ulysses_rank) != expected or set(ring_rank) != expected:
        raise InvalidConfiguration("process groups do not cover every rank exactly once")
    return GroupMaps(
        world_size=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        placement=placement,
        ulysses_groups=ulysses_groups,
        ring_groups=ring_groups,
        ulysses_rank=ulysses_rank,
        ring_rank=ring_rank,
    )


def contiguous_positions(sequence_length: int, world_size: int, rank: int) -> list[int]:
    if world_size < 1 or not 0 <= rank < world_size:
        raise InvalidConfiguration("rank is outside the process group")
    if sequence_length % world_size != 0:
        raise InvalidConfiguration(
            f"contiguous layout requires sequence {sequence_length} divisible by world size {world_size}"
        )
    local = sequence_length // world_size
    start = rank * local
    return list(range(start, start + local))


def usp_positions(
    sequence_length: int,
    rank: int,
    ulysses_degree: int,
    ring_degree: int,
    placement: str,
    variant: str,
) -> list[int]:
    """Global token positions owned by one rank, in the order the rank computes them."""

    world_size = ulysses_degree * ring_degree
    if variant == "basic":
        # basic_extract_local chunks the global sequence by the global rank.
        return contiguous_positions(sequence_length, world_size, rank)
    if variant != "zigzag":
        raise InvalidConfiguration(f"unsupported layout variant {variant!r}")
    maps = build_group_maps(world_size, ulysses_degree, ring_degree, placement)
    if sequence_length % (2 * ring_degree) != 0:
        raise InvalidConfiguration(
            f"zigzag requires sequence {sequence_length} divisible by 2 * ring_degree ({2 * ring_degree})"
        )
    if sequence_length % world_size != 0:
        raise InvalidConfiguration(
            f"zigzag Ulysses split requires sequence {sequence_length} divisible by world size {world_size}"
        )
    chunk = sequence_length // (2 * ring_degree)
    ring_rank = maps.ring_rank[rank]
    ulysses_rank = maps.ulysses_rank[rank]
    first = list(range(ring_rank * chunk, (ring_rank + 1) * chunk))
    partner = 2 * ring_degree - ring_rank - 1
    second = list(range(partner * chunk, (partner + 1) * chunk))
    concatenated = first + second
    if len(concatenated) % ulysses_degree != 0:
        raise InvalidConfiguration("zigzag concatenation is not divisible by the Ulysses degree")
    local = len(concatenated) // ulysses_degree
    return concatenated[ulysses_rank * local : (ulysses_rank + 1) * local]


def zigzag_ring_positions(sequence_length: int, world_size: int, rank: int) -> list[int]:
    """Zigzag ownership for a pure ring (Ulysses degree 1)."""

    return usp_positions(
        sequence_length,
        rank,
        ulysses_degree=1,
        ring_degree=world_size,
        placement="ulysses_low",
        variant="zigzag",
    )


def basic_usp_preserves_causal_ring_order(ulysses_degree: int, ring_degree: int, placement: str) -> bool:
    """Return whether basic contiguous chunks stay in causal ring order.

    Basic ring attention treats every token on ring rank i as earlier than every
    token on ring rank i+1. That is true for the upstream default ``ulysses_low``
    groups when chunks follow the global rank. It is false for ``ring_low`` plus
    ``basic_extract_local``, which concatenates strided chunks. Those cases are
    rejected rather than run.
    """

    world_size = ulysses_degree * ring_degree
    maps = build_group_maps(world_size, ulysses_degree, ring_degree, placement)
    positions = [
        usp_positions(world_size, rank, ulysses_degree, ring_degree, placement, "basic")
        for rank in range(world_size)
    ]
    member_of = {}
    for group in maps.ulysses_groups:
        for rank in group:
            member_of[rank] = group
    for group in maps.ring_groups:
        order: list[int] = []
        for rank in group:
            for member in member_of[rank]:
                order.extend(positions[member])
        if order != list(range(world_size)):
            return False
    return True


def positions_are_bijective(per_rank: list[list[int]], sequence_length: int) -> bool:
    flat = [position for rank_positions in per_rank for position in rank_positions]
    return sorted(flat) == list(range(sequence_length)) and len(flat) == sequence_length


def take_positions(tensor: torch.Tensor, positions: list[int]) -> torch.Tensor:
    """Differentiable gather along the sequence dimension."""

    index = torch.as_tensor(positions, dtype=torch.long, device=tensor.device)
    return tensor.index_select(1, index)


def assemble(pieces: list[torch.Tensor], positions: list[list[int]], sequence_length: int) -> torch.Tensor:
    """Invert a bijective shard map. The sequence axis of ``pieces`` is concatenated."""

    if len(pieces) != len(positions):
        raise InvalidConfiguration("each shard needs its position list")
    concatenated_positions: list[int] = []
    for rank_positions in positions:
        concatenated_positions.extend(rank_positions)
    if not positions_are_bijective(positions, sequence_length):
        raise InvalidConfiguration("refusing to assemble a shard map that is not bijective")
    concatenated = torch.cat(pieces, dim=1)
    location = torch.empty(sequence_length, dtype=torch.long, device=concatenated.device)
    source = torch.as_tensor(concatenated_positions, dtype=torch.long, device=concatenated.device)
    location[source] = torch.arange(sequence_length, device=concatenated.device)
    return concatenated.index_select(1, location)
