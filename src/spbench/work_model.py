"""Analytical causal-work model."""

from __future__ import annotations

from statistics import pstdev

from spbench.errors import InvalidConfiguration
from spbench.layouts import usp_positions, zigzag_ring_positions


def full_causal_pairs(sequence_length: int) -> int:
    if sequence_length < 1:
        raise InvalidConfiguration("sequence length must be positive")
    return sequence_length * (sequence_length + 1) // 2


def contiguous_valid_pairs(sequence_length: int, world_size: int, rank: int) -> int:
    """Valid causal pairs for contiguous rank ``rank``.

    For local length m = S/P and zero-based rank r, the assignment specifies
    m * (2*r*m + m + 1) / 2 pairs per head and per batch element.
    """

    if sequence_length % world_size != 0:
        raise InvalidConfiguration("contiguous work model requires S divisible by P")
    local = sequence_length // world_size
    numerator = local * (2 * rank * local + local + 1)
    if numerator % 2 != 0:
        raise InvalidConfiguration("contiguous pair formula was not an integer; the derivation is wrong")
    return numerator // 2


def pairs_for_query_positions(query_positions: list[int]) -> int:
    """Pairs owned by these queries. Query p attends to keys 0..p inclusive."""

    return sum(position + 1 for position in query_positions)


def query_owned_pairs(
    sequence_length: int,
    world_size: int,
    rank: int,
    variant: str,
    ulysses_degree: int | None = None,
    ring_degree: int | None = None,
    placement: str = "ulysses_low",
) -> int:
    if variant == "contiguous":
        return contiguous_valid_pairs(sequence_length, world_size, rank)
    if variant == "zigzag" and ulysses_degree in (None, 1):
        return pairs_for_query_positions(zigzag_ring_positions(sequence_length, world_size, rank))
    if variant == "zigzag":
        if ulysses_degree is None or ring_degree is None:
            raise InvalidConfiguration("USP zigzag work needs both degrees")
        positions = usp_positions(
            sequence_length, rank, ulysses_degree, ring_degree, placement, "zigzag"
        )
        return pairs_for_query_positions(positions)
    if variant == "ulysses":
        # After the all-to-all, every rank computes the full causal triangle for
        # its heads. Per head, the pair count does not depend on the rank.
        if sequence_length % world_size != 0:
            raise InvalidConfiguration("Ulysses requires S divisible by P")
        return full_causal_pairs(sequence_length)
    raise InvalidConfiguration(f"unknown work variant {variant!r}")


def ulysses_heads_on_rank(num_heads: int, ulysses_degree: int) -> int:
    if num_heads % ulysses_degree != 0:
        raise InvalidConfiguration(
            f"{num_heads} heads are not divisible by Ulysses degree {ulysses_degree}"
        )
    return num_heads // ulysses_degree


def ring_schedule_idle_fraction(world_size: int) -> float:
    """Idle slot fraction of unbalanced causal ring scheduling.

    DistFlashAttn equation in Section 3.2: (P^2 - P) / (2 P^2), which tends
    to 1/2 as P grows. This counts skipped future blocks in a P-step ring,
    not a measured GPU bubble.
    """

    if world_size < 1:
        raise InvalidConfiguration("world size must be positive")
    return (world_size * world_size - world_size) / (2 * world_size * world_size)


def distflash_paper_idle_fraction(world_size: int) -> float:
    """Published load-balanced idle fraction, arXiv:2310.03294 v2 equation (2).

    0 when P is odd and 1/(2P) when P is even. The August 2024 LightSeq
    ``is_idle`` predicate does not use this denominator. See the upstream audit.
    Do not treat the two quantities as the same measurement.
    """

    if world_size < 1:
        raise InvalidConfiguration("world size must be positive")
    if world_size % 2 == 1:
        return 0.0
    return 1.0 / (2 * world_size)


def lightseq_is_idle(rank: int, time_step: int, world_size: int) -> bool:
    """Literal ``is_idle`` from LightSeq async_communication.py at 8f486dad."""

    return rank < (world_size // 2) and time_step == world_size // 2


def lightseq_idle_slot_fraction(world_size: int) -> float:
    """Fraction of (rank, time_step) slots where the audited ``is_idle`` is true.

    The loop runs ``world_size // 2 + 1`` steps. This is a reading of the code,
    not equation (2) and not a profile.
    """

    steps = world_size // 2 + 1
    idle_slots = sum(
        1
        for rank in range(world_size)
        for time_step in range(steps)
        if lightseq_is_idle(rank, time_step, world_size)
    )
    return idle_slots / (world_size * steps)


def max_over_mean(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    return max(values) / mean


def coefficient_of_variation(values: list[float]) -> float | None:
    """Population coefficient of variation. The rank set is the whole population."""

    if not values:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    return pstdev(values) / mean


def imbalance_ratios(values: list[float]) -> dict[str, float | None]:
    return {
        "max_over_mean": max_over_mean(values),
        "coefficient_of_variation": coefficient_of_variation(values),
    }


def contiguous_rank_pairs(sequence_length: int, world_size: int) -> list[int]:
    return [contiguous_valid_pairs(sequence_length, world_size, rank) for rank in range(world_size)]


def explain_not_constant_flops(tokens_per_gpu: int, gpu_count: int) -> str:
    global_length = tokens_per_gpu * gpu_count
    return (
        f"Weak-context scaling holds {tokens_per_gpu} tokens on each GPU and grows the global "
        f"length to {global_length} at {gpu_count} GPUs. Dense causal attention on that global "
        "length still performs a quadratic number of query-key interactions. Those interactions "
        "are divided across GPUs, but each GPU's query block attends to a growing prefix, so "
        "FLOPs per GPU are not constant."
    )


def ulysses_communication_elements(sequence_length: int, heads: int, head_dim: int, world_size: int, qkv_tensors: int = 3) -> float | None:
    """Elements moved by one all-to-all of a sequence-sharded Q, K, or V tensor, per rank.

    A rank holds S/P * H * D elements and sends the fraction (P-1)/P of them.
    Forward Ulysses does this for Q, K, and V, then once more for the output.
    When S grows in proportion to P, per-rank volume stays flat. That is the
    Ulysses paper's constant-volume claim. It is arithmetic, not a measurement.
    """

    if world_size < 1 or sequence_length % world_size != 0 or heads % world_size != 0:
        return None
    local_elements = (sequence_length // world_size) * heads * head_dim
    per_tensor = local_elements * (world_size - 1) / world_size
    return qkv_tensors * per_tensor


def ring_forward_kv_elements(sequence_length: int, heads: int, head_dim: int, world_size: int) -> float | None:
    """Elements a rank sends during a causal-unaware count of KV ring steps.

    Each of K and V is length S/P. A full ring sends that block to the neighbor
    once per step for P-1 steps. Causal schedules skip some compute but the
    basic ring-flash implementation still circulates KV every step. This count
    follows that circulation, not a profile.
    """

    if world_size < 2 or sequence_length % world_size != 0:
        return None
    block = (sequence_length // world_size) * heads * head_dim
    return 2 * block * (world_size - 1)

