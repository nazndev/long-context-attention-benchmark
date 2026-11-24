"""Single-process stand-ins for the Ulysses all-to-all."""

from __future__ import annotations

import torch

from spbench.errors import InvalidConfiguration


def all_to_all_sequence_to_heads(shards: list[torch.Tensor]) -> list[torch.Tensor]:
    """[B, S/P, H, D] per rank -> [B, S, H/P, D] per rank.

    Rank j receives head shard j from every rank and concatenates those pieces
    in group-rank order along the sequence.
    """

    degree = len(shards)
    if degree == 0:
        raise InvalidConfiguration("all-to-all needs at least one rank")
    head_count = shards[0].shape[2]
    if head_count % degree != 0:
        raise InvalidConfiguration(f"{head_count} heads are not divisible by Ulysses degree {degree}")
    parts = [tensor.chunk(degree, dim=2) for tensor in shards]
    return [torch.cat([parts[source][dest] for source in range(degree)], dim=1) for dest in range(degree)]


def all_to_all_heads_to_sequence(shards: list[torch.Tensor]) -> list[torch.Tensor]:
    """Inverse of ``all_to_all_sequence_to_heads``."""

    degree = len(shards)
    if degree == 0:
        raise InvalidConfiguration("all-to-all needs at least one rank")
    if shards[0].shape[1] % degree != 0:
        raise InvalidConfiguration("gathered sequence is not divisible by the Ulysses degree")
    parts = [tensor.chunk(degree, dim=1) for tensor in shards]
    return [torch.cat([parts[source][dest] for source in range(degree)], dim=2) for dest in range(degree)]
