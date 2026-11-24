"""Deterministic, position-addressed Q, K, V, and dO."""

from __future__ import annotations

import torch

_KINDS = {"q": 1, "k": 2, "v": 3, "do": 4}


def _mix(seed: int, kind: str, position: int) -> int:
    return (seed * 1_000_003 + _KINDS[kind] * 100_003 + position * 97) % (2**63 - 1)


def generate_bshd(
    positions: list[int] | range,
    batch_size: int,
    heads: int,
    head_dim: int,
    seed: int,
    kind: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if kind not in _KINDS:
        raise ValueError(f"unknown tensor kind {kind}")
    rows = []
    generator = torch.Generator(device="cpu")
    for position in positions:
        generator.manual_seed(_mix(seed, kind, int(position)))
        rows.append(torch.randn(batch_size, heads, head_dim, generator=generator, dtype=torch.float32))
    if not rows:
        return torch.empty(batch_size, 0, heads, head_dim, dtype=dtype)
    return torch.stack(rows, dim=1).to(dtype=dtype)
