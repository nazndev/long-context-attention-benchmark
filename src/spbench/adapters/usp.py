"""Unified Sequence Parallelism through YunChang LongContextAttention."""

from __future__ import annotations

import torch

from spbench.adapters.base import Availability, require_cuda_distributed
from spbench.errors import InvalidConfiguration, UnavailableBackend


class UspAdapter:
    name = "usp"

    def availability(self) -> Availability:
        if not torch.cuda.is_available():
            return Availability(False, "CUDA is not available, so YunChang USP cannot be timed.")
        try:
            import yunchang  # noqa: F401
        except Exception as exc:
            return Availability(False, f"yunchang is not importable ({type(exc).__name__}: {exc}).")
        return Availability(True, "yunchang imported and CUDA is visible. FlashAttention compatibility is still unresolved.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        variant: str,
        placement: str,
        ulysses_degree: int,
        ring_degree: int,
        causal: bool,
    ) -> torch.Tensor:
        require_cuda_distributed("USP")
        if variant not in {"basic", "zigzag"}:
            raise InvalidConfiguration(f"USP variant must be basic or zigzag, got {variant}")
        if placement not in {"ulysses_low", "ring_low"}:
            raise InvalidConfiguration(f"USP placement must be ulysses_low or ring_low, got {placement}")
        if variant == "zigzag" and not causal:
            raise InvalidConfiguration("zigzag USP follows the causal zigzag ring schedule")
        try:
            import torch.distributed as dist
            from yunchang import LongContextAttention, set_seq_parallel_pg
            from yunchang.kernels import AttnType
        except Exception as exc:
            raise UnavailableBackend(f"yunchang is not importable ({type(exc).__name__}: {exc})") from exc
        set_seq_parallel_pg(
            ulysses_degree,
            ring_degree,
            dist.get_rank(),
            dist.get_world_size(),
            use_ulysses_low=(placement == "ulysses_low"),
        )
        attention = LongContextAttention(
            scatter_idx=2,
            gather_idx=1,
            ring_impl_type=variant,
            attn_type=AttnType.FA,
        )
        return attention(query, key, value, dropout_p=0.0, causal=causal)
