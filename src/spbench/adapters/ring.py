"""Ring Attention through zhuzilin/ring-flash-attention."""

from __future__ import annotations

import torch

from spbench.adapters.base import Availability, require_cuda_distributed
from spbench.errors import InvalidConfiguration, UnavailableBackend


class RingAdapter:
    name = "ring"

    def availability(self) -> Availability:
        if not torch.cuda.is_available():
            return Availability(False, "CUDA is not available, so ring-flash-attention cannot be timed.")
        try:
            import ring_flash_attn  # noqa: F401
        except Exception as exc:
            return Availability(False, f"ring_flash_attn is not importable ({type(exc).__name__}: {exc}).")
        return Availability(True, "ring_flash_attn imported and CUDA is visible.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        variant: str,
        causal: bool,
        process_group,
    ) -> torch.Tensor:
        require_cuda_distributed("Ring Attention")
        if variant == "zigzag" and not causal:
            raise InvalidConfiguration("upstream zigzag_ring_flash_attn_forward asserts causal == True")
        try:
            from ring_flash_attn import ring_flash_attn_func, zigzag_ring_flash_attn_func
        except Exception as exc:
            raise UnavailableBackend(f"ring_flash_attn is not importable ({type(exc).__name__}: {exc})") from exc
        if variant == "basic":
            function = ring_flash_attn_func
        elif variant == "zigzag":
            function = zigzag_ring_flash_attn_func
        else:
            raise InvalidConfiguration(f"unknown ring variant {variant}")
        return function(query, key, value, dropout_p=0.0, softmax_scale=None, causal=causal, group=process_group)
