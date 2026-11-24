"""FP32 dense attention for correctness checks."""

from __future__ import annotations

import torch

from spbench.adapters.base import Availability
from spbench.attention_math import reference_attention
from spbench.errors import UnavailableBackend


class ReferenceAdapter:
    name = "reference"

    def availability(self) -> Availability:
        return Availability(
            ready=True,
            reason="FP32 dense PyTorch attention is available for correctness checks only.",
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool = True) -> torch.Tensor:
        return reference_attention(query, key, value, causal=causal)

    def timed_forward(self, *args, **kwargs):
        raise UnavailableBackend(
            "The dense reference is excluded from timed runs so it cannot be reported as Ring, "
            "DistFlashAttn, Ulysses, USP, or the single-GPU baseline."
        )
