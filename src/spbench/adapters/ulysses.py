"""DeepSpeed Ulysses through DistributedAttention."""

from __future__ import annotations

import math

import torch
from torch import nn

from spbench.adapters.base import Availability, require_cuda_distributed
from spbench.errors import UnavailableBackend


class _SdpaLocal(nn.Module):
    """Local attention after Ulysses has already all-to-all'd the heads.

    The module is SDPA, not the flash-attn package. The benchmark record must
    keep that distinction. It is only constructed on the timed CUDA path.
    """

    def forward(self, query, key, value, *args, **kwargs):
        causal = bool(kwargs.get("causal", True))
        dropout_p = float(kwargs.get("dropout_p", 0.0))
        query_h = query.transpose(1, 2)
        key_h = key.transpose(1, 2)
        value_h = value.transpose(1, 2)
        output = torch.nn.functional.scaled_dot_product_attention(
            query_h,
            key_h,
            value_h,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=1.0 / math.sqrt(query.shape[-1]),
        )
        return output.transpose(1, 2)


class UlyssesAdapter:
    name = "ulysses"

    def availability(self) -> Availability:
        if not torch.cuda.is_available():
            return Availability(False, "CUDA is not available, so DeepSpeed Ulysses cannot be timed.")
        try:
            from deepspeed.sequence.layer import DistributedAttention  # noqa: F401
        except Exception as exc:
            return Availability(False, f"deepspeed.sequence.layer.DistributedAttention is not importable ({type(exc).__name__}: {exc}).")
        return Availability(True, "DeepSpeed DistributedAttention imported and CUDA is visible.")

    def forward(self, query, key, value, process_group, causal: bool = True):
        require_cuda_distributed("DeepSpeed Ulysses")
        try:
            from deepspeed.sequence.layer import DistributedAttention
        except Exception as exc:
            raise UnavailableBackend(
                f"deepspeed.sequence.layer.DistributedAttention is not importable ({type(exc).__name__}: {exc})"
            ) from exc
        attention = DistributedAttention(
            _SdpaLocal(),
            process_group,
            scatter_idx=2,
            gather_idx=1,
            validate_shards=True,
        )
        return attention(query, key, value, batch_dim_idx=0, causal=causal, dropout_p=0.0)
