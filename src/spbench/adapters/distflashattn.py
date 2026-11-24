"""DistFlashAttn through RulinShao/LightSeq at commit 8f486dad."""

from __future__ import annotations

import math

import torch

from spbench.adapters.base import Availability, require_cuda_distributed
from spbench.errors import InvalidConfiguration, UnavailableBackend


class DistFlashAttnAdapter:
    name = "distflashattn"

    def availability(self) -> Availability:
        if not torch.cuda.is_available():
            return Availability(False, "CUDA is not available, so the LightSeq Triton kernels cannot be timed.")
        try:
            import lightseq_async_attn  # noqa: F401
        except Exception as exc:
            return Availability(
                False,
                "lightseq_async_attn is not importable. LightSeq has no project license file and is not "
                f"vendored here ({type(exc).__name__}: {exc}).",
            )
        return Availability(True, "lightseq_async_attn imported and CUDA is visible. License of the checkout is still unresolved.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        require_cuda_distributed("DistFlashAttn")
        if not causal:
            raise InvalidConfiguration("DistFlashAttn noncausal execution is not available in the audited schedule")
        try:
            from async_communication import initialize_distributed
            from lightseq_async_attn import attention
        except Exception as exc:
            raise UnavailableBackend(
                "DistFlashAttn imports failed. Expected lightseq_async_attn.attention and "
                f"async_communication.initialize_distributed ({type(exc).__name__}: {exc})."
            ) from exc
        # Upstream forward reads BHSD. The conversion is part of the adapter and
        # therefore inside the timed region when the benchmark calls this method.
        query_bhsd = query.transpose(1, 2).contiguous()
        key_bhsd = key.transpose(1, 2).contiguous()
        value_bhsd = value.transpose(1, 2).contiguous()
        initialize_distributed()
        output = attention(query_bhsd, key_bhsd, value_bhsd, True, 1.0 / math.sqrt(query.shape[-1]))
        return output.transpose(1, 2).contiguous()
