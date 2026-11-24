"""Shared adapter surface."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spbench.errors import UnavailableBackend


@dataclass(frozen=True)
class Availability:
    ready: bool
    reason: str


def require_cuda_distributed(method: str) -> None:
    if not torch.cuda.is_available():
        raise UnavailableBackend(
            f"{method} timing requires NVIDIA CUDA. The CPU schedule checker is not a substitute "
            "and will not be used for a timed run."
        )
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise UnavailableBackend(f"{method} must be launched under torchrun with an initialized NCCL process group")
    if dist.get_backend() != "nccl":
        raise UnavailableBackend(f"{method} timed runs use NCCL. The current backend is {dist.get_backend()}.")
