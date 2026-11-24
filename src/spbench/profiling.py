"""Nsight command construction and overlap arithmetic."""

from __future__ import annotations

import shutil
from pathlib import Path

from spbench.metrics import overlap_fraction


def nsys_available() -> bool:
    return shutil.which("nsys") is not None


def build_nsys_command(torchrun_command: list[str], output_prefix: str) -> list[str]:
    if not torchrun_command:
        raise ValueError("a torchrun command is required")
    prefix = Path(output_prefix)
    return [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,nccl",
        "--cuda-trace-scope=process-tree",
        f"--output={prefix}",
        "--force-overwrite=true",
        *torchrun_command,
    ]


def overlap_from_device_intervals(
    compute_intervals_ms: list[tuple[float, float]] | None,
    communication_intervals_ms: list[tuple[float, float]] | None,
) -> dict[str, float | None]:
    from spbench.metrics import union_duration_ms

    overlap_ms, fraction = overlap_fraction(compute_intervals_ms, communication_intervals_ms)
    communication_ms = union_duration_ms(communication_intervals_ms)
    exposed_ms = None
    if overlap_ms is not None and communication_ms is not None:
        exposed_ms = communication_ms - overlap_ms
    return {
        "communication_active_ms": communication_ms,
        "overlap_ms": overlap_ms,
        "overlap_fraction": fraction,
        "exposed_communication_ms": exposed_ms,
    }
