"""Capacity-boundary labels."""

from __future__ import annotations

from typing import Any


def classify_capacity(
    *,
    last_success_length: int | None,
    first_confirmed_oom_length: int | None,
    stopping_reason: str,
) -> str:
    if stopping_reason == "oom_confirmed" and first_confirmed_oom_length is not None:
        return "oom_confirmed"
    if stopping_reason == "success" and first_confirmed_oom_length is None:
        return "censored"
    if stopping_reason in {
        "time_cap",
        "approved_length_cap",
        "unsupported",
        "gpu_unavailable",
        "timeout",
        "dependency_missing",
        "other_error",
    }:
        return "censored"
    raise ValueError(f"unrecognized stopping reason {stopping_reason!r}")


def capacity_record(
    *,
    method: str,
    topology: str,
    gpu_count: int,
    dtype: str,
    phase: str,
    last_success_length: int | None,
    first_confirmed_oom_length: int | None,
    attempted_max_length: int | None,
    stopping_reason: str,
    run_ids: list[str],
) -> dict[str, Any]:
    label = classify_capacity(
        last_success_length=last_success_length,
        first_confirmed_oom_length=first_confirmed_oom_length,
        stopping_reason=stopping_reason,
    )
    return {
        "method": method,
        "topology": topology,
        "gpu_count": gpu_count,
        "dtype": dtype,
        "phase": phase,
        "last_success_length": last_success_length,
        "first_confirmed_oom_length": first_confirmed_oom_length,
        "attempted_max_length": attempted_max_length,
        "stopping_reason": stopping_reason,
        "capacity_class": label,
        "run_ids": run_ids,
        "maximum_length_claimed": False,
    }
