"""JSONL result schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from spbench.errors import InvalidConfiguration

REQUIRED_FIELDS: tuple[str, ...] = (
    "run_id",
    "timestamp_utc",
    "git_commit",
    "upstream_commits",
    "environment_id",
    "method",
    "variant",
    "status",
    "error_reason",
    "nodes",
    "world_size",
    "rank",
    "host_id",
    "gpu_model",
    "gpu_memory_mib",
    "verified_transport",
    "ulysses_degree",
    "ring_degree",
    "global_sequence_length",
    "local_sequence_length",
    "batch_size",
    "heads",
    "head_dim",
    "dtype",
    "causal",
    "phase",
    "seed",
    "repeat",
    "iteration",
    "step_seconds",
    "max_rank_step_seconds",
    "global_tokens_per_second",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "device_peak_sampled_mib",
    "compute_active_ms",
    "communication_active_ms",
    "overlap_ms",
    "overlap_fraction",
    "valid_causal_pairs",
    "trace_path",
)

MEASUREMENT_FIELDS: tuple[str, ...] = (
    "step_seconds",
    "max_rank_step_seconds",
    "global_tokens_per_second",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "device_peak_sampled_mib",
    "compute_active_ms",
    "communication_active_ms",
    "overlap_ms",
    "overlap_fraction",
    "trace_path",
)


def empty_record() -> dict[str, Any]:
    return {field: None for field in REQUIRED_FIELDS}


def validate_record(record: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise InvalidConfiguration(f"result record is missing fields: {', '.join(missing)}")
    if record.get("status") != "success":
        for field in MEASUREMENT_FIELDS:
            if record.get(field) not in (None,):
                # A failed or blocked row may still name an error. It may not
                # carry timings, memory, or overlap. Zero is not a stand-in.
                raise InvalidConfiguration(
                    f"status {record.get('status')!r} cannot carry measurement field {field}"
                )
    if record.get("origin") == "fixture" and record.get("status") == "success":
        # Fixtures are allowed only when the caller labels them. Success is
        # permitted inside unit tests, but research writers reject the origin.
        return


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_record(record)
            handle.write(json.dumps(record, allow_nan=False, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            validate_record(record)
            records.append(record)
    return records
