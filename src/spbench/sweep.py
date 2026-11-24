"""Dry-run the experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spbench.cases import expand_matrix
from spbench.config import load_config
from spbench.dependencies import probe_dependencies
from spbench.hardware import detect_hardware, hypothetical_server_hardware


def build_matrix(config_path: str, include_hypothetical: bool) -> list[dict]:
    config = load_config(config_path)
    hardware = detect_hardware()
    dependencies = probe_dependencies()
    cases = expand_matrix(config, hardware, dependencies)
    if include_hypothetical:
        assumed = {name: True for name in dependencies}
        hypothetical = expand_matrix(config, hypothetical_server_hardware(), assumed)
        cases.extend(case for case in hypothetical if case.status == "hypothetical")
    return [case.to_dict() for case in cases]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify the Project 05 experiment matrix.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-hypothetical-plan", action="store_true")
    parser.add_argument("--output", default="results/processed/dry_run_matrix.json")
    args = parser.parse_args(argv)
    if not args.dry_run:
        print(
            "Refusing to launch the matrix. Use --dry-run to classify cases, and torchrun "
            "-m spbench.benchmark for one configuration after the GPU server is available."
        )
        return 2
    rows = build_matrix(args.config, args.include_hypothetical_plan)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "dry_run_plan",
        "measured": False,
        "counts": counts,
        "cases": rows,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
