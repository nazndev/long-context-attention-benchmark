"""Single-GPU PyTorch SDPA baseline."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spbench.attention_math import reference_attention
from spbench.capacity import capacity_record
from spbench.config import load_config
from spbench.hardware import detect_hardware
from spbench.inputs import generate_bshd
from spbench.metrics import global_tokens_per_second
from spbench.provenance import provenance
from spbench.schema import empty_record, write_jsonl

PHASES = ("forward", "forward_backward")
OOM_RETURN_CODE = 3


def detect_baseline_support() -> dict[str, Any]:
    hardware = detect_hardware()
    report: dict[str, Any] = {
        "status": "skipped_no_cuda",
        "dtype": None,
        "backend": None,
        "backend_errors": [],
        "reason": None,
        "implementation": "pytorch_sdpa_single_gpu",
        "cpu_schedule": False,
        "cuda_available": hardware["cuda_available"],
        "mps_available": hardware["mps_available"],
        "gpu_names": hardware["gpu_names"],
        "compute_capabilities": hardware["compute_capabilities"],
        "gpu_architectures": hardware["gpu_architectures"],
        "gpu_memory_mib": hardware["gpu_memory_mib"],
        "gpu_free_mib": hardware["gpu_free_mib"],
        "bf16_hardware": hardware["bf16_hardware"],
        "torch_version": hardware["torch_version"],
        "torch_cuda_build": hardware["torch_cuda_build"],
    }
    if not hardware["cuda_available"]:
        report["reason"] = (
            "No CUDA device is visible. The single-GPU baseline was not timed. "
            "MPS is not an NVIDIA baseline. CPU schedule checks do not establish GPU correctness."
        )
        return report
    import torch

    major = int(str(hardware["compute_capabilities"][0]).split(".", 1)[0])
    if major >= 8 and torch.cuda.is_bf16_supported():
        report["dtype"] = "bf16"
        report["torch_dtype"] = "bfloat16"
    else:
        report["dtype"] = "fp16"
        report["torch_dtype"] = "float16"
        report["reason"] = (
            "BF16 hardware is not available on this GPU. Measured rows from this process are FP16 "
            "and must not be pooled with BF16 runs."
        )
    backend, errors = _probe_sdpa_backend(getattr(torch, report["torch_dtype"]))
    report["backend"] = backend
    report["backend_errors"] = errors
    report["status"] = "ready" if backend != "unknown" else "backend_unknown"
    if backend == "unknown":
        report["reason"] = (
            "No named PyTorch SDPA backend completed a tiny causal attention. Timing is withheld. "
            "This is not recorded as FlashAttention."
        )
    elif backend == "pytorch_sdpa_math":
        report["reason"] = (report["reason"] or "") + (
            " The selected kernel is PyTorch SDPA math, not the flash-attn package and not an efficient SDPA kernel."
        )
    return report


def _probe_sdpa_backend(dtype) -> tuple[str, list[str]]:
    import torch
    import torch.nn.functional as functional

    query = torch.randn(1, 2, 32, 64, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    errors: list[str] = []
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:
        errors.append(f"sdpa_kernel import failed: {type(exc).__name__}: {exc}")
        return "pytorch_sdpa_unscoped", errors
    candidates = [
        (SDPBackend.FLASH_ATTENTION, "pytorch_sdpa_flash"),
        (SDPBackend.EFFICIENT_ATTENTION, "pytorch_sdpa_mem_efficient"),
        (SDPBackend.MATH, "pytorch_sdpa_math"),
    ]
    for backend, name in candidates:
        try:
            with sdpa_kernel([backend]):
                functional.scaled_dot_product_attention(query, key, value, is_causal=True)
            torch.cuda.synchronize()
            return name, errors
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return "unknown", errors


def _sdpa(query, key, value, backend_name: str):
    import torch.nn.functional as functional

    query_h = query.transpose(1, 2)
    key_h = key.transpose(1, 2)
    value_h = value.transpose(1, 2)
    if backend_name == "pytorch_sdpa_unscoped":
        output = functional.scaled_dot_product_attention(query_h, key_h, value_h, is_causal=True)
        return output.transpose(1, 2)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend = {
        "pytorch_sdpa_flash": SDPBackend.FLASH_ATTENTION,
        "pytorch_sdpa_mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "pytorch_sdpa_math": SDPBackend.MATH,
    }[backend_name]
    with sdpa_kernel([backend]):
        output = functional.scaled_dot_product_attention(query_h, key_h, value_h, is_causal=True)
    return output.transpose(1, 2)


def _base_record(config: dict, support: dict, sequence_length: int, phase: str, config_path: Path) -> dict[str, Any]:
    hardware = detect_hardware()
    identity = provenance(config_path)
    record = empty_record()
    record.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": identity["git_commit"],
            "upstream_commits": identity["upstream_commits"],
            "environment_id": hardware.get("platform"),
            "method": "single_gpu_sdpa",
            "variant": support.get("backend"),
            "nodes": 1,
            "world_size": 1,
            "rank": 0,
            "host_id": socket.gethostname(),
            "gpu_model": (hardware.get("gpu_names") or [None])[0],
            "gpu_memory_mib": (hardware.get("gpu_memory_mib") or [None])[0],
            "verified_transport": None,
            "global_sequence_length": sequence_length,
            "local_sequence_length": sequence_length,
            "batch_size": int(config["batch_size"]),
            "heads": int(config["num_heads"]),
            "head_dim": int(config["head_dim"]),
            "dtype": support.get("dtype"),
            "causal": True,
            "phase": phase,
            "seed": int(config["seed"]),
            "valid_causal_pairs": sequence_length * (sequence_length + 1) // 2,
            "origin": "measured" if support.get("status") == "ready" else "skipped",
            "implementation": "pytorch_sdpa_single_gpu",
            "cpu_schedule": False,
            "source_tree_sha256": identity["source_tree_sha256"],
            "config_sha256": identity.get("config_sha256"),
            "dependency_versions": identity["dependency_versions"],
            "compute_capability": (hardware.get("compute_capabilities") or [None])[0],
            "gpu_architecture": (hardware.get("gpu_architectures") or [None])[0],
            "gpu_free_mib": (hardware.get("gpu_free_mib") or [None])[0],
            "torch_version": hardware.get("torch_version"),
            "torch_cuda_build": hardware.get("torch_cuda_build"),
        }
    )
    return record


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text


def run_one_length(
    sequence_length: int,
    output: Path,
    config: dict,
    support: dict,
    phase: str,
    config_path: Path,
) -> int:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    if support["status"] != "ready":
        record = _base_record(config, support, sequence_length, phase, config_path)
        record.update(
            {
                "run_id": str(uuid.uuid4()),
                "status": "skipped",
                "error_reason": support.get("reason"),
                "origin": "skipped",
            }
        )
        write_jsonl(output, [record])
        return 0
    import torch

    dtype = getattr(torch, support["torch_dtype"])
    device = torch.device("cuda", 0)
    seed = int(config["seed"])
    heads = int(config["num_heads"])
    head_dim = int(config["head_dim"])
    batch = int(config["batch_size"])
    positions = list(range(sequence_length))
    try:
        query = generate_bshd(positions, batch, heads, head_dim, seed, "q").to(device=device, dtype=dtype)
        key = generate_bshd(positions, batch, heads, head_dim, seed, "k").to(device=device, dtype=dtype)
        value = generate_bshd(positions, batch, heads, head_dim, seed, "v").to(device=device, dtype=dtype)
        incoming = None
        if phase == "forward_backward":
            incoming = generate_bshd(positions, batch, heads, head_dim, seed, "do").to(device=device, dtype=dtype)
            query.requires_grad_(True)
            key.requires_grad_(True)
            value.requires_grad_(True)

        def step() -> None:
            if phase == "forward":
                with torch.no_grad():
                    _sdpa(query, key, value, support["backend"])
                return
            query.grad = None
            key.grad = None
            value.grad = None
            output_tensor = _sdpa(query, key, value, support["backend"])
            (output_tensor * incoming).sum().backward()

        for _ in range(int(config["warmup_iterations"])):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(int(config["measured_iterations"])):
            torch.cuda.synchronize()
            started = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    except Exception as exc:
        if not _is_oom(exc):
            raise
        torch.cuda.empty_cache()
        record = _base_record(config, support, sequence_length, phase, config_path)
        record.update(
            {
                "run_id": str(uuid.uuid4()),
                "status": "oom",
                "error_reason": str(exc),
                "origin": "measured",
                "gpu_model": torch.cuda.get_device_name(device),
            }
        )
        write_jsonl(output, [record])
        return OOM_RETURN_CODE
    records = []
    run_id = str(uuid.uuid4())
    for iteration, seconds in enumerate(times):
        record = _base_record(config, support, sequence_length, phase, config_path)
        record.update(
            {
                "run_id": run_id,
                "status": "success",
                "repeat": 0,
                "iteration": iteration,
                "step_seconds": seconds,
                "max_rank_step_seconds": seconds,
                "global_tokens_per_second": global_tokens_per_second(batch, sequence_length, seconds),
                "peak_allocated_mib": allocated,
                "peak_reserved_mib": reserved,
                "gpu_model": torch.cuda.get_device_name(device),
                "gpu_memory_mib": int(torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)),
                "origin": "measured",
            }
        )
        records.append(record)
    write_jsonl(output, records)
    return 0


def _relative_l2(actual, expected) -> float:
    return float(actual.norm() / expected.norm().clamp_min(1e-12))


def correctness_against_reference(support: dict, lengths: list[int], config: dict) -> list[dict]:
    """Compare the selected GPU kernel with FP32 dense attention. CPU calls return a skip."""

    if support["status"] != "ready":
        return [
            {"length": length, "phase": phase, "status": "skipped", "reason": support["reason"]}
            for phase in PHASES
            for length in lengths
        ]
    import torch

    dtype = getattr(torch, support["torch_dtype"])
    tolerance = config.get("tolerance_initial", {})
    rows = []
    for length in lengths:
        positions = list(range(length))
        query = generate_bshd(positions, 1, int(config["num_heads"]), int(config["head_dim"]), int(config["seed"]), "q")
        key = generate_bshd(positions, 1, int(config["num_heads"]), int(config["head_dim"]), int(config["seed"]), "k")
        value = generate_bshd(positions, 1, int(config["num_heads"]), int(config["head_dim"]), int(config["seed"]), "v")
        incoming = generate_bshd(positions, 1, int(config["num_heads"]), int(config["head_dim"]), int(config["seed"]), "do")
        reference = reference_attention(query, key, value, causal=True)
        reference_query = query.detach().clone().requires_grad_(True)
        reference_key = key.detach().clone().requires_grad_(True)
        reference_value = value.detach().clone().requires_grad_(True)
        reference_out = reference_attention(reference_query, reference_key, reference_value, causal=True)
        (reference_out * incoming).sum().backward()
        gpu_query = query.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
        gpu_key = key.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
        gpu_value = value.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
        gpu_out = _sdpa(gpu_query, gpu_key, gpu_value, support["backend"])
        output_delta = (gpu_out.float().cpu() - reference).abs()
        (gpu_out * incoming.to(device="cuda", dtype=dtype)).sum().backward()
        gradient_errors = {}
        for name, actual, expected in (
            ("dQ", gpu_query.grad, reference_query.grad),
            ("dK", gpu_key.grad, reference_key.grad),
            ("dV", gpu_value.grad, reference_value.grad),
        ):
            delta = (actual.float().cpu() - expected).abs()
            gradient_errors[name] = {"max_abs": float(delta.max()), "rel_l2": _relative_l2(delta, expected.abs())}
        rows.append(
            {
                "length": length,
                "status": "compared",
                "dtype": support["dtype"],
                "backend": support["backend"],
                "implementation": "pytorch_sdpa_single_gpu",
                "forward_max_abs": float(output_delta.max()),
                "forward_rel_l2": _relative_l2(output_delta, reference.abs()),
                "gradient_errors": gradient_errors,
                "initial_tolerance": tolerance,
                "note": "Observed errors against FP32 dense attention. The initial tolerance is not a pass by itself.",
            }
        )
    return rows


def capacity_from_attempts(phase: str, attempts: list[dict[str, Any]], support_status: str) -> dict[str, Any]:
    """Summarize one phase. A finished grid without OOM stays censored."""

    successes = [int(item["sequence_length"]) for item in attempts if item.get("returncode") == 0]
    ooms = [int(item["sequence_length"]) for item in attempts if item.get("returncode") == OOM_RETURN_CODE]
    other = [item for item in attempts if item.get("returncode") not in (0, OOM_RETURN_CODE, None)]
    last_success = max(successes) if successes else None
    first_oom = min(ooms) if ooms else None
    attempted_max = max((int(item["sequence_length"]) for item in attempts), default=None)
    if support_status != "ready":
        reason = "gpu_unavailable"
    elif first_oom is not None:
        reason = "oom_confirmed"
    elif other:
        reason = "other_error"
    elif successes:
        reason = "approved_length_cap"
    else:
        reason = "gpu_unavailable"
    record = capacity_record(
        method="single_gpu_sdpa",
        topology="single_node",
        gpu_count=1,
        dtype="",
        phase=phase,
        last_success_length=last_success,
        first_confirmed_oom_length=first_oom,
        attempted_max_length=attempted_max,
        stopping_reason=reason,
        run_ids=[],
    )
    record["tested_upper_bound"] = attempted_max if reason == "approved_length_cap" else None
    return record


def sweep_commands(config_path: str, output_dir: str, lengths: list[int]) -> list[list[str]]:
    commands = []
    destination = Path(output_dir)
    for phase in PHASES:
        for length in lengths:
            target = destination / f"baseline_{phase}_seq{length}.jsonl"
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "spbench.baseline",
                    "--config",
                    config_path,
                    "--once",
                    "--phase",
                    phase,
                    "--seq-len",
                    str(length),
                    "--output",
                    str(target),
                ]
            )
    return commands


def sweep_subprocess(config_path: str, output_dir: str) -> dict[str, Any]:
    support = detect_baseline_support()
    config = load_config(config_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    public_support = {key: value for key, value in support.items() if key != "torch_dtype"}
    manifest: dict[str, Any] = {"support": public_support, "lengths": [], "capacity": []}
    if support["status"] != "ready":
        for phase in PHASES:
            manifest["capacity"].append(
                capacity_from_attempts(phase, [], support["status"])
            )
        manifest["lengths"].append({"status": "skipped", "reason": support["reason"]})
        (destination / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    lengths = [int(length) for length in config["colab_baseline_sequence_lengths"]]
    for phase in PHASES:
        attempts = []
        for length in lengths:
            target = destination / f"baseline_{phase}_seq{length}.jsonl"
            command = [
                sys.executable,
                "-m",
                "spbench.baseline",
                "--config",
                config_path,
                "--once",
                "--phase",
                phase,
                "--seq-len",
                str(length),
                "--output",
                str(target),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=600)
            attempt = {
                "phase": phase,
                "sequence_length": length,
                "returncode": completed.returncode,
                "output": str(target),
                "stderr_tail": completed.stderr[-500:],
            }
            attempts.append(attempt)
            manifest["lengths"].append(attempt)
            if completed.returncode == OOM_RETURN_CODE:
                break
            if completed.returncode != 0:
                break
        summary = capacity_from_attempts(phase, attempts, support["status"])
        summary["dtype"] = support.get("dtype")
        manifest["capacity"].append(summary)
    (destination / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single-GPU SDPA baseline. Skips cleanly without CUDA.")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--phase", default="forward_backward", choices=PHASES)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--output-dir", default="results/raw/baseline")
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.sweep:
        sweep_subprocess(args.config, args.output_dir)
        return 0
    support = detect_baseline_support()
    if not args.once:
        print(json.dumps({key: value for key, value in support.items() if key != "torch_dtype"}, indent=2, default=str))
        return 0
    if args.seq_len is None or args.output is None:
        print("--once requires --seq-len and --output")
        return 2
    return run_one_length(args.seq_len, Path(args.output), load_config(args.config), support, args.phase, Path(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
