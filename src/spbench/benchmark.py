"""One timed configuration, launched with torchrun on an NVIDIA machine."""

from __future__ import annotations

import argparse
import os
import socket
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from spbench.adapters import ADAPTERS
from spbench.cases import Case, validate_case
from spbench.config import load_config
from spbench.errors import BenchmarkError, UnavailableBackend
from spbench.hardware import detect_hardware
from spbench.schema import empty_record, write_jsonl
from spbench.upstream import upstream_commit_map


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def blocked_record(case: Case, reason: str) -> dict:
    hardware = detect_hardware()
    record = empty_record()
    record.update(
        {
            "run_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "upstream_commits": upstream_commit_map(),
            "environment_id": hardware.get("platform"),
            "method": case.method,
            "variant": case.variant,
            "status": "blocked",
            "error_reason": reason,
            "nodes": case.nodes,
            "world_size": case.world_size,
            "rank": int(os.environ.get("RANK", "0")),
            "host_id": socket.gethostname(),
            "gpu_model": (hardware.get("gpu_names") or [None])[0],
            "gpu_memory_mib": (hardware.get("gpu_memory_mib") or [None])[0],
            "verified_transport": None,
            "ulysses_degree": case.ulysses_degree,
            "ring_degree": case.ring_degree,
            "global_sequence_length": case.global_sequence_length,
            "local_sequence_length": case.global_sequence_length // case.world_size,
            "batch_size": case.batch_size,
            "heads": case.heads,
            "head_dim": case.head_dim,
            "dtype": case.dtype,
            "causal": case.causal,
            "phase": case.phase,
            "seed": 42,
            "repeat": None,
            "iteration": None,
            "valid_causal_pairs": None,
            "origin": "blocker",
        }
    )
    return record


def case_from_args(args, config: dict) -> Case:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    nodes = int(os.environ.get("SPBENCH_NODES", "1"))
    if args.method == "ring":
        ulysses, ring, variant = 1, world_size, args.variant or "basic"
    elif args.method == "ulysses":
        ulysses, ring, variant = world_size, 1, "default"
    elif args.method == "distflashattn":
        ulysses, ring, variant = None, None, "balanced"
    elif args.method == "usp":
        ulysses = args.ulysses_degree
        ring = args.ring_degree
        variant = args.variant or "basic"
    else:
        raise UnavailableBackend(f"{args.method} is not a timed distributed method in this command")
    return Case(
        method=args.method,
        variant=variant,
        phase=args.phase,
        topology="two_nodes" if nodes == 2 else "single_node",
        nodes=nodes,
        world_size=world_size,
        global_sequence_length=args.seq_len,
        ulysses_degree=ulysses,
        ring_degree=ring,
        placement=args.placement,
        genuine_hybrid=bool(ulysses and ring and ulysses > 1 and ring > 1),
        dtype=config["dtype"],
        causal=True,
        batch_size=int(config["batch_size"]),
        heads=int(config["num_heads"]),
        kv_heads=int(config["kv_heads"]),
        head_dim=int(config["head_dim"]),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
    )


def refuse_or_prepare(args) -> Case:
    config = load_config(args.config)
    case = case_from_args(args, config)
    reasons = validate_case(case)
    if reasons:
        raise UnavailableBackend("invalid configuration: " + "; ".join(reasons))
    hardware = detect_hardware()
    if not hardware["cuda_available"]:
        raise UnavailableBackend(
            f"Refusing to time {case.method}. CUDA is not available. "
            "Dense attention and the CPU schedules are not substitutes."
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != case.world_size or case.world_size < 2:
        raise UnavailableBackend(
            "Distributed timing must be started with torchrun so WORLD_SIZE matches the case. "
            "Example: torchrun --standalone --nproc-per-node=2 -m spbench.benchmark "
            "--method ring --seq-len 4096 --phase forward_backward"
        )
    adapter = ADAPTERS[case.method]
    availability = adapter.availability()
    if not availability.ready:
        raise UnavailableBackend(availability.reason)
    return case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time one distributed attention configuration.")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--method", required=True, choices=["ring", "distflashattn", "ulysses", "usp"])
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--phase", required=True, choices=["forward", "forward_backward"])
    parser.add_argument("--ulysses-degree", type=int, default=None)
    parser.add_argument("--ring-degree", type=int, default=None)
    parser.add_argument("--placement", default="ulysses_low")
    parser.add_argument("--output", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args(argv)
    try:
        case = refuse_or_prepare(args)
    except (BenchmarkError, UnavailableBackend) as exc:
        print(str(exc))
        if args.output:
            try:
                config = load_config(args.config)
                case = case_from_args(args, config)
            except Exception:
                case = None
            if case is not None:
                write_jsonl(Path(args.output), [blocked_record(case, str(exc))])
        return 2
    return run_worker(args, case)


def _positions_for(case: Case, rank: int) -> list[int]:
    from spbench.layouts import contiguous_positions, usp_positions, zigzag_ring_positions

    if case.method == "ring" and case.variant == "zigzag":
        return zigzag_ring_positions(case.global_sequence_length, case.world_size, rank)
    if case.method == "usp":
        return usp_positions(
            case.global_sequence_length,
            rank,
            int(case.ulysses_degree),
            int(case.ring_degree),
            case.placement or "ulysses_low",
            case.variant,
        )
    return contiguous_positions(case.global_sequence_length, case.world_size, rank)


def _call_adapter(case: Case, query, key, value, process_group):
    adapter = ADAPTERS[case.method]
    if case.method == "ring":
        return adapter.forward(query, key, value, case.variant, True, process_group)
    if case.method == "ulysses":
        return adapter.forward(query, key, value, process_group, causal=True)
    if case.method == "distflashattn":
        return adapter.forward(query, key, value, True)
    if case.method == "usp":
        return adapter.forward(
            query,
            key,
            value,
            case.variant,
            case.placement or "ulysses_low",
            int(case.ulysses_degree),
            int(case.ring_degree),
            True,
        )
    raise UnavailableBackend(f"no timed adapter for {case.method}")


def run_worker(args, case: Case) -> int:
    """Time one launched rank. Iteration-boundary synchronizations stay outside the step."""

    import time

    import torch
    import torch.distributed as dist

    from spbench.inputs import generate_bshd
    from spbench.metrics import global_tokens_per_second
    from spbench.work_model import pairs_for_query_positions

    config = load_config(args.config)
    warmup_iterations = int(config["warmup_iterations"])
    measured_iterations = int(config["measured_iterations"])
    seed = int(config["seed"])
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=args.timeout_seconds))
    rank = dist.get_rank()
    try:
        local_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % local_gpus)
        device = torch.device("cuda", rank % local_gpus)
        if case.dtype == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise UnavailableBackend(
                    "This case is bf16, and this GPU does not support bf16. "
                    "Run a separate fp16 configuration instead of relabeling the result."
                )
            dtype = torch.bfloat16
        elif case.dtype == "fp16":
            dtype = torch.float16
        else:
            raise UnavailableBackend(f"unsupported timed dtype {case.dtype}")
        positions = _positions_for(case, rank)
        query = generate_bshd(positions, case.batch_size, case.heads, case.head_dim, seed, "q").to(device=device, dtype=dtype)
        key = generate_bshd(positions, case.batch_size, case.heads, case.head_dim, seed, "k").to(device=device, dtype=dtype)
        value = generate_bshd(positions, case.batch_size, case.heads, case.head_dim, seed, "v").to(device=device, dtype=dtype)
        incoming = generate_bshd(positions, case.batch_size, case.heads, case.head_dim, seed, "do").to(device=device, dtype=dtype)
        if case.phase == "forward_backward":
            query.requires_grad_(True)
            key.requires_grad_(True)
            value.requires_grad_(True)

        def one_step() -> None:
            if case.phase == "forward_backward":
                for tensor in (query, key, value):
                    tensor.grad = None
                output = _call_adapter(case, query, key, value, dist.group.WORLD)
                (output * incoming).sum().backward()
            else:
                with torch.no_grad():
                    _call_adapter(case, query, key, value, dist.group.WORLD)

        for _ in range(warmup_iterations):
            one_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        step_times: list[float] = []
        for _ in range(measured_iterations):
            torch.cuda.synchronize()
            started = time.perf_counter()
            one_step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - started)
        allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        run_id_box = [str(uuid.uuid4()) if rank == 0 else ""]
        dist.broadcast_object_list(run_id_box, src=0)
        records = []
        valid_pairs = pairs_for_query_positions(positions)
        for iteration, step_seconds in enumerate(step_times):
            holder = torch.tensor([step_seconds], device=device)
            dist.all_reduce(holder, op=dist.ReduceOp.MAX)
            max_seconds = float(holder.item())
            record = empty_record()
            record.update(
                {
                    "run_id": run_id_box[0],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "git_commit": _git_commit(),
                    "upstream_commits": upstream_commit_map(),
                    "environment_id": detect_hardware().get("platform"),
                    "method": case.method,
                    "variant": case.variant,
                    "status": "success",
                    "error_reason": None,
                    "nodes": case.nodes,
                    "world_size": case.world_size,
                    "rank": rank,
                    "host_id": socket.gethostname(),
                    "gpu_model": torch.cuda.get_device_name(device),
                    "gpu_memory_mib": int(torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)),
                    "verified_transport": None,
                    "ulysses_degree": case.ulysses_degree,
                    "ring_degree": case.ring_degree,
                    "global_sequence_length": case.global_sequence_length,
                    "local_sequence_length": len(positions),
                    "batch_size": case.batch_size,
                    "heads": case.heads,
                    "head_dim": case.head_dim,
                    "dtype": case.dtype,
                    "causal": True,
                    "phase": case.phase,
                    "seed": seed,
                    "repeat": 0,
                    "iteration": iteration,
                    "step_seconds": step_seconds,
                    "max_rank_step_seconds": max_seconds,
                    "global_tokens_per_second": global_tokens_per_second(case.batch_size, case.global_sequence_length, max_seconds),
                    "peak_allocated_mib": allocated,
                    "peak_reserved_mib": reserved,
                    "device_peak_sampled_mib": None,
                    "compute_active_ms": None,
                    "communication_active_ms": None,
                    "overlap_ms": None,
                    "overlap_fraction": None,
                    "valid_causal_pairs": valid_pairs,
                    "trace_path": None,
                    "origin": "measured",
                }
            )
            records.append(record)
        if args.output:
            destination = Path(args.output)
            if case.world_size > 1:
                destination = destination.with_name(f"{destination.stem}.rank{rank}{destination.suffix}")
        else:
            destination = Path("results/raw") / run_id_box[0] / f"rank{rank}.jsonl"
        write_jsonl(destination, records)
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
