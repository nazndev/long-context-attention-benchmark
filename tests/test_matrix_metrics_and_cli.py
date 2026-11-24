import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from spbench.benchmark import main as benchmark_main
from spbench.capacity import capacity_record, classify_capacity
from spbench.cases import Case, expand_matrix, validate_case
from spbench.config import load_config
from spbench.hardware import detect_hardware, hypothetical_server_hardware
from spbench.inputs import generate_bshd
from spbench.metrics import global_tokens_per_second, overlap_fraction, scaling_efficiency
from spbench.plots import plot_analytical_causal_imbalance, plot_throughput
from spbench.profiling import build_nsys_command, overlap_from_device_intervals
from spbench.schema import empty_record, read_jsonl, write_jsonl
from spbench.errors import InvalidConfiguration, PlotDataError
from spbench.sweep import main as sweep_main


def test_position_generator_is_independent_of_shard_order():
    full = generate_bshd(range(6), 1, 2, 4, 42, "q")
    odd = generate_bshd([1, 3, 5], 1, 2, 4, 42, "q")
    even = generate_bshd([0, 2, 4], 1, 2, 4, 42, "q")
    assert torch.equal(odd, full[:, [1, 3, 5]])
    assert torch.equal(even, full[:, [0, 2, 4]])


def test_overlap_is_null_when_communication_is_absent():
    overlap_ms, fraction = overlap_fraction([(0.0, 5.0)], [])
    assert overlap_ms is None and fraction is None
    overlap_ms, fraction = overlap_fraction([(0.0, 10.0)], [(2.0, 6.0), (5.0, 8.0)])
    # Communication union is [2, 8] (6 ms). Intersection with compute is [2, 8] if compute covers it? compute is [0, 10], so 6/6.
    assert abs(fraction - 1.0) < 1e-12
    overlap_ms, fraction = overlap_fraction([(0.0, 3.0)], [(2.0, 6.0)])
    assert abs(overlap_ms - 1.0) < 1e-12
    assert abs(fraction - 0.25) < 1e-12


def test_throughput_and_efficiency_do_not_invent_numbers():
    assert global_tokens_per_second(1, 1024, None) is None
    assert global_tokens_per_second(1, 1024, 0.0) is None
    assert global_tokens_per_second(1, 1024, 0.5) == 2048
    assert scaling_efficiency(None, 10.0, 4, 1) is None
    assert scaling_efficiency(40.0, 10.0, 4, 1) == 1


def test_schema_rejects_zero_filled_failures(tmp_path: Path):
    record = empty_record()
    record["status"] = "blocked"
    record["step_seconds"] = 0.0
    try:
        write_jsonl(tmp_path / "bad.jsonl", [record])
    except InvalidConfiguration:
        return
    raise AssertionError("a blocked row must not store a zero time")


def test_schema_round_trip_keeps_nulls(tmp_path: Path):
    record = empty_record()
    record.update({"run_id": "r", "status": "blocked", "method": "ring", "error_reason": "no cuda"})
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [record])
    loaded = read_jsonl(path)
    assert loaded[0]["step_seconds"] is None
    assert loaded[0]["overlap_fraction"] is None


def test_dry_run_rejects_invalid_degree_and_alignment():
    config = load_config("configs/benchmark.yaml")
    common = dict(
        phase="forward",
        topology="single_node",
        nodes=1,
        world_size=4,
        global_sequence_length=128,
        dtype="bf16",
        causal=True,
        batch_size=1,
        heads=32,
        kv_heads=32,
        head_dim=64,
        hidden_size=2048,
        dropout=0.0,
        genuine_hybrid=True,
    )
    reasons = validate_case(
        Case(
            method="ulysses",
            variant="default",
            ulysses_degree=4,
            ring_degree=1,
            placement=None,
            **{**common, "heads": 6, "kv_heads": 6, "hidden_size": 384, "genuine_hybrid": False},
        )
    )
    assert any("divisible" in reason for reason in reasons)
    alignment = validate_case(
        Case(
            method="ring",
            variant="zigzag",
            ulysses_degree=1,
            ring_degree=4,
            placement=None,
            **{**common, "global_sequence_length": 100, "genuine_hybrid": False},
        )
    )
    assert alignment
    ring_low = validate_case(
        Case(
            method="usp",
            variant="basic",
            ulysses_degree=2,
            ring_degree=2,
            placement="ring_low",
            **common,
        )
    )
    assert any("causal ring order" in reason for reason in ring_low)
    hardware = detect_hardware()
    rows = expand_matrix(config, hardware, {})
    assert rows
    assert any(row.status == "rejected" for row in rows)
    for row in rows:
        if row.method in {"ring", "distflashattn", "ulysses", "usp"}:
            assert row.status != "planned"
        if row.status == "planned":
            assert row.method == "single_gpu_sdpa"
            assert hardware["cuda_available"]


def test_sweep_cli_writes_a_plan_without_measurements(tmp_path: Path):
    output = tmp_path / "plan.json"
    code = sweep_main(
        ["--config", "configs/benchmark.yaml", "--dry-run", "--include-hypothetical-plan", "--output", str(output)]
    )
    assert code == 0
    payload = json.loads(output.read_text())
    assert payload["measured"] is False
    assert payload["counts"].get("rejected", 0) > 0
    assert payload["counts"].get("hypothetical", 0) > 0
    assert "global_tokens_per_second" not in payload["cases"][0]


def test_benchmark_cli_refuses_without_cuda(tmp_path: Path):
    output = tmp_path / "blocked.jsonl"
    code = benchmark_main(
        ["--method", "ring", "--seq-len", "128", "--phase", "forward", "--output", str(output)]
    )
    assert code == 2
    rows = read_jsonl(output)
    assert rows[0]["status"] == "blocked"
    assert rows[0]["step_seconds"] is None


def test_research_plot_rejects_fixtures_and_analytical_plot_is_labeled(tmp_path: Path):
    analytical = pd.DataFrame(
        {
            "origin": ["analytical", "analytical"],
            "layout": ["contiguous", "zigzag"],
            "rank": [0, 0],
            "valid_causal_pairs": [3, 5],
        }
    )
    analytical_path = tmp_path / "analytical.png"
    plot_analytical_causal_imbalance(analytical, analytical_path)
    assert analytical_path.exists()
    fixture = pd.DataFrame(
        {
            "origin": ["fixture"],
            "status": ["success"],
            "method": ["ring"],
            "topology": ["single_node"],
            "global_sequence_length": [1024],
            "global_tokens_per_second": [10.0],
        }
    )
    try:
        plot_throughput(fixture, Path("results/figures/measured/should_not_exist.png"))
    except PlotDataError:
        return
    raise AssertionError("fixture rows must not become a measured figure")


def test_capacity_lower_bound_is_not_a_maximum():
    assert classify_capacity(last_success_length=1024, first_confirmed_oom_length=None, stopping_reason="time_cap") == "censored"
    record = capacity_record(
        method="ring",
        topology="single_node",
        gpu_count=2,
        dtype="bf16",
        phase="forward",
        last_success_length=1024,
        first_confirmed_oom_length=None,
        attempted_max_length=1024,
        stopping_reason="gpu_unavailable",
        run_ids=[],
    )
    assert record["capacity_class"] == "censored"
    assert record["maximum_length_claimed"] is False


def test_nsys_command_names_child_cuda_and_nccl_traces():
    command = build_nsys_command(["torchrun", "--nproc-per-node=2", "-m", "spbench.benchmark"], "traces/rank")
    rendered = " ".join(command)
    assert "--trace=cuda,nvtx,nccl" in rendered
    assert "--cuda-trace-scope=process-tree" in rendered
    summary = overlap_from_device_intervals(None, [(0.0, 1.0)])
    assert summary["overlap_fraction"] is None


def test_gloo_smoke_starts_and_returns():
    completed = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", "-m", "spbench.distributed_smoke"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "SPBENCH_SMOKE_BACKEND": "gloo"},
    )
    assert completed.returncode == 0, completed.stderr
    assert "gloo_smoke rank=0 world=2" in completed.stdout
    assert "gloo_smoke rank=1 world=2" in completed.stdout


def test_hypothetical_hardware_is_not_the_detected_machine():
    detected = detect_hardware()
    hypothetical = hypothetical_server_hardware()
    assert detected["hypothetical"] is False
    assert hypothetical["hypothetical"] is True
    assert hypothetical["two_node_100gbe_verified"] is True
    assert detected["two_node_100gbe_verified"] is False
