"""Contracts for the Colab single-GPU path. These do not time a GPU."""

from spbench.baseline import PHASES, capacity_from_attempts, sweep_commands
from spbench.provenance import git_commit, torch_was_preserved
from scripts.make_colab_package import should_pack


def test_torch_guard_rejects_a_replaced_or_missing_install():
    assert torch_was_preserved(("2.15.0", "12.4"), ("2.15.0", "12.4"))
    assert not torch_was_preserved(("2.15.0", "12.4"), ("2.16.0", "12.4"))
    assert not torch_was_preserved(("2.15.0", "12.4"), ("2.15.0", None))
    assert not torch_was_preserved((None, None), ("2.15.0", None))


def test_sweep_commands_cover_both_phases_separately():
    commands = sweep_commands("configs/benchmark.yaml", "results/raw/baseline", [1024, 2048])
    rendered = [" ".join(command) for command in commands]
    assert len(rendered) == 4
    assert any("--phase forward " in command and "--seq-len 1024" in command for command in rendered)
    assert any("--phase forward_backward " in command and "--seq-len 2048" in command for command in rendered)
    assert set(PHASES) == {"forward", "forward_backward"}


def test_capacity_summary_does_not_promote_a_censored_bound():
    finished = capacity_from_attempts(
        "forward",
        [{"sequence_length": 1024, "returncode": 0}, {"sequence_length": 2048, "returncode": 0}],
        "ready",
    )
    assert finished["capacity_class"] == "censored"
    assert finished["last_success_length"] == 2048
    assert finished["first_confirmed_oom_length"] is None
    assert finished["tested_upper_bound"] == 2048
    assert finished["maximum_length_claimed"] is False
    oom = capacity_from_attempts(
        "forward_backward",
        [{"sequence_length": 1024, "returncode": 0}, {"sequence_length": 2048, "returncode": 3}],
        "ready",
    )
    assert oom["capacity_class"] == "oom_confirmed"
    assert oom["last_success_length"] == 1024
    assert oom["first_confirmed_oom_length"] == 2048


def test_git_commit_is_absent_or_a_full_hash():
    commit = git_commit()
    if commit is None:
        return
    assert len(commit) == 40
    int(commit, 16)


def test_colab_package_omits_internal_and_cpu_lock_files():
    assert not should_pack("CSE598_Project_05_Colab_Research_Plan_v3.yml")
    assert not should_pack("requirements-cpu-dev.txt")
    assert not should_pack(".venv/lib/python3.12/site-packages/torch/__init__.py")
    assert not should_pack("results/raw/baseline/baseline_manifest.json")
    assert should_pack("src/spbench/baseline.py")
    assert should_pack("notebooks/02_colab_single_gpu_baseline.ipynb")
    assert should_pack("configs/benchmark.yaml")
    assert should_pack("STATUS.md")
