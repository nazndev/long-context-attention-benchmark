"""Write the four project notebooks. Run from the repository root."""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP = '''import os
import subprocess
import sys
import zipfile
from pathlib import Path

def discover(start):
    root = Path(start).resolve()
    pool = [root, *root.parents]
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and child.name not in {".venv", ".git"}:
                pool.append(child)
                for grandchild in child.iterdir():
                    if grandchild.is_dir() and grandchild.name not in {".venv", ".git"}:
                        pool.append(grandchild)
    for candidate in pool:
        if (candidate / "src" / "spbench" / "__init__.py").is_file():
            return candidate
    return None

if ZIP_PATH:
    dest = Path("/content/long-context-attention-benchmark" if os.path.isdir("/content") else "colab_unzipped")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(dest)
    project = discover(dest)
elif REPO_URL:
    if not REPO_COMMIT:
        raise ValueError("Set REPO_COMMIT to a full commit hash. Do not track a moving branch.")
    dest = Path("/content/long-context-attention-benchmark" if os.path.isdir("/content") else "colab_clone")
    if not (dest / ".git").exists():
        subprocess.check_call(["git", "clone", REPO_URL, str(dest)])
    subprocess.check_call(["git", "checkout", REPO_COMMIT], cwd=dest)
    project = discover(dest)
else:
    project = Path(PROJECT_DIR).resolve() if PROJECT_DIR else discover(Path.cwd())

if project is None:
    raise FileNotFoundError(
        "No project source found. On a fresh Colab runtime, upload the project ZIP and set ZIP_PATH, "
        "or set REPO_URL and REPO_COMMIT."
    )
os.chdir(project)
src = project / "src"
for entry in (src, project):
    entry_text = str(entry)
    if entry_text not in sys.path:
        sys.path.insert(0, entry_text)
print("Project directory:", project)
'''

INSTALL = '''import importlib.util
import sys
bootstrap_path = project / "scripts" / "colab_bootstrap.py"
spec = importlib.util.spec_from_file_location("colab_bootstrap", bootstrap_path)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
bootstrap.install_project(project)
src = project / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
import spbench
print("spbench", spbench.__version__)
'''

CONFIG = '''# These values clone the project on a fresh Colab runtime.
# Leave ZIP_PATH and PROJECT_DIR empty.
REPO_URL = "https://github.com/nazndev/long-context-attention-benchmark.git"
REPO_COMMIT = "6befb712fb4666afbf8961f484f24e0cf4c9511d"
ZIP_PATH = ""
PROJECT_DIR = ""
'''


def notebook(cells):
    nb = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return nb


def write(name, cells):
    path = ROOT / "notebooks" / name
    nbformat.write(notebook(cells), path)
    print("wrote", path)


def md(text):
    return new_markdown_cell(text.strip() + "\n")


def code(text):
    return new_code_cell(text.strip() + "\n")

write(
    "01_colab_setup_and_preliminary.ipynb",
    [
        md("""
# CSE598/EEE589 Project 05 — setup and preliminary checks

**Long Sequence Attention Across GPUs.** This notebook prepares a fresh runtime, runs the CPU correctness tests, shows the analytical causal-imbalance model, and classifies the experiment matrix. It does not measure distributed throughput.

## Run this on a fresh Google Colab runtime

1. In Colab, open this notebook from GitHub: `nazndev/long-context-attention-benchmark`, file `notebooks/01_colab_setup_and_preliminary.ipynb`.
2. **Runtime → Change runtime type → GPU → Save.**
3. The configuration cell already has the repository URL and commit. Do not change it.
4. **Runtime → Run all.**
5. Download the ZIP the last cell writes before the runtime is recycled.

The installer keeps Colab's existing PyTorch. It does not install `requirements-cpu-dev.txt`.
        """),
        md("## Configuration"),
        code(CONFIG),
        md("## Repository bootstrap"),
        code(BOOTSTRAP),
        md("## Install the project without upgrading an existing PyTorch"),
        code(INSTALL),
        md("## Runtime and dependency preflight"),
        code("""
import json
from spbench.dependencies import probe_dependencies
from spbench.hardware import detect_hardware

hardware = detect_hardware()
dependencies = probe_dependencies()
print(json.dumps({"hardware": hardware, "dependencies": dependencies}, indent=2, default=str))
if hardware["mps_available"] and not hardware["cuda_available"]:
    print("Apple MPS is visible. It is not used as an NVIDIA baseline.")
        """),
        md("## CPU reference and partition tests"),
        code("""
import subprocess
import sys
completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=project, text=True, capture_output=True)
print(completed.stdout)
print(completed.stderr)
if completed.returncode != 0:
    raise RuntimeError("CPU tests failed. Distributed timing was not started.")
print("CPU tests passed. This does not establish GPU or NCCL correctness.")
        """),
        md("## Analytical causal imbalance"),
        code("""
from spbench.demo import write_causal_imbalance_figure
summary = write_causal_imbalance_figure(project / "results" / "figures" / "analytical" / "causal_imbalance_s1024_p8.png")
print(summary["weak_context_note"])
for layout, ratios in summary["imbalance"].items():
    print(layout, {key: round(value, 4) for key, value in ratios.items()})
print("Figure:", summary["path"])
print("These pair counts are formulas, not kernel times.")
        """),
        md("## Dry-run experiment matrix"),
        code("""
from spbench.sweep import main as sweep_main
code = sweep_main([
    "--config", str(project / "configs" / "benchmark.yaml"),
    "--dry-run",
    "--include-hypothetical-plan",
    "--output", str(project / "results" / "processed" / "dry_run_matrix.json"),
])
if code != 0:
    raise RuntimeError(f"dry-run exited {code}")
print("Dry-run classification finished. Rejected rows are illegal configurations. Blocked rows were not executed.")
        """),
        md("## Export the preliminary evidence"),
        code("""
import zipfile
from datetime import datetime, timezone
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
archive_path = project / "results" / f"phase1_evidence_{stamp}.zip"
include = [
    project / "reports" / "preliminary_findings.md",
    project / "STATUS.md",
    project / "results" / "processed" / "dry_run_matrix.json",
    project / "results" / "figures" / "analytical" / "causal_imbalance_s1024_p8.png",
    project / "configs" / "benchmark.yaml",
    project / "docs" / "upstream_audit.md",
]
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in include:
        if path.is_file():
            archive.write(path, path.relative_to(project))
        else:
            print("missing, not packaged:", path)
print("Evidence archive:", archive_path)
print("Download this file from the Colab file browser, or copy it out of results/ locally.")
        """),
    ],
)

write(
    "02_colab_single_gpu_baseline.ipynb",
    [
        md("""
# CSE598/EEE589 Project 05 — single-GPU baseline

This notebook is independent of notebook 01. It detects the assigned CUDA device, selects BF16 or a separately labeled FP16, names the PyTorch SDPA backend that actually runs, and times forward and forward-plus-backward separately. It does not call the CPU schedule and it does not time Ring, DistFlashAttn, Ulysses, or USP. Without CUDA it records a skip.

## Fresh Colab runtime

1. Open `notebooks/02_colab_single_gpu_baseline.ipynb` from the same GitHub repository in a new Colab runtime.
2. **Runtime → Change runtime type → GPU → Save.**
3. The configuration cell already has the repository URL and commit. Do not change it.
4. **Runtime → Run all.**
5. Download the baseline ZIP from the last cell. If the manifest says `skipped_no_cuda`, no throughput was measured.

The bounded lengths are 1024, 2048, 4096, 8192, and 16384. A longer length is a separate decision, not the default. FP16 results, if a T4 forces that dtype, stay labeled FP16.
        """),
        md("## Configuration"),
        code(CONFIG),
        md("## Independent bootstrap"),
        code(BOOTSTRAP),
        code(INSTALL),
        md("## Dtype and backend support"),
        code("""
import json
from spbench.baseline import detect_baseline_support
support = detect_baseline_support()
public = {key: value for key, value in support.items() if key != "torch_dtype"}
print(json.dumps(public, indent=2, default=str))
        """),
        md("## Small forward and gradient check, only if CUDA is ready"),
        code("""
from spbench.baseline import correctness_against_reference
from spbench.config import load_config
config = load_config(project / "configs" / "benchmark.yaml")
rows = correctness_against_reference(support, config["correctness_lengths"], config)
for row in rows:
    print(row)
        """),
        md("## Bounded single-GPU sweep in subprocesses"),
        code("""
from spbench.baseline import sweep_subprocess
manifest = sweep_subprocess(str(project / "configs" / "benchmark.yaml"), str(project / "results" / "raw" / "baseline"))
print(manifest["support"].get("status"), manifest["support"].get("reason"))
print("backend:", manifest["support"].get("backend"), "dtype:", manifest["support"].get("dtype"))
print("GPU:", manifest["support"].get("gpu_names"), manifest["support"].get("gpu_architectures"))
print("memory MiB total/free:", manifest["support"].get("gpu_memory_mib"), manifest["support"].get("gpu_free_mib"))
for item in manifest.get("capacity", []):
    print("capacity", item["phase"], item["capacity_class"], "last_success", item["last_success_length"], "first_oom", item["first_confirmed_oom_length"], "tested_upper_bound", item.get("tested_upper_bound"))
        """),
        md("## Plots only from measured success rows"),
        code("""
import json
from pathlib import Path
import pandas as pd
from spbench.plots import plot_memory, plot_throughput
from spbench.schema import read_jsonl

frames = []
for path in sorted((project / "results" / "raw" / "baseline").glob("*.jsonl")):
    frames.extend(read_jsonl(path))
frame = pd.DataFrame(frames) if frames else pd.DataFrame()
if frame.empty or "origin" not in frame.columns:
    measured = frame.iloc[0:0]
else:
    measured = frame[(frame["origin"] == "measured") & (frame["status"] == "success")].copy()
if measured.empty:
    print("No measured baseline rows. No throughput or memory figure was written.")
else:
    figure_dir = project / "results" / "figures" / "measured"
    for phase, group in measured.groupby("phase"):
        plot_throughput(group, figure_dir / f"single_gpu_throughput_{phase}.png")
        plot_memory(group, figure_dir / f"single_gpu_memory_{phase}.png")
        print("Wrote measured figures for", phase, "from", len(group), "rows.")
(project / "results" / "processed" / "baseline_support.json").write_text(json.dumps(public, indent=2, default=str))
        """),
        md("## Export"),
        code("""
import zipfile
from datetime import datetime, timezone
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
archive_path = project / "results" / f"baseline_evidence_{stamp}.zip"
root = project / "results"
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in (project / "results" / "raw" / "baseline").glob("*"):
        if path.is_file():
            archive.write(path, path.relative_to(project))
    support_path = project / "results" / "processed" / "baseline_support.json"
    if support_path.is_file():
        archive.write(support_path, support_path.relative_to(project))
    figure_dir = project / "results" / "figures" / "measured"
    if figure_dir.is_dir():
        for path in figure_dir.glob("single_gpu_*.png"):
            archive.write(path, path.relative_to(project))
print("Baseline archive:", archive_path)
        """),
    ],
)

write(
    "03_distributed_gpu_experiments.ipynb",
    [
        md("""
# CSE598/EEE589 Project 05 — distributed experiments

This notebook repeats setup, then launches distributed workers only when at least two NVIDIA GPUs are visible to this runtime. Workers are external `torchrun` processes. They are not started inside the notebook kernel. Two ordinary Colab sessions are not a 100 Gb Ethernet cluster.

If the gate fails, the notebook prints the blocker and the server commands. It does not fabricate ranks, bandwidth, or overlap.
        """),
        md("## Configuration"),
        code(CONFIG),
        md("## Setup"),
        code(BOOTSTRAP),
        code(INSTALL),
        md("## GPU-count gate"),
        code("""
import json
from spbench.dependencies import probe_dependencies
from spbench.hardware import detect_hardware
hardware = detect_hardware()
dependencies = probe_dependencies()
gpu_count = int(hardware["cuda_device_count"])
print(json.dumps({"cuda_device_count": gpu_count, "gpu_names": hardware["gpu_names"], "dependencies": dependencies}, indent=2))
launch = gpu_count >= 2 and dependencies.get("ring_flash_attn", False)
print("NCCL launch permitted for Ring:" , bool(launch))
print("True USP (both degrees greater than 1) needs at least 4 visible GPUs. Visible:", gpu_count)
if gpu_count < 2:
    print("BLOCKER: fewer than 2 CUDA devices.")
    print("Single-node server command, after nvidia-smi and dependency installation:")
    print("torchrun --standalone --nproc-per-node=2 -m spbench.benchmark --method ring --seq-len 4096 --phase forward_backward")
    print("Two-node server command, after the 100 GbE interface is verified:")
    print("torchrun --nnodes=2 --nproc-per-node=2 --node-rank=0 --master-addr=<approved_node0_ip> --master-port=<approved_port> -m spbench.benchmark --method usp --ulysses-degree 2 --ring-degree 2 --seq-len 4096 --phase forward_backward")
    print("No distributed worker was started.")
        """),
        md("## External torchrun, only when the gate is open"),
        code("""
import subprocess
import sys
if gpu_count < 2:
    print("Skipped torchrun. The gate above explains why.")
else:
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={min(gpu_count, 2)}",
        "-m", "spbench.benchmark", "--method", "ring", "--variant", "basic", "--seq-len", "128", "--phase", "forward",
        "--output", str(project / "results" / "raw" / "ring_smoke.jsonl"),
    ]
    completed = subprocess.run(command, cwd=project, text=True, capture_output=True, timeout=180)
    print(completed.stdout)
    print(completed.stderr)
    print("return code", completed.returncode)
    if completed.returncode != 0:
        print("The worker failed. Its status is a failure, not a throughput result.")
        """),
    ],
)

write(
    "04_results_and_decision_guide.ipynb",
    [
        md("""
# CSE598/EEE589 Project 05 — results and decision guide

Load exported runs, check that hardware and dtype can be compared, and draw measured figures only from successful rows. Untested choices stay marked untested. This notebook does not declare a winning algorithm.
        """),
        md("## Configuration"),
        code(CONFIG + '\nRESULTS_DIR = ""\n'),
        md("## Setup"),
        code(BOOTSTRAP),
        code(INSTALL),
        md("## Load runs and validate the schema"),
        code("""
import json
from pathlib import Path
import pandas as pd
from spbench.schema import read_jsonl

results_root = Path(RESULTS_DIR) if RESULTS_DIR else project / "results"
rows = []
for path in sorted(results_root.rglob("*.jsonl")):
    rows.extend(read_jsonl(path))
frame = pd.DataFrame(rows) if rows else pd.DataFrame()
print("jsonl rows", len(frame))
if frame.empty:
    print("No JSONL runs were found. Measured plots and the decision guide stay empty.")
else:
    print(frame.groupby(["origin", "status", "method"], dropna=False).size())
plan_path = results_root / "processed" / "dry_run_matrix.json"
if plan_path.is_file():
    plan = json.loads(plan_path.read_text())
    print("dry-run counts", plan["counts"])
    print("dry-run measured flag", plan["measured"])
        """),
        md("## Measured plots, or an explicit skip"),
        code("""
from spbench.plots import plot_memory, plot_throughput
measured = pd.DataFrame()
if not frame.empty and "origin" in frame.columns:
    measured = frame[(frame["origin"] == "measured") & (frame["status"] == "success")].copy()
if measured.empty:
    print("No successful measured rows. Refusing to draw throughput or memory figures.")
else:
    comparable = measured.groupby(["dtype", "gpu_model", "topology"], dropna=False).size()
    print("Comparable groups (do not pool across these keys):")
    print(comparable)
    plot_throughput(measured, project / "results" / "figures" / "measured" / "throughput.png")
    plot_memory(measured, project / "results" / "figures" / "measured" / "memory.png")
        """),
        md("## Decision guide status"),
        code("""
guide = project / "reports" / "generated" / "decision_guide_status.md"
guide.parent.mkdir(parents=True, exist_ok=True)
if measured.empty:
    text = (
        "# Decision guide status\\n\\n"
        "No successful measured distributed runs are loaded. "
        "A choice by sequence length, GPU count, or network cannot be made. "
        "Ring Attention, DistFlashAttn, DeepSpeed Ulysses, and USP remain untested on the required hardware.\\n"
    )
else:
    text = (
        "# Decision guide status\\n\\n"
        f"Loaded {len(measured)} successful measured rows. "
        "Inspect dtype, GPU model, and topology before comparing them. "
        "This notebook does not name a universal winner.\\n"
    )
guide.write_text(text)
print(guide.read_text())
        """),
    ],
)

if __name__ == "__main__":
    pass
