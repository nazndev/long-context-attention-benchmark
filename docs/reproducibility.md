# Reproducibility

## What was executed in Phase 1

Host: Apple M4, macOS, no `nvidia-smi`, no CUDA device. PyTorch MPS can see the Apple GPU. Timed baselines do not use it.

Development environment, created in `.venv` and not installed onto Colab:

- Python 3.12
- torch 2.15.0, macOS arm64 wheel, CUDA build absent, `torch.distributed.is_nccl_available()` is false
- pytest 9.2.0, pandas 3.1.1, matplotlib 3.11.2, PyYAML 6.0.4, nbformat 5.11.1

The full pin of that environment is `requirements-cpu-dev.txt`. Installing it on Colab or on a Linux GPU server would replace the CUDA build of PyTorch. Do not do that.

Commands that were run from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch pyyaml pandas matplotlib pytest nbformat nbclient ipykernel jupyter
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 -m spbench.distributed_smoke
bash scripts/preflight.sh
.venv/bin/python -m spbench.sweep --config configs/benchmark.yaml --dry-run --include-hypothetical-plan --output results/processed/dry_run_matrix.json
```

`pytest` is the authority for the CPU checks. The Gloo smoke test starts two processes, barriers, and destroys the process group. It does not measure NCCL. `preflight.sh` records that `nvidia-smi` is absent and does not label a network interface.

Notebook execution status is in `STATUS.md`. A notebook cell that did not run on Colab is not described as a Colab result.

## Fresh Google Colab runtime

1. Open [Google Colab](https://colab.research.google.com/) and upload `notebooks/01_colab_setup_and_preliminary.ipynb` via **File → Upload notebook**.
2. **Runtime → Change runtime type → GPU → Save.** The assigned accelerator is whatever Colab provides. This project does not require a specific GPU name for the preliminary notebook.
3. On your computer, from this directory:

```bash
zip -r spbench_phase1.zip . -x '.venv/*' -x '**/__pycache__/*' -x '.git/*'
```

4. Upload `spbench_phase1.zip` in the Colab file browser so the path is `/content/spbench_phase1.zip`.
5. In the configuration cell set `ZIP_PATH = "/content/spbench_phase1.zip"`. Leave `REPO_URL` empty until a real remote and an immutable commit exist. Do not put credentials in the notebook.
6. **Runtime → Run all.**
7. Download the ZIP written under `results/` before the runtime is deleted.

Repeat the upload and **Run all** separately for `notebooks/02_colab_single_gpu_baseline.ipynb`. Notebook 02 is the single-GPU baseline. If the runtime has no CUDA device, its manifest says `skipped_no_cuda` and it writes no throughput.

Notebooks 03 and 04 are safe to open. Notebook 03 starts `torchrun` only when at least two CUDA devices are visible and the upstream import succeeds. Two Colab sessions are not a two-server Ethernet experiment.

The first code cells call `scripts/colab_bootstrap.py`. That installer runs `pip install -e . --no-deps` and then installs PyYAML, pandas, and matplotlib. If PyTorch is already importable, it is not upgraded.

## GPU server, after access is approved

Do not run the Mac lockfile. On the server:

```bash
bash scripts/preflight.sh
python -m torch.utils.collect_env
```

Record the driver, CUDA, NCCL, and `nvidia-smi topo -m` output in a new `environment.yaml` based on `configs/environment.example.yaml`. Set `NCCL_SOCKET_IFNAME` explicitly for any cross-node job. Do not force `NCCL_P2P_DISABLE`, `NCCL_IB_DISABLE`, or a GID index unless the topology check says so, and keep the same values for every method in one comparison.

Install the audited upstream commits only after checking that FlashAttention’s private API still matches ring-flash-attention 0.1.8 and yunchang 0.6.4. Pin whatever combination actually imports, and keep DistFlashAttn in its own checkout because it has no license file and is not packaged.

Then:

```bash
torchrun --standalone --nproc-per-node=2 -m spbench.benchmark \
  --method ring --seq-len 4096 --phase forward_backward
```

Profiling, only after a timed configuration has succeeded, and only with `nsys profile --help` checked on that machine:

```bash
nsys profile --trace=cuda,nvtx,nccl --cuda-trace-scope=process-tree \
  --output=traces/host0_ring \
  torchrun --standalone --nproc-per-node=2 -m spbench.benchmark \
  --method ring --seq-len 4096 --phase forward_backward
```

The trace file must be unique per host. Profiling runs are not the timing runs.

## Result rules

- JSONL fields that were not measured are `null`. They are not zero.
- `origin=measured` is required before a figure is written under `results/figures/measured`.
- CPU schedule checks and the analytical imbalance figure are not measurements.
- BF16 and FP16 stay in separate tables.
- A shorter successful length is not reported as the hardware maximum. See `spbench.capacity`.
