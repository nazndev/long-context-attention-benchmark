# Long-sequence attention benchmark

CSE598/EEE589 Project 05. Compare Ring Attention, DistFlashAttn, DeepSpeed Ulysses, and Unified Sequence Parallelism on long causal sequences. The experiment definition that belongs in the repository is `configs/benchmark.yaml`.

Phase 1 is a CPU-checked implementation and a Colab-ready single-GPU baseline. It does not contain GPU measurements. `reports/preliminary_findings.md` separates analytical checks, measurements, and pending work.

## Run notebooks 01 and 02 on a fresh Colab GPU runtime

The configuration cell in each notebook already points at this repository and commit `6befb712fb4666afbf8961f484f24e0cf4c9511d`.

### Notebook 01

1. In Colab, open `notebooks/01_colab_setup_and_preliminary.ipynb` from GitHub.
2. **Runtime → Change runtime type → GPU → Save.**
3. **Runtime → Run all.** Do not edit the configuration cell.
4. Download the evidence ZIP from the last cell before the session ends.

The installer keeps the PyTorch that is already in the runtime. If that PyTorch is missing, the notebook stops. It does not install a CPU wheel over a CUDA runtime, and it does not install `requirements-cpu-dev.txt`.

### Notebook 02

Use a new runtime so the baseline does not depend on notebook 01's kernel state.

1. Open `notebooks/02_colab_single_gpu_baseline.ipynb` from GitHub in a new runtime.
2. **Runtime → Change runtime type → GPU → Save.**
3. **Runtime → Run all.** Do not edit the configuration cell.
4. Download the baseline ZIP from the last cell.

Notebook 02 prints the GPU name, architecture, total memory, and free memory. It selects BF16 only on compute capability 8.0 or newer when BF16 is supported. Otherwise the rows are FP16. It names the PyTorch SDPA backend that completed a tiny probe. Forward and forward-plus-backward are timed in separate subprocesses for lengths 1024, 2048, 4096, 8192, and 16384. An out-of-memory failure stops that phase and records the last success and the first OOM. If every length finishes, the manifest records a tested upper bound and does not call it the hardware maximum. Throughput and memory figures are written only from successful measured rows, one pair of figures per phase.

Bring those downloaded ZIPs back. They are the preliminary empirical findings for the multi-GPU request. This Mac cannot produce them.

Notebook 03 launches `torchrun` only when the runtime sees at least two CUDA GPUs and the upstream package imports. Two Colab tabs are not a 100 Gb Ethernet experiment. Notebook 04 only plots rows with `origin=measured` and `status=success`.

## Local CPU checks

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch pyyaml pandas matplotlib pytest nbformat nbclient ipykernel jupyter
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q
.venv/bin/python -m spbench.sweep --config configs/benchmark.yaml --dry-run \
  --include-hypothetical-plan --output results/processed/dry_run_matrix.json
.venv/bin/python -m spbench.benchmark --method ring --seq-len 4096 --phase forward_backward
```

The benchmark command exits 2 on this machine and writes a blocked record if `--output` is set. It does not fall back to dense attention.

## Server commands, after approval

```bash
bash scripts/preflight.sh
bash scripts/run_single_node.sh
```

`run_single_node.sh` exits 2 when `nvidia-smi` is missing. The two-node script also requires `MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK`, and `NCCL_SOCKET_IFNAME`.

Details, licenses, and the commit pins are in `docs/upstream_audit.md` and `docs/reproducibility.md`.
