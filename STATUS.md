# Requirement traceability

Course: CSE598/EEE589 Project 05. Experiment definition in this repository: `configs/benchmark.yaml`.

Status values: `not_started`, `partial`, `blocked`, `verified`. A blocked or skipped item is not verified. Phase 1 local checks passed. The project is not complete.

Host for every command below: Apple M4, no NVIDIA GPU. Python 3.12 virtualenv `.venv`, torch 2.15.0 with no CUDA build. Date: 2026-11-24.

## Professor requirements

| ID | Status | Evidence | What is missing |
| --- | --- | --- | --- |
| T1 | blocked | Adapters and `python -m spbench.benchmark` exist. On this host the command exits 2 and does not time Ring, DistFlashAttn, or Ulysses. | Successful runs from 2 GPUs upward, with a per-method capacity table and real OOM or censored status. No run IDs yet. |
| T2 | blocked | Topology fields and the Nsight command builder are in the code. `scripts/preflight.sh` recorded that `nvidia-smi` is absent and refused to guess a NIC. | Matched PCIe and 100 GbE runs, verified transport, per-rank throughput and memory, and profiler traces. |
| T3 | partial | Genuine USP is defined as both degrees greater than 1. CPU checks match dense attention for legal layouts. Zigzag query-owned pairs are exactly balanced: each rank owns \(c(S+1)\) pairs. `basic` + `ring_low` is rejected (132 matrix rows) because it is not causal. | A successful 4-GPU (or larger) hybrid run, and measured per-rank kernel time. The pair-count identity is not that measurement. |
| D1 | blocked | Plot functions exist and refuse fixture rows. Notebook 02 and notebook 04 wrote no measured figure. One analytical figure exists. | Throughput and memory plots from measured runs, for each method, GPU count, and sequence length. |
| D2 | not_started | `reports/decision_guide.md` and `reports/generated/decision_guide_status.md` both say no choice can be made. | A 1–2 page PDF whose choices cite measured run IDs. |
| P1 | partial | `reports/preliminary_findings.md`, `docs/upstream_audit.md`, CPU tests, dry-run, and locally executed notebooks. | The same notebooks have not been run inside Google Colab. The single-GPU baseline was skipped here because CUDA is absent. |

## Phase 1 acceptance

| Check | Status | Evidence |
| --- | --- | --- |
| CPU tests pass | verified | `.venv/bin/python -m pytest -q` → 33 passed. The original 28 are classified in `docs/phase1_technical_review.md`. Five more lock the Colab baseline contract. |
| Dry-run rejects illegal cases | verified | `results/processed/dry_run_matrix.json`. Counts: blocked 1144, duplicate 176, rejected 132, hypothetical 1144, planned 0. All 132 rejections are `usp/basic/ring_low`. |
| No fabricated GPU results | verified | No file under `results/figures/measured` except `.gitkeep`. Baseline manifest status is `skipped_no_cuda`. |
| Unresolved dependencies named | verified | `ring_flash_attn`, `lightseq`, `deepspeed`, `yunchang`, and `flash_attn` do not import. LightSeq has no license file. FlashAttention HEAD versus ring-flash-attention 0.1.8 is unresolved. |
| Notebooks valid and executed locally | verified | nbformat validation passed. Local kernel `spbench-cpu` executed every code cell in all four notebooks with no cell error. |
| Colab package prepared | verified | `python3 scripts/make_colab_package.py` writes `spbench_colab.zip`. It omits `.venv`, `requirements-cpu-dev.txt`, and the internal research plan. |
| Colab GPU execution | pending | Notebooks 01 and 02 have not been run inside Google Colab. Local execution only confirms the skip path. |

## Commands and results

```text
.venv/bin/python -m pytest -q
33 passed

.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 -m spbench.distributed_smoke
gloo_smoke rank=0 world=2
gloo_smoke rank=1 world=2
exit 0
```

This Gloo run is a process-group lifecycle check. It is not an NCCL measurement.

```text
bash scripts/preflight.sh
nvidia-smi: not installed
ETH_IFACE unset; no interface was labeled 100 Gb Ethernet
log: results/processed/preflight_local.txt

bash scripts/run_single_node.sh
nvidia-smi is missing. Refusing to launch a single-node GPU run.
exit 2
```

CPU FP32 schedule check, \(S=32\), 4 heads, head dimension 8, against a separate softmax. Not a GPU result.

| Check | max abs | relative L2 |
| --- | --- | --- |
| Ring basic, 4 ranks | 2.98e-7 | 1.16e-7 |
| Ring zigzag, 4 ranks | 2.98e-7 | 1.21e-7 |
| Ulysses, 4 ranks | 0 | 0 |
| DistFlashAttn, 4 ranks | 2.98e-7 | 1.16e-7 |
| USP 2×2 ulysses_low basic | 4.17e-7 | 1.23e-7 |
| USP 2×2 ring_low zigzag | 3.58e-7 | 1.14e-7 |

Ulysses is exactly zero because that checker reuses the reference softmax after the all-to-all. The zero means the exchange is bijective, not that a GPU kernel was tested. DistFlashAttn gradients at the same shape: dQ 7.15e-7, dK 3.73e-7, dV 5.96e-7 max abs.

## Notebook cells executed locally

Kernel `spbench-cpu`. No code cell raised.

| Notebook | Result |
| --- | --- |
| `notebooks/01_colab_setup_and_preliminary.ipynb` | Re-ran locally after the Colab fixes. CPU tests passed. Dry-run still has zero planned measured rows. The notebook shipped in `spbench_colab.zip` has no saved outputs, so a Colab run is not confused with this Mac. |
| `notebooks/02_colab_single_gpu_baseline.ipynb` | Re-ran locally. Status `skipped_no_cuda`. Capacity for both `forward` and `forward_backward` is `censored`, with no last success and no OOM, because no CUDA device exists. No throughput or memory figure was written. |
| `notebooks/03_distributed_gpu_experiments.ipynb` | Earlier local run printed `BLOCKER: fewer than 2 CUDA devices` and did not call torchrun. Not part of the Colab measurement target. |
| `notebooks/04_results_and_decision_guide.ipynb` | Earlier local run found 0 measured JSONL rows and refused measured plots. |

## Phase 1 technical review before Colab

Full classification of the original 28 tests, the CPU-versus-CUDA boundary, and the notebook cell review: `docs/phase1_technical_review.md`. Compatibility, LightSeq's missing license, odd DistFlashAttn GPU counts, and the invalid `basic` + `ring_low` layout: `docs/compatibility_matrix.md`.

Colab changes made after that review:

- The installer keeps the existing PyTorch version and CUDA build. It does not pip-install torch, and it does not install `requirements-cpu-dev.txt`.
- ZIP discovery searches one directory level below the extract root.
- The single-GPU sweep times `forward` and `forward_backward` separately, with a CUDA synchronize before and after each measured iteration.
- Records store GPU name, architecture, total and free memory, dtype, SDPA backend, dependency versions, config SHA-256, source-tree SHA-256, and `git_commit` (null until this directory is a git checkout).
- An OOM in a subprocess stops that phase. A finished grid is stored as `tested_upper_bound` with `maximum_length_claimed: false`.
- Throughput and memory figures are written only from `origin=measured` and `status=success`, separately per phase.

No distributed GPU benchmark was started.

Pending on Colab: the actual dtype, SDPA backend, synchronized timings, peak allocated memory, peak reserved memory, and the capacity line for this runtime's GPU. Pending on the server: every distributed method, both topologies, overlap traces, and the multi-GPU capacity search.

## Later phases

| Phase | Status |
| --- | --- |
| Phase 2 GPU correctness and the small optimizer training check | not_started. Needs at least 2 NVIDIA GPUs. |
| Phase 3 single-node scaling | not_started |
| Phase 4 two-node and genuine USP | not_started. Also needs verified 100 Gb Ethernet. |
| Phase 5 decision guide PDF and final report | not_started. `reports/final_report.md` is a placeholder. |
