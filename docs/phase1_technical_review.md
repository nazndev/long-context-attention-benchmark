# Phase 1 technical review

Date: 2026-11-24. Host: Apple M4, no NVIDIA GPU. This review does not add distributed GPU runs.

## 1. What the original 28 tests check

The count of 28 is the suite before the Colab-contract tests were added. Categories overlap when one test checks both a layout and a gradient.

### Mathematical correctness and backward gradients

These compare a CPU reconstruction with a separate FP32 softmax. They are not CUDA kernel tests.

| Test | What it checks |
| --- | --- |
| `test_reference_is_causal_and_future_tokens_do_not_leak` | Dense causal mask. Changing a future key or value leaves earlier outputs unchanged. |
| `test_ring_basic_and_zigzag_match_reference_and_gradients` | Ring basic and zigzag outputs, dQ, dK, and dV. Executed pair count equals the causal triangle. |
| `test_ring_basic_noncausal_matches_and_zigzag_noncausal_is_rejected` | Noncausal basic ring matches dense attention. Zigzag refuses noncausal because upstream asserts `causal == True`. |
| `test_ulysses_matches_reference_and_balances_heads` | Ulysses exchange plus the reference softmax. Error can be exact zero because the softmax is not independent. Head work is balanced. |
| `test_distflash_even_worlds_match_and_odd_world_is_rejected` | DistFlashAttn schedule at 2 and 4 ranks, including gradients. Odd world size raises. |
| `test_usp_layouts_that_preserve_causal_order_match_reference` | Legal USP layouts, including gradients. |
| `test_usp_ring_low_basic_does_not_match_dense_causal_attention` | Invalid USP rank map is numerically wrong. It is not relabeled as a working method. |

### Tensor layouts and communication schedules

| Test | What it checks |
| --- | --- |
| `test_contiguous_formula_matches_the_triangle` | Contiguous pair formula sums to \(S(S+1)/2\). |
| `test_small_contiguous_example` | S=4, P=2 gives 3 and 7 pairs. |
| `test_zigzag_pair_counts_are_exactly_equal_across_ranks` | Each zigzag rank owns \(c(S+1)\) pairs. |
| `test_zigzag_is_bijective_and_more_balanced_than_contiguous` | Zigzag positions are a permutation and are more balanced than contiguous ranks. |
| `test_usp_group_maps_and_token_coverage` | YunChang `ulysses_low` and `ring_low` groups, and bijective token ownership. |
| `test_only_ulysses_low_basic_preserves_causal_ring_order_at_world_4` | `basic` + `ring_low` does not keep causal order. |
| `test_idle_fractions_are_labeled_separately` | Paper idle fraction and the LightSeq `is_idle` slot fraction are different numbers. |
| `test_bad_head_split_is_rejected_by_the_work_model` | 6 heads cannot be split across 4 Ulysses ranks. |
| `test_position_generator_is_independent_of_shard_order` | The same global position produces the same value in every shard. |
| `test_dry_run_rejects_invalid_degree_and_alignment` | Head divisibility, zigzag alignment, and the invalid USP layout are rejected. The local matrix has zero planned GPU rows. |

### Metrics, schema, and launch safety

| Test | What it checks |
| --- | --- |
| `test_overlap_is_null_when_communication_is_absent` | Overlap is null when communication duration is zero. A partial overlap uses union-then-intersect. |
| `test_throughput_and_efficiency_do_not_invent_numbers` | Missing or zero time stays null. |
| `test_schema_rejects_zero_filled_failures` | A blocked row cannot store a zero step time. |
| `test_schema_round_trip_keeps_nulls` | JSONL keeps null measurements. |
| `test_sweep_cli_writes_a_plan_without_measurements` | Dry-run JSON is marked `measured: false`. |
| `test_benchmark_cli_refuses_without_cuda` | `python -m spbench.benchmark --method ring` exits 2 and writes a blocked row. |
| `test_research_plot_rejects_fixtures_and_analytical_plot_is_labeled` | Fixture rows cannot be saved as measured figures. |
| `test_capacity_lower_bound_is_not_a_maximum` | A time cap or missing GPU is censored, not a maximum length. |
| `test_nsys_command_names_child_cuda_and_nccl_traces` | The Nsight command string. No trace was collected. |
| `test_gloo_smoke_starts_and_returns` | Two Gloo processes start, barrier, and exit. This is not NCCL. |
| `test_hypothetical_hardware_is_not_the_detected_machine` | The 8-GPU plan is labeled hypothetical. |

## 2. CPU emulation versus upstream CUDA

| Path | Role |
| --- | --- |
| `src/spbench/algorithms/` | CPU schedule emulation used by the tests above. |
| `src/spbench/adapters/` | Call sites for ring-flash-attention, LightSeq, DeepSpeed, and YunChang. They raise if CUDA or the package is missing. |
| `src/spbench/baseline.py` | Single-GPU PyTorch SDPA only. `cpu_schedule` is false on its records. |
| `src/spbench/benchmark.py` | Distributed timing entry. On this host it exits 2. |

## 3. Notebooks 01 and 02 on a fresh runtime

Reviewed cell by cell after the Colab fixes:

Notebook 01. Configuration cell only assigns paths. Bootstrap unzips or clones, then searches one directory level below the extract root so a wrapped ZIP still resolves. Install runs `pip install -e . --no-deps` and then PyYAML, pandas, matplotlib, and pytest, without upgrading an existing torch. If torch is missing, the installer stops instead of fetching a CPU wheel. The next cell prints hardware and import probes. Pytest then runs the CPU suite. The analytical figure is labeled as a formula. The dry-run writes a plan, not measurements. The last cell zips evidence.

Notebook 02 repeats that bootstrap so it does not depend on notebook 01. It prints architecture, total memory, free memory, dtype, and the SDPA backend probe. The small-length cell compares forward outputs and gradients with FP32 attention only when CUDA is ready; otherwise it prints a skip. The sweep launches one subprocess per length and per phase (`forward`, then `forward_backward`). An OOM return stops that phase. If the grid finishes, the capacity record is `censored` with `tested_upper_bound` set and `maximum_length_claimed` false. Plots are written only from `origin=measured` and `status=success`, separately for each phase.

Notebooks 03 and 04 were not the execution target. Notebook 03 still refuses to start `torchrun` below two CUDA devices.

## 4. Sensitive-file review

Ignored and omitted from `spbench_colab.zip`: `.venv/`, `requirements-cpu-dev.txt`, the research-plan YAML, `results/raw`, `results/processed`, credential filenames, and `*.pem`. `results/processed/preflight_local.txt` records interface names without addresses and is gitignored with the rest of `results/processed`. No API keys were found in the project sources.

## 5. What remains pending

Google Colab has not executed these notebooks in this session. Until it does, there is no single-GPU throughput, peak allocated memory, or peak reserved memory. Distributed benchmarking stays blocked.
