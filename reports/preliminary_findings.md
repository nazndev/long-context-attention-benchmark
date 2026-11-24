# Preliminary findings

**CSE598/EEE589 Project 05 — Long Sequence Attention Across GPUs**

Performance and scalability evaluation of Ring Attention, DistFlashAttn, DeepSpeed Ulysses, and Unified Sequence Parallelism.

This report is the early-findings and testing plan for the case where a multi-GPU NVIDIA server is not yet available. Statements are marked **analytical**, **measured**, or **pending**. Published speedups from the papers are claims by those authors. They are not results of this project.

## Research questions

1. How do throughput and peak memory scale with global sequence length and GPU count?
2. How does a single server (expected PCIe, only if `nvidia-smi topo` shows that) compare with two servers on verified 100 Gb Ethernet, at the same GPU count?
3. How much NCCL communication is overlapped with attention computation, using device intervals from Nsight rather than asynchronous API calls?
4. Does causal-aware token placement reduce rank imbalance, and which USP placement is better on the actual topology?

The working hypotheses, which are not findings, are:

- Ulysses may do well when the head count divides the Ulysses degree and all-to-all bandwidth is high.
- Ring may do better when the link is slow, because it moves KV only to a neighbor and can overlap that transfer with a block.
- DistFlashAttn may reduce the causal idle time that a plain ring leaves on early ranks.
- USP with Ulysses inside the node and ring across nodes may fit a machine whose NVLink or PCIe domain is faster than Ethernet.
- FlashAttention removes the stored \(S \times S\) score matrix. Exact attention still does quadratic arithmetic.

## What the four algorithms communicate

Notation: global sequence \(S\), GPUs \(P\), heads \(H\), head dimension \(D\), batch 1. The assigned model shape is \(H = 32\), \(D = 64\), hidden size 2048, causal attention, dropout 0, dtype BF16 unless a GPU without BF16 forces a separately labeled FP16 run.

### Ring Attention

Each rank holds \(S/P\) tokens. For \(P\) steps the KV block circulates to the neighbor. Basic causal code skips the attention *compute* for future blocks (`step > rank`) but still circulates KV. A full circulation sends about \(2 \cdot (S/P) \cdot H \cdot D \cdot (P-1)\) elements per rank for K and V in the forward pass. That count is **analytical**. Backward sends the gradient KV as well.

Per-rank activation storage stays linear in \(S/P\) if the implementation never assembles the score matrix. Arithmetic remains quadratic in \(S\). The early rank has less causal work than the late rank. For contiguous rank \(r\) and local length \(m = S/P\), valid pairs per head and batch element are

\[
m(2rm + m + 1)/2.
\]

The sum over ranks is \(S(S+1)/2\). This identity is checked in the CPU tests. For \(S = 4\), \(P = 2\) the counts are 3 and 7.

Zigzag gives rank \(r\) the chunk \(r\) and the chunk \(2P-1-r\) from a split into \(2P\) equal chunks of length \(c = S/(2P)\). The query-owned pair count on every rank is exactly \(c(S+1)\). It does not depend on \(r\), and the \(P\) ranks still sum to \(S(S+1)/2\). So the max/mean ratio is 1 and the coefficient of variation is 0 for every legal zigzag layout. Notebook 01 prints that result at \(S = 1024\), \(P = 8\): contiguous max/mean 1.8741, zigzag max/mean 1. The figure is `results/figures/analytical/causal_imbalance_s1024_p8.png`. This is an **analytical** identity about pair counts. It is not a statement about kernel time. A balanced pair count can still be unbalanced in the profile if the kernel’s block schedule or the network does not follow the pair count.

### DistFlashAttn

The sequence is also split into contiguous chunks of \(S/P\). Early ranks that would sit idle instead compute attention for a later rank’s queries against their own KV, then send the partial output and the softmax statistics back. The owner merges those statistics the same way FlashAttention merges two blocks. The paper’s published idle fraction is 0 for odd \(P\) and \(1/(2P)\) for even \(P\). The August 2024 code’s `is_idle` predicate does not use that denominator. Both are documented in `docs/upstream_audit.md`. Neither has been **measured** here.

The audited loop is correct for even \(P\) and is rejected for odd \(P\). Head dimension must be 16, 32, 64, or 128, and the local sequence must be a multiple of the kernel’s `BLOCK_M` of 128. The assigned \(D = 64\) satisfies the head-dimension rule. LightSeq has no license file. The kernels were not copied into this repository.

DistFlashAttn is not a relabeling of zigzag ring. Zigzag moves tokens. DistFlashAttn moves partial outputs while leaving the token shards contiguous.

### DeepSpeed Ulysses

Each rank starts with \(S/P\) tokens and all heads. An all-to-all gathers the sequence and scatters the heads, so each rank then holds the full sequence for \(H/P\) heads. A second all-to-all restores the layout. Forward communication is four all-to-alls (Q, K, V, and the output) unless an implementation fuses QKV. Per rank, one tensor of \(S/P \cdot H \cdot D\) elements sends the fraction \((P-1)/P\). If \(S\) grows in proportion to \(P\), that volume stays flat. That is the paper’s constant-communication claim. It is **analytical**.

Causal work per head is the full triangle on every rank, so the head split does not create the contiguous-token imbalance. The constraint is \(H\) divisible by the Ulysses degree. With 32 heads, degrees 2, 4, and 8 are legal. Degree 6 with 6 heads is rejected by the dry-run.

On current DeepSpeed, `DistributedAttention` defaults `gather_idx` to 0. The BSHD adapter passes `gather_idx=1` and `batch_dim_idx=0`. That call has not yet been checked on a GPU.

### Unified Sequence Parallelism

USP uses both mechanisms: Ulysses all-to-all inside one process group and ring exchange inside the orthogonal group. The product of the degrees is the world size. Both degrees must be greater than 1 for the run to count as genuine USP. `(1, 2)` and `(2, 1)` are controls that should reproduce pure ring and pure Ulysses.

`ulysses_low` puts consecutive ranks in the Ulysses group. With the usual `torchrun` numbering that keeps all-to-all inside a node and the ring across nodes. `ring_low` swaps the groups. For **zigzag**, both placements matched dense causal attention in the CPU checker. For **basic** contiguous chunks, `ring_low` scrambles causal order: ring rank 0 can hold chunks 0 and 2 while ring rank 1 holds chunks 1 and 3. Those 132 matrix rows are **rejected**, not scheduled. The CPU checker’s output for that layout is not close to the dense reference, which is why the rejection is in the code rather than left as a tuning choice.

## NVIDIA communication and profiling notes

Read from the NCCL 2.32.3 user guide, not from a local NCCL build (this Mac reports NCCL unavailable):

- `ncclAlltoAll` is the collective Ulysses needs. In-place all-to-all is documented as unsupported. Each rank’s send buffer holds `nranks * count` elements.
- Neighbor exchange is `ncclSend` / `ncclRecv`, fused with `ncclGroupStart` / `ncclGroupEnd` so the two directions progress together. That is the ring’s point-to-point pattern.
- `NCCL_IB_DISABLE=1` forces the socket path instead of InfiniBand or RoCE. `NCCL_P2P_LEVEL` selects how far CUDA peer access is used (NVLink, same PCI switch, across the CPU, and so on). These variables will be recorded and held fixed inside one comparison. They will not be guessed.
- `NCCL_SOCKET_IFNAME` chooses the Ethernet device. The preflight script refuses to pick an interface when `ETH_IFACE` is unset.

Read from the Nsight Systems user guide (2026 documentation set):

- The intended profile command is `nsys profile --trace=cuda,nvtx,nccl --cuda-trace-scope=process-tree`.
- `--cuda-trace-scope=process-tree` is what captures `torchrun`’s child processes.
- Overlap is the duration of the intersection of the union of compute intervals and the union of communication intervals, divided by the communication-union duration. If communication duration is zero, the fraction is null, not zero.
- `nsys` is not installed here. No trace exists. The command builder is tested as a string, not as a profile.

PCIe versus 100 Gb Ethernet is a required comparison. This machine cannot establish either link. A future single-node run will be labeled with the topology `nvidia-smi topo` actually prints. An NVLink machine will not be called PCIe unless the transport is forced and verified.

## What was actually run

| Kind | What | Result |
| --- | --- | --- |
| Analytical | Contiguous pair formula and zigzag balance | Identities hold in `tests/test_layouts_and_work.py` |
| Analytical | CPU reconstructions of ring, zigzag, Ulysses, DistFlashAttn, and legal USP placements | Match FP32 dense attention; see errors below |
| Measured GPU | Throughput, memory, overlap, NCCL | Not run. No CUDA device |
| Pending Colab | Notebook 02 on whatever GPU Colab assigns | Not run in this session |

CPU FP32, sequence 32, 4 heads, head dimension 8, one batch. Maximum absolute error and relative L2 against a separate softmax reference:

| Check | max abs | relative L2 |
| --- | --- | --- |
| Ring basic, 4 ranks | 2.98e-7 | 1.16e-7 |
| Ring zigzag, 4 ranks | 2.98e-7 | 1.21e-7 |
| DistFlashAttn, 4 ranks | 2.98e-7 | 1.16e-7 |
| USP 2×2, `ulysses_low`, basic | 4.17e-7 | 1.23e-7 |
| USP 2×2, `ring_low`, zigzag | 3.58e-7 | 1.14e-7 |
| DistFlashAttn dQ / dK / dV | 7.15e-7 / 3.73e-7 / 5.96e-7 | 2.05e-7 / 1.99e-7 / 1.42e-7 |

Ulysses at the same shape matched with error 0. That check reuses the reference softmax after the all-to-all, so a zero only means the exchange put the right tokens and heads back. It is not an independent kernel comparison.

The initial BF16 tolerance in the config is `rtol = atol = 0.05`. It was not used to accept these FP32 checks, and it will not be loosened later to hide a GPU mismatch. The same-precision GPU comparison still has to be reported with max absolute error and relative L2.

`pytest`: 27 passed, including the matrix rejections, the schema rule that a blocked row cannot store a zero time, and a two-process Gloo start/barrier/shutdown. Gloo is not NCCL.

Dry-run of `configs/benchmark.yaml` on this host, 1452 case rows:

| Status | Count | Meaning |
| --- | --- | --- |
| blocked | 1144 | Legal, but this host has 0 CUDA devices and the upstream packages are not installed |
| duplicate | 176 | At 2 GPUs, `ring_low` builds the same groups as `ulysses_low` |
| rejected | 132 | `usp` + `basic` + `ring_low` breaks causal token order |
| planned | 0 | Nothing was eligible to time |

A hypothetical plan that assumes the requested servers and installed dependencies has 1144 rows labeled `hypothetical`, plus the same 132 rejections and 176 duplicates. Those hypothetical rows are not measurements. Of the USP rows in that plan, 396 are genuine hybrids (both degrees greater than 1) and 176 are pure-method controls.

Weak-context points with 4096 or 8192 tokens per GPU at 2, 4, and 8 GPUs already fall inside the initial length grid (8192 through 65536). They do not need extra runs. They are still not constant FLOPs per GPU: each query attends to a longer prefix as the global length grows.

## Experiment matrix and OOM protocol

Initial lengths: 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576. GPU counts: 2, 4, 8, plus a supplemental single-GPU baseline. Topologies: one node, and two nodes with equal ranks per node. Phases: forward, and forward plus backward. Warmup 5, measured iterations 10, 3 repeats, method order shuffled inside a matched configuration, median and dispersion reported. Seed 42. The same position-addressed Q, K, V, and dO are used for every method. Timing includes communication, attention, and required layout changes. It excludes generation, validation gathers, and disk IO. Iteration-boundary CUDA synchronizations are outside the step so they do not split a overlapped send from its compute.

Throughput is \(B \cdot S / \max_r t_r\). Peak allocated and reserved memory are reset after warmup and reported per rank and as the max across GPUs, in MiB. NVML device memory is null until a sampler is available. Overlap is null until a trace supplies device intervals.

Capacity search, separately per method, topology, GPU count, dtype, and phase: start from a length that succeeded, double, then bisect the last success and the first confirmed OOM if time allows. Stop reasons:

- `oom_confirmed`: a CUDA OOM from a fresh process
- `success`: every rank finished
- `censored`: time cap, approved length cap, missing GPU, unsupported kernel, or a non-OOM failure

A censored boundary is not a maximum sequence length.

The single-GPU Colab sweep, when a GPU exists, is only 1024 through 16384 and is not evidence about distributed scaling.

## GPU allocation request

Ordinary Colab, including a single GPU, cannot answer the assignment:

- Task T1 needs Ring, DistFlashAttn, and Ulysses from 2 GPUs upward until memory fails.
- Task T2 needs the same methods on one server and on two servers linked by 100 Gb Ethernet, with throughput, peak memory, and overlap.
- Task T3 needs genuine USP, so both degrees exceed 1, which means at least 4 GPUs, plus a causal imbalance measurement.

Requested setup:

- One Linux server with at least 4 NVIDIA GPUs, preferably 8, identical models.
- A second server with the same GPU count and a verified 100 Gb Ethernet link. Please record whether that link is TCP sockets or RoCE. The paper’s own DistFlashAttn cluster used 100 Gb InfiniBand; the assignment asks for 100 Gb Ethernet, so those are not the same fabric and this project will not rename one as the other.
- `nvidia-smi topo` output so PCIe is not confused with NVLink.
- NCCL, Nsight Systems, and enough disk for per-rank JSONL and one trace set per method.
- No competing jobs, and permission to set clocks only if the site allows it.
- A decision on LightSeq: the paper is CC-BY-4.0, but the code repository has no license file. The kernels were not vendored. Instructor guidance is needed before that checkout is copied onto the server.

## Outstanding risks

1. FlashAttention HEAD is an FA4 commit as of 2026-11-23. ring-flash-attention 0.1.8 and yunchang 0.6.4 were written against FlashAttention 2.x private APIs. A joint environment is not pinned until those imports succeed on the server.
2. DeepSpeed’s BSHD index choice is audited but not numerically checked on CUDA. Pull request 7868 reports shape bugs on other paths.
3. Ring’s backward casts `dq` to BF16 inside the audited function. Gradient comparisons need the observed error, not only a pass/fail bit.
4. DistFlashAttn’s paper idle fraction and the code’s `is_idle` predicate disagree. The benchmark will plot both only after saying which one a trace supports.
5. `basic` + `ring_low` is an invalid causal layout. Zigzag does not have that defect in the CPU checker. The GPU run still has to confirm zigzag.
6. YunChang’s `AttnType.TORCH_EFFICIENT` has no backward. It will not be used for `forward_backward`.
7. This host has no measurement. Any later claim about which method is faster has to cite run IDs from `results/raw`.

## Local notebook execution

On 2026-11-24 the four notebooks were executed in the local Python 3.12 virtualenv with the `spbench-cpu` kernel. Every code cell finished without an error. That execution is not a Colab run and not a GPU run.

- Notebook 01 ran the CPU suite, wrote the analytical figure, and classified the matrix as 1144 blocked, 176 duplicate, 132 rejected, and 1144 hypothetical. Planned measured rows: 0. The suite later grew to 33 tests; see `STATUS.md`.
- Notebook 02 reported `skipped_no_cuda` and wrote no throughput or memory figure.
- Notebook 03 saw 0 CUDA devices, printed the server `torchrun` commands, and did not start a worker.
- Notebook 04 loaded 0 JSONL measurement rows and wrote `reports/generated/decision_guide_status.md` stating that no method can be chosen.

Colab still has to be run by uploading the ZIP. Until that happens, notebook 02’s GPU sweep, SDPA backend name, and any FP16-versus-BF16 label are **pending**.

## Next step

Run notebook 01, then notebook 02, on a fresh Colab GPU runtime using the instructions in `README.md`. Bring the downloaded manifests back. The distributed matrix stays blocked until the server request is approved.
