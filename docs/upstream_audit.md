# Upstream audit

Audit date: 2026-11-24. Commits below are the revisions whose source was read. None of Ring, DistFlashAttn, Ulysses, or USP was installed in the Phase 1 environment, and none was timed.

## Ring Attention

Paper: Hao Liu, Matei Zaharia, and Pieter Abbeel, “Ring Attention with Blockwise Transformers for Near-Infinite Context,” arXiv:2310.01889v4.

The paper’s code link, `https://github.com/lhao499/llm_large_context`, resolved to the same repository and commit as `https://github.com/haoliuhl/ringattention`:

- commit `d2ea1af9a288f85ea2fd74690ece1c16d2eebc83` (2025-10-13, “upgrade jax version”)
- license Apache-2.0
- import path `from ringattention import ringattention, blockwise_feedforward`
- the public example is a JAX `shard_map` over query/key/value blocks, with `query_chunk_size` and `key_chunk_size`

That repository is the algorithm reference. The timed PyTorch path is `https://github.com/zhuzilin/ring-flash-attention`:

- commit `786677930bce4f6022166899c88ce2c00c814ee2` (2025-09-10, “bump to 0.1.8”)
- PyPI/package version 0.1.8
- license MIT
- batch API: `ring_flash_attn_func`, `zigzag_ring_flash_attn_func`, and also `stripe_flash_attn_func`
- each has `*_func`, `*_kvpacked_func`, and `*_qkvpacked_func` forms
- tensors are BSHD; `dropout_p` defaults to 0; `group` is the process group
- zigzag’s forward function asserts `causal == True`
- basic causal attention computes a block only when `step <= rank`, and the triangular mask is used only on step 0
- KV is sent to the next rank and received from the previous rank

Known limitations stated in that README, and visible in `ring_flash_attn_backward`: block outputs are accumulated from FlashAttention’s low-precision values, and the basic backward path returns `dq` cast to `bfloat16`. Dropout and `window_size` are unsupported. The README also says high performance wants NVLink. A PCIe-versus-Ethernet comparison still has to be measured; the README’s NVLink remark is not a substitute for that measurement.

Stripe attention is implemented upstream and is not one of the assigned variants. This benchmark does not relabel it as Ring or DistFlashAttn.

## DistFlashAttn / LightSeq

Paper: Dacheng Li, Rulin Shao, Anze Xie, Eric P. Xing, Xuezhe Ma, Ion Stoica, Joseph E. Gonzalez, and Hao Zhang, “DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training,” arXiv:2310.03294v2. The paper text is marked CC-BY-4.0.

Repository: `https://github.com/RulinShao/LightSeq`

- commit `8f486dad3d0670057dfbe3b30c003080b61c5325` (2024-08-19)
- GitHub license API returned 404
- the tree contains `README.md` and `lightseq/` only, and there is no `LICENSE` file
- there is no `pyproject.toml` or package init
- the code was not copied into this project

Public entry point, from `lightseq_async_attn.py`:

```python
attention = _attention.apply  # attention(q, k, v, causal, sm_scale)
```

Q, K, and V are BHSD in the forward. Head dimension must be one of 16, 32, 64, or 128. `BLOCK_M` is 128 and `BLOCK_N` is 64. The forward imports Triton and, optionally, FlashAttention’s private backward. `initialize_distributed()` builds one sequence-parallel group over the whole world. `maybe_send_recv_fwd_qkvo` comments that only an even number of machines is supported. The `P // 2 + 1` step loop does not cover every causal KV block when `P` is odd, so odd world sizes are rejected.

The default `comm_mode` is `"lightseq"`. The same file contains a `"sync"` ablation. That ablation is not a second required algorithm.

Paper equation (2) says the load-balanced idle fraction is 0 when `P` is odd and `1/(2P)` when `P` is even. The audited `is_idle` predicate is true only for ranks below `P/2` at time step `P/2`. For `P = 8` that is 4 idle slots out of 40 rank-steps, which is 0.1, not `1/16`. Those two quantities are both recorded and are not treated as the same number. A GPU trace is required before either is called an observed bubble.

## DeepSpeed Ulysses

Paper: Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He, “DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models,” arXiv:2309.14509v2.

Repository: `https://github.com/deepspeedai/DeepSpeed`

- commit `140a2fbcd253d600e094ff7e659064180928246b` (2026-11-24)
- license Apache-2.0
- tutorial: `https://www.deepspeed.ai/tutorials/ds-sequence/` says DeepSpeed 0.10.2 or newer

The class this benchmark will call is `deepspeed.sequence.layer.DistributedAttention`. At the audited commit its signature is:

```python
DistributedAttention(
    local_attention,
    sequence_process_group,
    scatter_idx=2,
    gather_idx=0,          # class default
    sp_stream=None,
    num_kv_heads=None,
    validate_shards=False,
)
forward(query, key, value, batch_dim_idx, ...)
```

For BSHD tensors the head axis is 2 and the sequence axis is 1, so this project passes `scatter_idx=2`, `gather_idx=1`, and `batch_dim_idx=0`. Using the default `gather_idx=0` would be the wrong layout. `validate_shards=True` adds a metadata all-gather; the benchmark turns it on for correctness runs. Pull request 7868 was still open on the audit date and describes shape bugs for other index combinations. Those paths are not used. Numerical agreement on the GPU is still required before a timed sweep.

`deepspeed.runtime.sequence_parallel.ulysses_sp.UlyssesSPAttentionHF` is a different, sequence-first `[sequence, batch, heads, dim]` wrapper aimed at Hugging Face modules. It is not the microbenchmark adapter.

Heads must be divisible by the Ulysses degree. The assigned configuration has 32 heads, so degrees 2, 4, and 8 are legal. A case that breaks the rule is rejected rather than padded.

## USP / YunChang

Paper: Jiarui Fang and Shangchun Zhao, “USP: A Unified Sequence Parallelism Approach for Long Context Generative AI,” arXiv:2405.07719v5.

Repository: `https://github.com/feifeibear/long-context-attention`

- commit `a5f8f600d435ced180a97c91c68e87ff3cfb8f89` (2026-09-16)
- package `yunchang` 0.6.4
- license Apache-2.0
- `requires-python >= 3.7`, `torch>=2.3`, optional `flash-attn>=2.6`

Calls:

```python
set_seq_parallel_pg(ulysses_degree, ring_degree, rank, world_size, use_ulysses_low=True)
LongContextAttention(scatter_idx=2, gather_idx=1, ring_impl_type="basic", attn_type=AttnType.FA)
```

`set_seq_parallel_pg` must run before the module is constructed because the constructor reads a process-global singleton. `use_ulysses_low=True` makes Ulysses groups contiguous ranks and ring groups strided. `use_ulysses_low=False` swaps that. With `torchrun`’s usual numbering, contiguous ranks are the topology-aware choice when Ulysses should stay inside a node and the ring should cross nodes. The checked-in hybrid test assigns `use_ring_low_dim = True` and then calls `set_seq_parallel_pg` without passing it, so that test does not exercise the alternate placement.

Token extraction:

- `basic_extract_local` takes a contiguous global-rank chunk
- `zigzag_extract_local` chunks the sequence into `2 * ring_degree` pieces, keeps the ring rank’s pair of end chunks, then splits that pair across the Ulysses rank

The README says `AttnType.TORCH_EFFICIENT` has no backward. Forward/backward timing therefore requests `AttnType.FA` and does not silently switch. The hybrid test also names `TORCH_MATH`, `TORCH_FLASH`, `FA3`, and several other kernels. Those are not interchangeable measurement backends.

Genuine USP in this project means both degrees are greater than 1. Degree pairs `(1, 2)` and `(2, 1)` are pure-method controls.

## FlashAttention

Repository: `https://github.com/Dao-AILab/flash-attention`

- commit `b038b4974f26ba3bc8ab3c8b840c60b0432ce39b` (2026-11-23), an FA4-era change
- license BSD-3-Clause

`ring-flash-attention` 0.1.8 and `yunchang` 0.6.4 call FlashAttention 2 private functions (`_flash_attn_forward` / `_flash_attn_backward`) and branch on a 4-tuple versus an 8-tuple return. Installing current FlashAttention HEAD beside those packages is an unresolved compatibility question. Official FA2 CUDA kernels are not claimed for compute capability below 8.0. A Colab T4 is in that class. The single-GPU notebook then uses PyTorch SDPA and labels the backend it actually selected. SDPA’s math kernel is not called FlashAttention.

## What Phase 1 did not do

No upstream package was patched. No result from a paper’s tables was copied into `results/raw`. The CPU schedule modules reimplement control flow well enough to compare against dense attention; they are not the kernels that will be timed.
