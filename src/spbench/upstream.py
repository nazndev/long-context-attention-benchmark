"""Commits and licenses inspected on 2026-11-24."""

from __future__ import annotations

UPSTREAM: dict[str, dict[str, str | None]] = {
    "ring_attention_jax": {
        "url": "https://github.com/haoliuhl/ringattention",
        "commit": "d2ea1af9a288f85ea2fd74690ece1c16d2eebc83",
        "date": "2025-10-13",
        "license": "Apache-2.0",
        "version": None,
        "note": "The paper URL github.com/lhao499/llm_large_context resolved to this same commit.",
    },
    "ring_flash_attention": {
        "url": "https://github.com/zhuzilin/ring-flash-attention",
        "commit": "786677930bce4f6022166899c88ce2c00c814ee2",
        "date": "2025-09-10",
        "license": "MIT",
        "version": "0.1.8",
        "note": "Backward casts dq to bfloat16 inside ring_flash_attn_backward.",
    },
    "lightseq": {
        "url": "https://github.com/RulinShao/LightSeq",
        "commit": "8f486dad3d0670057dfbe3b30c003080b61c5325",
        "date": "2024-08-19",
        "license": None,
        "version": None,
        "note": "GitHub license API returned 404 and the tree has no LICENSE file. The paper text is CC-BY-4.0. Code was not vendored.",
    },
    "deepspeed": {
        "url": "https://github.com/deepspeedai/DeepSpeed",
        "commit": "140a2fbcd253d600e094ff7e659064180928246b",
        "date": "2026-11-24",
        "license": "Apache-2.0",
        "version": None,
        "note": "DistributedAttention default gather_idx is 0. BSHD calls must pass gather_idx=1 and batch_dim_idx=0.",
    },
    "yunchang": {
        "url": "https://github.com/feifeibear/long-context-attention",
        "commit": "a5f8f600d435ced180a97c91c68e87ff3cfb8f89",
        "date": "2026-09-16",
        "license": "Apache-2.0",
        "version": "0.6.4",
        "note": "Requires torch>=2.3. Optional flash-attn>=2.6. README says TORCH_EFFICIENT has no backward.",
    },
    "flash_attention": {
        "url": "https://github.com/Dao-AILab/flash-attention",
        "commit": "b038b4974f26ba3bc8ab3c8b840c60b0432ce39b",
        "date": "2026-11-23",
        "license": "BSD-3-Clause",
        "version": None,
        "note": "HEAD is an FA4-era commit. ring-flash-attention 0.1.8 and yunchang 0.6.4 document flash-attn 2.x private APIs. Joint compatibility is unresolved.",
    },
}


def upstream_commit_map() -> dict[str, str | None]:
    return {name: str(meta["commit"]) if meta["commit"] else None for name, meta in UPSTREAM.items()}
