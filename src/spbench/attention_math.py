"""Exact attention arithmetic used by the CPU reference and schedule checks."""

from __future__ import annotations

import math

import torch

from spbench.errors import InvalidConfiguration


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Dense scaled dot-product attention.

    q, k, v are BSHD: [batch, sequence, heads, head_dim]. A query at position i
    attends to keys 0..i inclusive when causal is true. Scale is 1/sqrt(head_dim).
    """

    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise InvalidConfiguration("reference attention expects matching BSHD Q, K, and V")
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if causal:
        length = q.shape[1]
        # Key index j is forbidden when j > i.
        future = torch.triu(torch.ones(length, length, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, v)


def block_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One attention block plus the online-softmax statistics (m, l).

    The returned output is normalized. m is the row max and l is the sum of
    exp(score - m), both shaped [batch, heads, queries].
    """

    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise InvalidConfiguration("block attention expects BSHD tensors")
    if k.shape[0] != q.shape[0] or k.shape[2] != q.shape[2] or k.shape[3] != q.shape[3]:
        raise InvalidConfiguration("block attention batch, heads, and head_dim must match")
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if causal:
        if q.shape[1] != k.shape[1]:
            raise InvalidConfiguration("a causal block requires equal query and key lengths")
        length = q.shape[1]
        future = torch.triu(torch.ones(length, length, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
    row_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - row_max)
    row_sum = exp_scores.sum(dim=-1, keepdim=True)
    output = torch.einsum("bhqk,bkhd->bqhd", exp_scores / row_sum, v)
    return output, row_max.squeeze(-1), row_sum.squeeze(-1)


def merge_statistics(
    left: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge two normalized blocks with the FlashAttention online-softmax update."""

    output_left, max_left, sum_left = left
    output_right, max_right, sum_right = right
    merged_max = torch.maximum(max_left, max_right)
    weight_left = torch.exp(max_left - merged_max) * sum_left
    weight_right = torch.exp(max_right - merged_max) * sum_right
    merged_sum = weight_left + weight_right
    # Statistics are [batch, heads, queries]. Outputs are BSHD.
    scale_left = (weight_left / merged_sum).permute(0, 2, 1).unsqueeze(-1)
    scale_right = (weight_right / merged_sum).permute(0, 2, 1).unsqueeze(-1)
    return scale_left * output_left + scale_right * output_right, merged_max, merged_sum


def merge_query_slice(
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    block: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    start: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge a block that covers only queries [start, start + block_length)."""

    output, row_max, row_sum = state
    block_output, block_max, block_sum = block
    length = block_output.shape[1]
    merged_output, merged_max, merged_sum = merge_statistics(
        (output[:, start : start + length], row_max[:, :, start : start + length], row_sum[:, :, start : start + length]),
        (block_output, block_max, block_sum),
    )
    output_parts = []
    max_parts = []
    sum_parts = []
    if start:
        output_parts.append(output[:, :start])
        max_parts.append(row_max[:, :, :start])
        sum_parts.append(row_sum[:, :, :start])
    output_parts.append(merged_output)
    max_parts.append(merged_max)
    sum_parts.append(merged_sum)
    end = start + length
    if end < output.shape[1]:
        output_parts.append(output[:, end:])
        max_parts.append(row_max[:, :, end:])
        sum_parts.append(row_sum[:, :, end:])
    return torch.cat(output_parts, dim=1), torch.cat(max_parts, dim=2), torch.cat(sum_parts, dim=2)


def geometric_pairs(query_len: int, key_len: int, causal: bool) -> int:
    """Valid query-key pairs in one block, for a single head and batch element."""

    if causal:
        if query_len != key_len:
            raise InvalidConfiguration("causal pair count requires a square block")
        return query_len * (query_len + 1) // 2
    return query_len * key_len
