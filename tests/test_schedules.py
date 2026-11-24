"""CPU checks that the schedules reproduce dense causal attention.

These tests do not establish GPU or NCCL correctness.
"""

from __future__ import annotations

import torch

from spbench.algorithms.distflash_schedule import distflash_attention_global
from spbench.algorithms.ring_schedule import ring_attention_global
from spbench.algorithms.ulysses_schedule import ulysses_attention_global
from spbench.algorithms.usp_schedule import usp_attention_global
from spbench.attention_math import reference_attention
from spbench.errors import InvalidConfiguration
from spbench.work_model import full_causal_pairs


def _tensors(sequence, heads=4, dim=8, seed=42):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(1, sequence, heads, dim, generator=generator)
    key = torch.randn(1, sequence, heads, dim, generator=generator)
    value = torch.randn(1, sequence, heads, dim, generator=generator)
    incoming = torch.randn(1, sequence, heads, dim, generator=generator)
    return query, key, value, incoming


def _assert_close(actual, expected, label):
    delta = (actual - expected).abs()
    rel = float(delta.norm() / expected.norm().clamp_min(1e-12))
    max_abs = float(delta.max())
    assert rel < 1e-4 and max_abs < 1e-4, f"{label}: max_abs={max_abs:.3e} rel_l2={rel:.3e}"


def _grads(query, key, value, incoming, function):
    tensors = [item.detach().clone().requires_grad_(True) for item in (query, key, value)]
    output = function(*tensors)
    (output * incoming).sum().backward()
    return output.detach(), [item.grad.detach() for item in tensors]


def _assert_grads(query, key, value, incoming, function, label):
    reference_out, reference_grads = _grads(query, key, value, incoming, lambda q, k, v: reference_attention(q, k, v, True))
    actual_out, actual_grads = _grads(query, key, value, incoming, function)
    _assert_close(actual_out, reference_out, label + " output")
    for name, actual, expected in zip(("dQ", "dK", "dV"), actual_grads, reference_grads, strict=True):
        _assert_close(actual, expected, f"{label} {name}")


def test_reference_is_causal_and_future_tokens_do_not_leak():
    query, key, value, _ = _tensors(16, heads=2)
    output = reference_attention(query, key, value, True)
    position = 6
    key2 = key.clone()
    value2 = value.clone()
    key2[:, position:] += 0.7
    value2[:, position:] += 0.7
    output2 = reference_attention(query, key2, value2, True)
    _assert_close(output[:, :position], output2[:, :position], "future key/value perturbation")


def test_ring_basic_and_zigzag_match_reference_and_gradients():
    query, key, value, incoming = _tensors(16, heads=2, dim=8)
    for variant in ("basic", "zigzag"):
        result = ring_attention_global(query, key, value, world_size=2, variant=variant, causal=True)
        expected_pairs = full_causal_pairs(16) * 2
        assert result.total_executed_pairs() == expected_pairs
        _assert_grads(
            query,
            key,
            value,
            incoming,
            lambda q, k, v, variant=variant: ring_attention_global(q, k, v, 2, variant, True).output,
            f"ring {variant}",
        )


def test_ring_basic_noncausal_matches_and_zigzag_noncausal_is_rejected():
    query, key, value, _ = _tensors(12, heads=2)
    result = ring_attention_global(query, key, value, world_size=2, variant="basic", causal=False)
    _assert_close(result.output, reference_attention(query, key, value, False), "ring noncausal")
    try:
        ring_attention_global(query, key, value, world_size=2, variant="zigzag", causal=False)
    except InvalidConfiguration:
        return
    raise AssertionError("zigzag noncausal should be unavailable")


def test_ulysses_matches_reference_and_balances_heads():
    query, key, value, incoming = _tensors(16, heads=4, dim=8)
    result = ulysses_attention_global(query, key, value, world_size=4, causal=True)
    assert result.total_executed_pairs() == full_causal_pairs(16) * 4
    assert len(set(result.executed_pairs_per_head)) == 1
    _assert_grads(
        query,
        key,
        value,
        incoming,
        lambda q, k, v: ulysses_attention_global(q, k, v, 4, True).output,
        "ulysses",
    )


def test_distflash_even_worlds_match_and_odd_world_is_rejected():
    query, key, value, incoming = _tensors(16, heads=2, dim=8)
    for world in (2, 4):
        result = distflash_attention_global(query, key, value, world_size=world, causal=True)
        assert result.total_executed_pairs() == full_causal_pairs(16) * 2
        _assert_grads(
            query,
            key,
            value,
            incoming,
            lambda q, k, v, world=world: distflash_attention_global(q, k, v, world, True).output,
            f"distflash P={world}",
        )
    try:
        distflash_attention_global(query, key, value, world_size=3, causal=True)
    except InvalidConfiguration:
        return
    raise AssertionError("odd DistFlashAttn world size should be rejected")


def test_usp_layouts_that_preserve_causal_order_match_reference():
    query, key, value, incoming = _tensors(32, heads=4, dim=8)
    configs = [
        (2, 2, "basic", "ulysses_low"),
        (2, 2, "zigzag", "ulysses_low"),
        (2, 2, "zigzag", "ring_low"),
        (4, 1, "basic", "ulysses_low"),
        (1, 4, "zigzag", "ulysses_low"),
    ]
    for ulysses_degree, ring_degree, variant, placement in configs:
        result = usp_attention_global(
            query, key, value, ulysses_degree, ring_degree, variant, placement, causal=True
        )
        assert result.total_executed_pairs() == full_causal_pairs(32) * 4, (ulysses_degree, ring_degree, variant, placement)
        _assert_grads(
            query,
            key,
            value,
            incoming,
            lambda q, k, v, ud=ulysses_degree, rd=ring_degree, variant=variant, placement=placement: usp_attention_global(
                q, k, v, ud, rd, variant, placement, True
            ).output,
            f"usp ud={ulysses_degree} rd={ring_degree} {variant} {placement}",
        )


def test_usp_ring_low_basic_does_not_match_dense_causal_attention():
    query, key, value, _ = _tensors(16, heads=4, dim=8)
    result = usp_attention_global(query, key, value, 2, 2, "basic", "ring_low", causal=True)
    reference = reference_attention(query, key, value, True)
    assert not torch.allclose(result.output, reference, rtol=1e-4, atol=1e-4)
