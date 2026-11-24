from spbench.errors import InvalidConfiguration
from spbench.layouts import (
    basic_usp_preserves_causal_ring_order,
    build_group_maps,
    positions_are_bijective,
    usp_positions,
    zigzag_ring_positions,
)
from spbench.work_model import (
    contiguous_rank_pairs,
    contiguous_valid_pairs,
    distflash_paper_idle_fraction,
    full_causal_pairs,
    imbalance_ratios,
    lightseq_idle_slot_fraction,
    query_owned_pairs,
    ring_schedule_idle_fraction,
)


def test_contiguous_formula_matches_the_triangle():
    sequence, world = 128, 4
    pairs = contiguous_rank_pairs(sequence, world)
    assert pairs[0] == contiguous_valid_pairs(sequence, world, 0)
    assert sum(pairs) == full_causal_pairs(sequence)
    assert pairs == [query_owned_pairs(sequence, world, rank, "contiguous") for rank in range(world)]


def test_small_contiguous_example():
    # S=4, P=2. Rank 0 owns positions 0,1 (1+2 pairs). Rank 1 owns 2,3 (3+4).
    assert contiguous_valid_pairs(4, 2, 0) == 3
    assert contiguous_valid_pairs(4, 2, 1) == 7


def test_zigzag_pair_counts_are_exactly_equal_across_ranks():
    sequence, world = 1024, 8
    chunk = sequence // (2 * world)
    pairs = [query_owned_pairs(sequence, world, rank, "zigzag") for rank in range(world)]
    assert pairs == [chunk * (sequence + 1)] * world
    assert sum(pairs) == full_causal_pairs(sequence)


def test_zigzag_is_bijective_and_more_balanced_than_contiguous():
    sequence, world = 64, 4
    per_rank = [zigzag_ring_positions(sequence, world, rank) for rank in range(world)]
    assert positions_are_bijective(per_rank, sequence)
    zigzag = [query_owned_pairs(sequence, world, rank, "zigzag") for rank in range(world)]
    contiguous = contiguous_rank_pairs(sequence, world)
    assert sum(zigzag) == full_causal_pairs(sequence)
    assert imbalance_ratios(zigzag)["max_over_mean"] < imbalance_ratios(contiguous)["max_over_mean"]


def test_usp_group_maps_and_token_coverage():
    maps = build_group_maps(8, 2, 4, "ulysses_low")
    assert maps.ulysses_groups[0] == (0, 1)
    assert maps.ring_groups[0] == (0, 2, 4, 6)
    alternate = build_group_maps(8, 2, 4, "ring_low")
    assert maps.ulysses_groups != alternate.ulysses_groups
    for placement in ("ulysses_low", "ring_low"):
        for variant in ("basic", "zigzag"):
            owned = [usp_positions(64, rank, 2, 4, placement, variant) for rank in range(8)]
            assert positions_are_bijective(owned, 64)


def test_only_ulysses_low_basic_preserves_causal_ring_order_at_world_4():
    assert basic_usp_preserves_causal_ring_order(2, 2, "ulysses_low")
    assert not basic_usp_preserves_causal_ring_order(2, 2, "ring_low")


def test_idle_fractions_are_labeled_separately():
    assert abs(ring_schedule_idle_fraction(8) - (64 - 8) / (2 * 64)) < 1e-12
    assert distflash_paper_idle_fraction(8) == 1 / 16
    assert distflash_paper_idle_fraction(7) == 0
    # The code predicate's slot fraction is not equation (2).
    assert abs(lightseq_idle_slot_fraction(8) - (1 / 16)) > 1e-6


def test_bad_head_split_is_rejected_by_the_work_model():
    try:
        from spbench.work_model import ulysses_heads_on_rank

        ulysses_heads_on_rank(6, 4)
    except InvalidConfiguration:
        return
    raise AssertionError("6 heads cannot be split across 4 Ulysses ranks")
