from __future__ import annotations

import numpy as np

from scripts import run_pfme_analysis as analysis


def test_base_deck_has_52_cards() -> None:
    values, counts = np.unique(analysis.BASE_DECK_VALUES, return_counts=True)
    count_map = dict(zip(values.tolist(), counts.tolist()))
    assert len(analysis.BASE_DECK_VALUES) == 52
    assert count_map == {2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 16, 11: 4}


def test_soft_hand_total() -> None:
    cards = np.zeros(24, dtype=np.int8)
    cards[:2] = [11, 6]
    total, soft = analysis.hand_total(cards, 2)
    assert total == 17
    assert soft

    cards[:3] = [11, 6, 10]
    total, soft = analysis.hand_total(cards, 3)
    assert total == 17
    assert not soft


def test_inventory_distribution_uses_q21_against_q20() -> None:
    _, _, probabilities, metadata = analysis.inventory_distribution()
    assert probabilities.sum() == 1.0
    assert metadata["optimized_order"] == 21
    assert metadata["naive_order"] == 20


def test_paired_simulation_is_seed_reproducible() -> None:
    sizes = np.array([25], dtype=np.int64)
    first_a, first_b = analysis.simulate_paired_blackjack(
        8, sizes, 123456, 0, 1, *analysis.config_args(analysis.BASE_CONFIG)
    )
    second_a, second_b = analysis.simulate_paired_blackjack(
        8, sizes, 123456, 0, 1, *analysis.config_args(analysis.BASE_CONFIG)
    )
    assert np.array_equal(first_a, second_a)
    assert np.array_equal(first_b, second_b)

