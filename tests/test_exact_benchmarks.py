from __future__ import annotations

import numpy as np
import pytest

from pfme.exact_benchmarks import (
    bernoulli_difference_probabilities,
    bernoulli_risk_table,
    inventory_moments,
    inventory_risk_table,
    normal_required_sample_size,
)


def test_inventory_optimum_and_advantage() -> None:
    moments = inventory_moments()
    assert moments["mean_optimal"] == pytest.approx(76.29881495097729)
    assert moments["mean_comparator"] == pytest.approx(75.12611113021585)
    assert moments["mean_difference"] == pytest.approx(1.17270382076144)
    assert moments["variance_reduction"] == pytest.approx(0.957582432777945)


def test_inventory_exact_planning_examples() -> None:
    table = inventory_risk_table(np.array([96, 191])).set_index("sample_size")
    assert table.loc[96, "exact_strict_reversal_probability"] == pytest.approx(0.05284359718905554)
    assert table.loc[191, "exact_strict_reversal_probability"] == pytest.approx(0.010530414954417843)


def test_inventory_required_sample_sizes() -> None:
    moments = inventory_moments()
    assert normal_required_sample_size(moments["mean_difference"], moments["paired_variance"], 0.05) == 96
    assert normal_required_sample_size(moments["mean_difference"], moments["paired_variance"], 0.01) == 191


def test_bernoulli_difference_distribution() -> None:
    probabilities = bernoulli_difference_probabilities()
    assert probabilities == pytest.approx({"minus": 0.19, "zero": 0.57, "plus": 0.24})
    assert probabilities["plus"] - probabilities["minus"] == pytest.approx(0.05)


def test_bernoulli_strict_tie_and_nonpositive_are_separate() -> None:
    row = bernoulli_risk_table(np.array([100]), replications=20_000).iloc[0]
    assert row["paired_exact_strict_reversal_probability"] == pytest.approx(0.19997152041769642)
    assert row["paired_exact_tie_probability"] == pytest.approx(0.0454774808828367)
    assert row["paired_exact_nonpositive_probability"] == pytest.approx(0.24544900130053313)
    assert row["paired_exact_nonpositive_probability"] == pytest.approx(
        row["paired_exact_strict_reversal_probability"] + row["paired_exact_tie_probability"]
    )

