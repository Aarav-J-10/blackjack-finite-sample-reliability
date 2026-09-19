"""Analytical inventory and Bernoulli benchmarks for PFME.

The functions in this module distinguish a strict reversal (sample mean
difference < 0) from a tie.  This matches the formal definition in the paper.
For discrete models we also report the nonpositive probability so published
tables that used ``<= 0`` can be audited explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class InventoryParameters:
    demand_mean: float = 20.0
    sale_price: float = 12.0
    purchase_cost: float = 7.0
    salvage_value: float = 2.0
    shortage_penalty: float = 4.0
    optimal_order: int = 21
    comparator_order: int = 20


@dataclass(frozen=True)
class BernoulliParameters:
    environment_probability: float = 0.5
    a_easy: float = 0.80
    a_hard: float = 0.40
    b_easy: float = 0.70
    b_hard: float = 0.40


def inventory_profit(order: int, demand: np.ndarray, p: InventoryParameters) -> np.ndarray:
    """Return one-period profit for an array of integer demand values."""
    sold = np.minimum(order, demand)
    leftover = np.maximum(order - demand, 0)
    shortage = np.maximum(demand - order, 0)
    return (
        p.sale_price * sold
        - p.purchase_cost * order
        + p.salvage_value * leftover
        - p.shortage_penalty * shortage
    )


def inventory_moments(p: InventoryParameters = InventoryParameters()) -> dict[str, float]:
    """Compute effectively exact moments using a negligible Poisson-tail truncation."""
    demand = np.arange(0, 160)
    probabilities = stats.poisson.pmf(demand, p.demand_mean)
    probabilities[-1] += 1.0 - probabilities.sum()
    rule_a = inventory_profit(p.optimal_order, demand, p)
    rule_b = inventory_profit(p.comparator_order, demand, p)
    mean_a = float(probabilities @ rule_a)
    mean_b = float(probabilities @ rule_b)
    var_a = float(probabilities @ (rule_a - mean_a) ** 2)
    var_b = float(probabilities @ (rule_b - mean_b) ** 2)
    covariance = float(probabilities @ ((rule_a - mean_a) * (rule_b - mean_b)))
    difference = rule_a - rule_b
    mean_difference = mean_a - mean_b
    paired_variance = float(probabilities @ (difference - mean_difference) ** 2)
    unpaired_variance = var_a + var_b
    return {
        "mean_optimal": mean_a,
        "mean_comparator": mean_b,
        "mean_difference": mean_difference,
        "variance_optimal": var_a,
        "variance_comparator": var_b,
        "covariance": covariance,
        "paired_variance": paired_variance,
        "unpaired_variance": unpaired_variance,
        "variance_reduction": 1.0 - paired_variance / unpaired_variance,
        "high_demand_probability": float(stats.poisson.sf(20, p.demand_mean)),
    }


def inventory_risk_table(
    sample_sizes: np.ndarray,
    p: InventoryParameters = InventoryParameters(),
) -> pd.DataFrame:
    """Return exact and normal paired reversal probabilities for inventory."""
    moments = inventory_moments(p)
    mu = moments["mean_difference"]
    sigma = math.sqrt(moments["paired_variance"])
    high_probability = moments["high_demand_probability"]
    rows: list[dict[str, float | int]] = []
    for n_value in sample_sizes:
        n = int(n_value)
        # Sum difference = 14 K - 5 n.  Strict reversal requires 14 K < 5 n.
        strict_threshold = math.ceil(5 * n / 14) - 1
        nonpositive_threshold = math.floor(5 * n / 14)
        strict = float(stats.binom.cdf(strict_threshold, n, high_probability))
        nonpositive = float(stats.binom.cdf(nonpositive_threshold, n, high_probability))
        rows.append(
            {
                "sample_size": n,
                "exact_strict_reversal_probability": strict,
                "exact_tie_probability": nonpositive - strict,
                "exact_nonpositive_probability": nonpositive,
                "normal_reversal_probability": float(stats.norm.cdf(-mu * math.sqrt(n) / sigma)),
            }
        )
    return pd.DataFrame(rows)


def bernoulli_difference_probabilities(
    p: BernoulliParameters = BernoulliParameters(),
) -> dict[str, float]:
    """Return the exact paired difference probabilities."""
    w = p.environment_probability
    plus = w * p.a_easy * (1.0 - p.b_easy) + (1.0 - w) * p.a_hard * (1.0 - p.b_hard)
    minus = w * (1.0 - p.a_easy) * p.b_easy + (1.0 - w) * (1.0 - p.a_hard) * p.b_hard
    zero = 1.0 - plus - minus
    return {"minus": minus, "zero": zero, "plus": plus}


def _paired_exact_coefficients(n: int, probabilities: dict[str, float]) -> np.ndarray:
    """Return coefficients for sums from -n through +n."""
    coefficients = np.array([1.0])
    kernel = np.array([probabilities["minus"], probabilities["zero"], probabilities["plus"]])
    for _ in range(n):
        coefficients = np.convolve(coefficients, kernel)
    return coefficients


def bernoulli_risk_table(
    sample_sizes: np.ndarray,
    replications: int = 100_000,
    seed: int = 20_260_919,
    p: BernoulliParameters = BernoulliParameters(),
) -> pd.DataFrame:
    """Compare exact, Monte Carlo, and normal Bernoulli reversal risks."""
    probabilities = bernoulli_difference_probabilities(p)
    mean_a = p.environment_probability * p.a_easy + (1.0 - p.environment_probability) * p.a_hard
    mean_b = p.environment_probability * p.b_easy + (1.0 - p.environment_probability) * p.b_hard
    mu = mean_a - mean_b
    variance = probabilities["plus"] + probabilities["minus"] - mu**2
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    for n_value in sample_sizes:
        n = int(n_value)
        coefficients = _paired_exact_coefficients(n, probabilities)
        exact_strict = float(coefficients[:n].sum())
        exact_tie = float(coefficients[n])

        counts = rng.multinomial(
            n,
            [probabilities["minus"], probabilities["zero"], probabilities["plus"]],
            size=replications,
        )
        sums = counts[:, 2] - counts[:, 0]

        y = np.arange(n + 1)
        y_probability = stats.binom.pmf(y, n, mean_b)
        unpaired_strict = float(np.sum(y_probability * stats.binom.cdf(y - 1, n, mean_a)))
        unpaired_nonpositive = float(np.sum(y_probability * stats.binom.cdf(y, n, mean_a)))

        rows.append(
            {
                "sample_size": n,
                "paired_exact_strict_reversal_probability": exact_strict,
                "paired_exact_tie_probability": exact_tie,
                "paired_exact_nonpositive_probability": exact_strict + exact_tie,
                "paired_monte_carlo_strict_reversal_probability": float(np.mean(sums < 0)),
                "paired_monte_carlo_tie_probability": float(np.mean(sums == 0)),
                "paired_normal_reversal_probability": float(stats.norm.cdf(-mu * math.sqrt(n) / math.sqrt(variance))),
                "unpaired_exact_strict_reversal_probability": unpaired_strict,
                "unpaired_exact_tie_probability": unpaired_nonpositive - unpaired_strict,
                "unpaired_exact_nonpositive_probability": unpaired_nonpositive,
                "replications": replications,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def normal_required_sample_size(mean: float, variance: float, target_risk: float) -> int:
    """Return the one-sided normal-approximation planning sample size."""
    z_value = stats.norm.ppf(1.0 - target_risk)
    return math.ceil(z_value**2 * variance / mean**2)

