"""
BLACKJACK STRATEGY COMPARISON: FULL REPLICATION STUDY

Design
------
- 1,000 independent paired replications
- 10,000 original rounds per strategy in every replication
- Sample sizes extracted from each replication:
  n = 100, 500, 1,000, 2,500, 5,000, 10,000
- Total simulated original rounds:
  1,000 x 10,000 x 2 = 20,000,000

The smaller n conditions are prefixes of each 10,000-round run. This avoids
simulating the same experimental condition repeatedly and reduces the total
from 38.2 million to 20 million original rounds.

Outputs
-------
- Replication-level return dataset
- Paired-difference dataset
- Descriptive statistics
- Paired t-tests, Wilcoxon sensitivity tests, and Cohen's dz
- Monte Carlo standard errors
- Confidence-interval convergence results
- Five publication-ready figures
- Excel workbook, CSV files, and ZIP archive

The script stores only replication-level results by default. Set
SAVE_MINIMAL_ROUND_LEVEL = True to additionally save 20 million minimal
round-level rows (replication, strategy, round number, net return) as two
compressed CSV files. The detailed card/action history is deliberately not
stored because it is not required for the planned inference and would create
an unnecessarily enormous dataset. Humans already invented enough storage
problems without encouragement.
"""

from __future__ import annotations

import csv
import gzip
import math
import os
import random
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bar is optional
    tqdm = None


# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

MASTER_SEED = 20260622
REPLICATIONS = 1_000
SAMPLE_SIZES = (100, 500, 1_000, 2_500, 5_000, 10_000)
MAX_ROUNDS_PER_STRATEGY = max(SAMPLE_SIZES)

# Use all but one logical CPU by default. Set to 1 for fully serial execution.
CPU_COUNT = os.cpu_count() or 1
N_WORKERS = min(8, max(1, CPU_COUNT - 1))

# Saving 20 million minimal round rows is optional and disabled by default.
SAVE_MINIMAL_ROUND_LEVEL = False

NUM_DECKS = 6
PENETRATION = 0.75
BLACKJACK_PAYOUT = 1.5
MAX_SPLITS = 3
ALLOW_DOUBLE_AFTER_SPLIT = True
RESPLIT_ACES = False

OUTPUT_FOLDER = Path("blackjack_replication_study")


# ============================================================
# 2. CARD AND SHOE DEFINITIONS
# ============================================================

RANKS = (
    "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "J", "Q", "K", "A"
)
VALUES = (2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11)
BASE_DECK = tuple(zip(RANKS, VALUES))


class Shoe:
    """Finite multi-deck blackjack shoe with cut-card reshuffling."""

    def __init__(
        self,
        num_decks: int = NUM_DECKS,
        penetration: float = PENETRATION,
        rng: random.Random | None = None,
    ) -> None:
        self.num_decks = num_decks
        self.penetration = penetration
        self.rng = rng or random.Random()
        self.initial_size = 52 * num_decks
        self.cut_remaining = int(self.initial_size * (1.0 - penetration))
        self.cards: list[tuple[str, int]] = []
        self.reshuffle()

    def reshuffle(self) -> None:
        self.cards = list(BASE_DECK) * self.num_decks
        self.rng.shuffle(self.cards)

    def needs_shuffle(self) -> bool:
        return len(self.cards) <= self.cut_remaining

    def deal(self) -> tuple[str, int]:
        if not self.cards:
            self.reshuffle()
        return self.cards.pop()


# ============================================================
# 3. HAND UTILITIES
# ============================================================


def hand_value(cards: list[tuple[str, int]]) -> tuple[int, bool]:
    """Return the best legal total and whether the hand is soft."""
    total = sum(value for _, value in cards)
    aces_as_eleven = sum(1 for _, value in cards if value == 11)

    while total > 21 and aces_as_eleven > 0:
        total -= 10
        aces_as_eleven -= 1

    return total, aces_as_eleven > 0


def is_blackjack(cards: list[tuple[str, int]]) -> bool:
    total, _ = hand_value(cards)
    return len(cards) == 2 and total == 21


# ============================================================
# 4. STRATEGIES
# ============================================================


def basic_strategy(
    cards: list[tuple[str, int]],
    dealer_upcard: tuple[str, int],
    can_double: bool,
    can_split: bool,
) -> str:
    """Six-deck S17, DAS, no-surrender basic strategy."""
    total, soft = hand_value(cards)
    dealer = dealer_upcard[1]

    # Pair splitting
    if can_split and len(cards) == 2 and cards[0][1] == cards[1][1]:
        pair_value = cards[0][1]

        if pair_value == 11:  # A,A
            return "split"
        if pair_value == 10:  # 10-value pairs
            return "stand"
        if pair_value == 9:
            return "split" if dealer in (2, 3, 4, 5, 6, 8, 9) else "stand"
        if pair_value == 8:
            return "split"
        if pair_value == 7:
            return "split" if dealer in (2, 3, 4, 5, 6, 7) else "hit"
        if pair_value == 6:
            return "split" if dealer in (2, 3, 4, 5, 6) else "hit"
        if pair_value == 4:
            return "split" if dealer in (5, 6) else "hit"
        if pair_value in (2, 3):
            return "split" if dealer in (2, 3, 4, 5, 6, 7) else "hit"
        # 5,5 falls through and is played as hard 10.

    # Soft totals
    if soft:
        if total in (13, 14):
            return "double" if can_double and dealer in (5, 6) else "hit"
        if total in (15, 16):
            return "double" if can_double and dealer in (4, 5, 6) else "hit"
        if total == 17:
            return "double" if can_double and dealer in (3, 4, 5, 6) else "hit"
        if total == 18:
            if can_double and dealer in (3, 4, 5, 6):
                return "double"
            return "stand" if dealer in (2, 7, 8) else "hit"
        if total == 19:
            return "double" if can_double and dealer == 6 else "stand"
        if total >= 20:
            return "stand"
        return "hit"

    # Hard totals
    if total <= 8:
        return "hit"
    if total == 9:
        return "double" if can_double and dealer in (3, 4, 5, 6) else "hit"
    if total == 10:
        return "double" if can_double and dealer in range(2, 10) else "hit"
    if total == 11:
        return "double" if can_double and dealer in range(2, 11) else "hit"
    if total == 12:
        return "stand" if dealer in (4, 5, 6) else "hit"
    if 13 <= total <= 16:
        return "stand" if dealer in (2, 3, 4, 5, 6) else "hit"
    return "stand"


def simplified_strategy(
    cards: list[tuple[str, int]],
    dealer_upcard: tuple[str, int],
    can_double: bool,
    can_split: bool,
) -> str:
    """Inexperienced-player strategy used as the comparison condition."""
    del dealer_upcard, can_double, can_split
    total, soft = hand_value(cards)
    if soft:
        return "stand" if total >= 18 else "hit"
    return "stand" if total >= 12 else "hit"


STRATEGY_FUNCTIONS: dict[str, Callable[..., str]] = {
    "Basic": basic_strategy,
    "Simplified": simplified_strategy,
}


# ============================================================
# 5. LIGHTWEIGHT ROUND SIMULATOR
# ============================================================


@dataclass
class PlayerHand:
    cards: list[tuple[str, int]]
    bet: float = 1.0
    from_split: bool = False
    split_aces: bool = False


def dealer_play(
    dealer_cards: list[tuple[str, int]],
    shoe: Shoe,
) -> list[tuple[str, int]]:
    """Dealer stands on all 17s, including soft 17."""
    while True:
        total, _ = hand_value(dealer_cards)
        if total >= 17:
            return dealer_cards
        dealer_cards.append(shoe.deal())


def play_round_return(strategy_name: str, shoe: Shoe) -> float:
    """Simulate one original round and return net units won or lost."""
    if shoe.needs_shuffle():
        shoe.reshuffle()

    # Standard alternating deal: player, dealer, player, dealer.
    player_card_1 = shoe.deal()
    dealer_card_1 = shoe.deal()
    player_card_2 = shoe.deal()
    dealer_card_2 = shoe.deal()

    player_initial = [player_card_1, player_card_2]
    dealer_cards = [dealer_card_1, dealer_card_2]
    dealer_upcard = dealer_card_1

    player_blackjack = is_blackjack(player_initial)
    dealer_blackjack = is_blackjack(dealer_cards)

    if player_blackjack or dealer_blackjack:
        if player_blackjack and dealer_blackjack:
            return 0.0
        if player_blackjack:
            return BLACKJACK_PAYOUT
        return -1.0

    strategy_function = STRATEGY_FUNCTIONS[strategy_name]

    pending_hands: deque[PlayerHand] = deque(
        [PlayerHand(cards=player_initial.copy())]
    )
    completed_hands: list[PlayerHand] = []
    split_count = 0

    while pending_hands:
        player_hand = pending_hands.popleft()

        if player_hand.split_aces:
            completed_hands.append(player_hand)
            continue

        replaced_by_split = False

        while True:
            player_total, _ = hand_value(player_hand.cards)
            if player_total >= 21:
                break

            can_double = (
                len(player_hand.cards) == 2
                and (ALLOW_DOUBLE_AFTER_SPLIT or not player_hand.from_split)
            )

            can_split = (
                strategy_name == "Basic"
                and len(player_hand.cards) == 2
                and player_hand.cards[0][1] == player_hand.cards[1][1]
                and split_count < MAX_SPLITS
                and not (
                    player_hand.cards[0][1] == 11
                    and player_hand.from_split
                    and not RESPLIT_ACES
                )
            )

            action = strategy_function(
                player_hand.cards,
                dealer_upcard,
                can_double,
                can_split,
            )

            if action == "stand":
                break

            if action == "hit":
                player_hand.cards.append(shoe.deal())
                continue

            if action == "double" and can_double:
                player_hand.bet *= 2.0
                player_hand.cards.append(shoe.deal())
                break

            if action == "split" and can_split:
                split_count += 1
                first_card, second_card = player_hand.cards
                splitting_aces = first_card[1] == 11

                first_hand = PlayerHand(
                    cards=[first_card, shoe.deal()],
                    bet=player_hand.bet,
                    from_split=True,
                    split_aces=splitting_aces,
                )
                second_hand = PlayerHand(
                    cards=[second_card, shoe.deal()],
                    bet=player_hand.bet,
                    from_split=True,
                    split_aces=splitting_aces,
                )

                # Preserve natural left-to-right play order.
                pending_hands.appendleft(second_hand)
                pending_hands.appendleft(first_hand)
                replaced_by_split = True
                break

            # Defensive fallback for an impossible strategy response.
            player_hand.cards.append(shoe.deal())

        if not replaced_by_split:
            completed_hands.append(player_hand)

    at_least_one_live_hand = any(
        hand_value(hand.cards)[0] <= 21 for hand in completed_hands
    )

    if at_least_one_live_hand:
        dealer_play(dealer_cards, shoe)

    dealer_total, _ = hand_value(dealer_cards)
    dealer_bust = dealer_total > 21

    round_return = 0.0

    for player_hand in completed_hands:
        player_total, _ = hand_value(player_hand.cards)

        if player_total > 21:
            round_return -= player_hand.bet
        elif dealer_bust:
            round_return += player_hand.bet
        elif player_total > dealer_total:
            round_return += player_hand.bet
        elif player_total < dealer_total:
            round_return -= player_hand.bet
        # Equal totals are pushes and add zero.

    return round_return


# ============================================================
# 6. ONE PAIRED REPLICATION
# ============================================================


def simulate_strategy(
    strategy_name: str,
    seed: int,
    rounds: int,
) -> np.ndarray:
    """Return one strategy's round-level returns for one replication."""
    rng = random.Random(seed)
    shoe = Shoe(rng=rng)
    returns = np.empty(rounds, dtype=np.float32)

    for index in range(rounds):
        returns[index] = play_round_return(strategy_name, shoe)

    return returns


def replication_seed(replication: int) -> int:
    """Generate a deterministic, well-separated seed for a replication."""
    return MASTER_SEED + replication * 1_000_003


def run_one_replication(replication: int) -> dict:
    """
    Run both strategies with the same replication seed.

    Using the same seed creates a common-random-number pairing. The streams
    diverge after strategy-dependent card consumption, but each pair begins
    from the same shuffled shoe and reproducibility state.
    """
    seed = replication_seed(replication)

    basic_returns = simulate_strategy(
        "Basic", seed, MAX_ROUNDS_PER_STRATEGY
    )
    simplified_returns = simulate_strategy(
        "Simplified", seed, MAX_ROUNDS_PER_STRATEGY
    )

    basic_cumulative = np.cumsum(basic_returns, dtype=np.float64)
    simplified_cumulative = np.cumsum(simplified_returns, dtype=np.float64)

    summary_rows: list[dict] = []

    for n in SAMPLE_SIZES:
        basic_total = float(basic_cumulative[n - 1])
        simplified_total = float(simplified_cumulative[n - 1])

        summary_rows.extend(
            [
                {
                    "replication": replication,
                    "seed": seed,
                    "n": n,
                    "strategy": "Basic",
                    "total_return_units": basic_total,
                    "mean_return_per_round": basic_total / n,
                    "return_percentage": 100.0 * basic_total / n,
                },
                {
                    "replication": replication,
                    "seed": seed,
                    "n": n,
                    "strategy": "Simplified",
                    "total_return_units": simplified_total,
                    "mean_return_per_round": simplified_total / n,
                    "return_percentage": 100.0 * simplified_total / n,
                },
            ]
        )

    result = {"summary_rows": summary_rows}

    if SAVE_MINIMAL_ROUND_LEVEL:
        result["basic_returns"] = basic_returns
        result["simplified_returns"] = simplified_returns

    return result


# ============================================================
# 7. STATISTICAL ANALYSIS
# ============================================================


def descriptive_statistics(replication_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (n, strategy), group in replication_df.groupby(["n", "strategy"]):
        values = group["return_percentage"].to_numpy(dtype=float)
        count = values.size
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        se = sd / math.sqrt(count)
        t_critical = float(stats.t.ppf(0.975, df=count - 1))
        q1, q3 = np.percentile(values, [25, 75])

        rows.append(
            {
                "n": int(n),
                "strategy": strategy,
                "replications": count,
                "mean_return_pct": mean,
                "median_return_pct": float(np.median(values)),
                "standard_deviation_pct": sd,
                "interquartile_range_pct": float(q3 - q1),
                "minimum_return_pct": float(np.min(values)),
                "maximum_return_pct": float(np.max(values)),
                "monte_carlo_standard_error_pct": se,
                "ci_95_lower_pct": mean - t_critical * se,
                "ci_95_upper_pct": mean + t_critical * se,
                "ci_95_width_pct": 2.0 * t_critical * se,
            }
        )

    return pd.DataFrame(rows).sort_values(["n", "strategy"]).reset_index(drop=True)


def create_paired_dataset(replication_df: pd.DataFrame) -> pd.DataFrame:
    pivot = replication_df.pivot(
        index=["replication", "seed", "n"],
        columns="strategy",
        values="return_percentage",
    ).reset_index()

    pivot.columns.name = None
    pivot["paired_difference_pct"] = pivot["Basic"] - pivot["Simplified"]
    return pivot.sort_values(["n", "replication"]).reset_index(drop=True)


def paired_inference(paired_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for n, group in paired_df.groupby("n"):
        basic = group["Basic"].to_numpy(dtype=float)
        simplified = group["Simplified"].to_numpy(dtype=float)
        differences = group["paired_difference_pct"].to_numpy(dtype=float)

        count = differences.size
        mean_difference = float(np.mean(differences))
        sd_difference = float(np.std(differences, ddof=1))
        se_difference = sd_difference / math.sqrt(count)
        t_critical = float(stats.t.ppf(0.975, df=count - 1))
        ci_lower = mean_difference - t_critical * se_difference
        ci_upper = mean_difference + t_critical * se_difference

        # One-sided paired t-test: H1 Basic > Simplified.
        try:
            t_result = stats.ttest_rel(basic, simplified, alternative="greater")
            t_statistic = float(t_result.statistic)
            t_p_one_sided = float(t_result.pvalue)
        except TypeError:  # compatibility with older SciPy versions
            t_two_sided = stats.ttest_rel(basic, simplified)
            t_statistic = float(t_two_sided.statistic)
            t_p_one_sided = (
                float(t_two_sided.pvalue) / 2.0
                if t_statistic > 0
                else 1.0 - float(t_two_sided.pvalue) / 2.0
            )

        # Wilcoxon sensitivity analysis. Zero differences are omitted.
        try:
            wilcoxon_result = stats.wilcoxon(
                differences,
                alternative="greater",
                zero_method="wilcox",
                correction=False,
            )
            wilcoxon_statistic = float(wilcoxon_result.statistic)
            wilcoxon_p_one_sided = float(wilcoxon_result.pvalue)
        except ValueError:
            wilcoxon_statistic = float("nan")
            wilcoxon_p_one_sided = float("nan")

        cohen_dz = (
            mean_difference / sd_difference
            if sd_difference > 0
            else float("nan")
        )

        rows.append(
            {
                "n": int(n),
                "replications": count,
                "mean_basic_return_pct": float(np.mean(basic)),
                "mean_simplified_return_pct": float(np.mean(simplified)),
                "mean_paired_difference_pct": mean_difference,
                "median_paired_difference_pct": float(np.median(differences)),
                "sd_paired_difference_pct": sd_difference,
                "iqr_paired_difference_pct": float(
                    np.percentile(differences, 75)
                    - np.percentile(differences, 25)
                ),
                "minimum_paired_difference_pct": float(np.min(differences)),
                "maximum_paired_difference_pct": float(np.max(differences)),
                "skewness_paired_difference": float(
                    stats.skew(differences, bias=False)
                ),
                "mcse_mean_difference_pct": se_difference,
                "difference_ci_95_lower_pct": ci_lower,
                "difference_ci_95_upper_pct": ci_upper,
                "difference_ci_95_width_pct": ci_upper - ci_lower,
                "paired_t_statistic": t_statistic,
                "paired_t_p_one_sided": t_p_one_sided,
                "wilcoxon_statistic": wilcoxon_statistic,
                "wilcoxon_p_one_sided": wilcoxon_p_one_sided,
                "cohen_dz": cohen_dz,
            }
        )

    return pd.DataFrame(rows).sort_values("n").reset_index(drop=True)


# ============================================================
# 8. FIGURES
# ============================================================


def make_figures(
    replication_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    descriptive_df: pd.DataFrame,
    inference_df: pd.DataFrame,
) -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Figure 1: Replication-level return distributions
    labels = []
    data = []
    positions = []
    position = 1

    for n in SAMPLE_SIZES:
        for strategy in ("Basic", "Simplified"):
            values = replication_df.loc[
                (replication_df["n"] == n)
                & (replication_df["strategy"] == strategy),
                "return_percentage",
            ].to_numpy()
            data.append(values)
            labels.append(f"{n}\n{strategy[0]}")
            positions.append(position)
            position += 1
        position += 0.8

    plt.figure(figsize=(15, 7))
    plt.boxplot(data, positions=positions, widths=0.65, showfliers=False)
    plt.axhline(0, linewidth=1)
    plt.xticks(positions, labels)
    plt.xlabel("Sample size and strategy (B = Basic, S = Simplified)")
    plt.ylabel("Replication return (%)")
    plt.title("Replication-Level Return Distributions")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "01_replication_return_distributions.png", dpi=300)
    plt.close()

    # Figure 2: Paired-difference distributions
    difference_data = [
        paired_df.loc[paired_df["n"] == n, "paired_difference_pct"].to_numpy()
        for n in SAMPLE_SIZES
    ]

    plt.figure(figsize=(11, 7))
    plt.boxplot(difference_data, labels=[str(n) for n in SAMPLE_SIZES], showfliers=False)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Rounds per strategy in each replication")
    plt.ylabel("Basic minus Simplified return (percentage points)")
    plt.title("Distribution of Paired Return Differences")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "02_paired_difference_distributions.png", dpi=300)
    plt.close()

    # Figure 3: Sample-size sensitivity
    plt.figure(figsize=(10, 7))
    for strategy in ("Basic", "Simplified"):
        subset = descriptive_df[descriptive_df["strategy"] == strategy]
        means = subset["mean_return_pct"].to_numpy()
        lower = subset["ci_95_lower_pct"].to_numpy()
        upper = subset["ci_95_upper_pct"].to_numpy()
        yerr = np.vstack([means - lower, upper - means])

        plt.errorbar(
            subset["n"],
            means,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=strategy,
        )

    plt.axhline(0, linewidth=1)
    plt.xscale("log")
    plt.xlabel("Rounds per strategy in each replication (log scale)")
    plt.ylabel("Mean replication return (%)")
    plt.title("Sample-Size Sensitivity of Estimated Strategy Returns")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "03_sample_size_sensitivity.png", dpi=300)
    plt.close()

    # Figure 4: Monte Carlo standard errors
    plt.figure(figsize=(10, 7))
    for strategy in ("Basic", "Simplified"):
        subset = descriptive_df[descriptive_df["strategy"] == strategy]
        plt.plot(
            subset["n"],
            subset["monte_carlo_standard_error_pct"],
            marker="o",
            label=strategy,
        )

    plt.plot(
        inference_df["n"],
        inference_df["mcse_mean_difference_pct"],
        marker="o",
        label="Paired difference",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Rounds per strategy in each replication (log scale)")
    plt.ylabel("Monte Carlo standard error (percentage points, log scale)")
    plt.title("Monte Carlo Standard Error by Sample Size")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "04_monte_carlo_standard_errors.png", dpi=300)
    plt.close()

    # Figure 5: Confidence-interval convergence
    plt.figure(figsize=(10, 7))
    for strategy in ("Basic", "Simplified"):
        subset = descriptive_df[descriptive_df["strategy"] == strategy]
        plt.plot(
            subset["n"],
            subset["ci_95_width_pct"],
            marker="o",
            label=f"{strategy} mean",
        )

    plt.plot(
        inference_df["n"],
        inference_df["difference_ci_95_width_pct"],
        marker="o",
        label="Mean paired difference",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Rounds per strategy in each replication (log scale)")
    plt.ylabel("95% confidence-interval width (percentage points, log scale)")
    plt.title("Confidence-Interval Convergence Across Sample Sizes")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "05_confidence_interval_convergence.png", dpi=300)
    plt.close()


# ============================================================
# 9. OUTPUT HELPERS
# ============================================================


def initialise_round_level_files() -> tuple[Path, Path]:
    basic_path = OUTPUT_FOLDER / "basic_round_level_minimal.csv.gz"
    simplified_path = OUTPUT_FOLDER / "simplified_round_level_minimal.csv.gz"

    header = ["replication", "strategy", "round_in_replication", "net_return"]

    for path in (basic_path, simplified_path):
        with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(header)

    return basic_path, simplified_path


def append_round_level_rows(
    path: Path,
    replication: int,
    strategy: str,
    returns: np.ndarray,
) -> None:
    with gzip.open(path, "at", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(
            (replication, strategy, index, float(value))
            for index, value in enumerate(returns, start=1)
        )


def save_outputs(
    replication_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    descriptive_df: pd.DataFrame,
    inference_df: pd.DataFrame,
) -> Path:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    settings_df = pd.DataFrame(
        {
            "parameter": [
                "master_seed",
                "replications",
                "sample_sizes",
                "maximum_rounds_per_strategy_per_replication",
                "total_original_rounds",
                "number_of_workers",
                "number_of_decks",
                "penetration",
                "dealer_soft_17_rule",
                "blackjack_payout",
                "maximum_splits",
                "double_after_split",
                "resplit_aces",
                "surrender",
                "pairing_method",
                "saved_minimal_round_level_data",
            ],
            "value": [
                MASTER_SEED,
                REPLICATIONS,
                ", ".join(map(str, SAMPLE_SIZES)),
                MAX_ROUNDS_PER_STRATEGY,
                REPLICATIONS * MAX_ROUNDS_PER_STRATEGY * 2,
                N_WORKERS,
                NUM_DECKS,
                PENETRATION,
                "stand",
                BLACKJACK_PAYOUT,
                MAX_SPLITS,
                ALLOW_DOUBLE_AFTER_SPLIT,
                RESPLIT_ACES,
                False,
                "same seed for both strategies within each replication",
                SAVE_MINIMAL_ROUND_LEVEL,
            ],
        }
    )

    replication_df.to_csv(OUTPUT_FOLDER / "replication_level_returns.csv", index=False)
    paired_df.to_csv(OUTPUT_FOLDER / "paired_differences.csv", index=False)
    descriptive_df.to_csv(OUTPUT_FOLDER / "descriptive_statistics.csv", index=False)
    inference_df.to_csv(OUTPUT_FOLDER / "paired_inference.csv", index=False)
    settings_df.to_csv(OUTPUT_FOLDER / "experiment_settings.csv", index=False)

    workbook_path = OUTPUT_FOLDER / "blackjack_replication_results.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        replication_df.to_excel(writer, sheet_name="Replication Returns", index=False)
        paired_df.to_excel(writer, sheet_name="Paired Differences", index=False)
        descriptive_df.to_excel(writer, sheet_name="Descriptive Statistics", index=False)
        inference_df.to_excel(writer, sheet_name="Paired Inference", index=False)
        settings_df.to_excel(writer, sheet_name="Experiment Settings", index=False)

    archive_base = "blackjack_replication_study"
    shutil.make_archive(archive_base, "zip", OUTPUT_FOLDER)
    return Path(f"{archive_base}.zip")


# ============================================================
# 10. MAIN EXPERIMENT
# ============================================================


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("BLACKJACK REPLICATION STUDY")
    print("=" * 72)
    print(f"Replications: {REPLICATIONS:,}")
    print(f"Rounds per strategy per replication: {MAX_ROUNDS_PER_STRATEGY:,}")
    print(f"Total original rounds: {REPLICATIONS * MAX_ROUNDS_PER_STRATEGY * 2:,}")
    print(f"Sample sizes: {SAMPLE_SIZES}")
    print(f"Workers: {N_WORKERS}")
    print(f"Save minimal 20-million-row dataset: {SAVE_MINIMAL_ROUND_LEVEL}")

    basic_round_path = None
    simplified_round_path = None

    if SAVE_MINIMAL_ROUND_LEVEL:
        basic_round_path, simplified_round_path = initialise_round_level_files()

    all_summary_rows: list[dict] = []
    replication_numbers = range(1, REPLICATIONS + 1)

    if N_WORKERS == 1:
        result_iterator = map(run_one_replication, replication_numbers)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=N_WORKERS)
        result_iterator = executor.map(
            run_one_replication,
            replication_numbers,
            chunksize=1,
        )

    if tqdm is not None:
        result_iterator = tqdm(
            result_iterator,
            total=REPLICATIONS,
            desc="Running paired replications",
        )

    try:
        for replication, result in enumerate(result_iterator, start=1):
            all_summary_rows.extend(result["summary_rows"])

            if SAVE_MINIMAL_ROUND_LEVEL:
                append_round_level_rows(
                    basic_round_path,
                    replication,
                    "Basic",
                    result["basic_returns"],
                )
                append_round_level_rows(
                    simplified_round_path,
                    replication,
                    "Simplified",
                    result["simplified_returns"],
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    replication_df = pd.DataFrame(all_summary_rows)
    replication_df = replication_df.sort_values(
        ["n", "strategy", "replication"]
    ).reset_index(drop=True)

    expected_rows = REPLICATIONS * len(SAMPLE_SIZES) * 2
    assert len(replication_df) == expected_rows
    assert replication_df["return_percentage"].notna().all()

    paired_df = create_paired_dataset(replication_df)
    descriptive_df = descriptive_statistics(replication_df)
    inference_df = paired_inference(paired_df)

    make_figures(
        replication_df,
        paired_df,
        descriptive_df,
        inference_df,
    )

    zip_path = save_outputs(
        replication_df,
        paired_df,
        descriptive_df,
        inference_df,
    )

    print("\nDESCRIPTIVE STATISTICS")
    print("-" * 72)
    print(descriptive_df.round(6).to_string(index=False))

    print("\nPAIRED INFERENCE")
    print("-" * 72)
    print(inference_df.round(6).to_string(index=False))

    print("\nFILES CREATED")
    print("-" * 72)
    for path in sorted(OUTPUT_FOLDER.iterdir()):
        print(path)
    print(f"ZIP archive: {zip_path}")

    # Automatic download when run in Google Colab.
    try:
        from google.colab import files

        files.download(str(zip_path))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
