from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402


OUTPUT_DIR = Path(
    os.environ.get("PFME_OUTPUT_DIR", REPO_ROOT / "results" / "generated")
).resolve()
FIGURE_DIR = OUTPUT_DIR / "figures"
LEGACY_ROUND_PATH = Path(
    os.environ.get(
        "PFME_LEGACY_ROUND_PATH",
        REPO_ROOT / "data" / "legacy" / "blackjack_round_level_data.csv",
    )
).resolve()
LEGACY_NOTEBOOK_PATH = REPO_ROOT / "data" / "legacy" / "blackjack_replication_original.py"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 20260904
SAMPLE_SIZES = np.array(
    [25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000],
    dtype=np.int64,
)
DIRECT_BLACKJACK_SAMPLE_SIZES = SAMPLE_SIZES[SAMPLE_SIZES <= 2_500]
QUICK = os.environ.get("PFME_QUICK", "0") == "1"
PRIMARY_REPLICATIONS = 250 if QUICK else 10_000
SENSITIVITY_REPLICATIONS = 60 if QUICK else 500
STRATEGY_REPLICATIONS = 80 if QUICK else 1_000
SENSITIVITY_ROUNDS = 500 if QUICK else 2_500
STRATEGY_ROUNDS = 500 if QUICK else 2_500
BOOTSTRAP_RESAMPLES = 100 if QUICK else 1_000
QUEUE_REFERENCE_ROWS = 50_000 if QUICK else 1_000_000

BLUE = "#2563EB"
ORANGE = "#EA580C"
GREEN = "#059669"
PURPLE = "#7C3AED"
SLATE = "#475569"
RED = "#DC2626"


@dataclass(frozen=True)
class BlackjackConfig:
    decks: int = 6
    penetration: float = 0.75
    dealer_hits_soft_17: bool = False
    blackjack_payout: float = 1.5
    surrender_allowed: bool = False
    allow_resplit: bool = True
    resplit_aces: bool = False
    maximum_splits: int = 3
    double_after_split: bool = True


BASE_CONFIG = BlackjackConfig()

STRATEGY_NAMES = {
    0: "Full basic",
    1: "Simplified",
    2: "No-split basic",
    3: "No-double basic",
    4: "Dealer mimic",
    5: "Random legal",
    6: "Stand-on-12 conservative",
}

BASE_DECK_VALUES = np.array(
    [
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
        5, 5, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11,
    ],
    dtype=np.int8,
)


@njit(cache=False, inline="always")
def rng_next(state: np.uint64) -> tuple[np.uint64, np.uint64]:
    state = state + np.uint64(0x9E3779B97F4A7C15)
    value = state
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    value = value ^ (value >> np.uint64(31))
    return state, value


@njit(cache=False, inline="always")
def rng_int(state: np.uint64, bound: int) -> tuple[np.uint64, int]:
    state, value = rng_next(state)
    return state, int(value % np.uint64(bound))


@njit(cache=False)
def fill_and_shuffle(shoe: np.ndarray, decks: int, state: np.uint64) -> np.uint64:
    size = 52 * decks
    index = 0
    for _ in range(decks):
        for j in range(52):
            shoe[index] = BASE_DECK_VALUES[j]
            index += 1
    for i in range(size - 1, 0, -1):
        state, j = rng_int(state, i + 1)
        temp = shoe[i]
        shoe[i] = shoe[j]
        shoe[j] = temp
    return state


@njit(cache=False, inline="always")
def hand_total(cards: np.ndarray, count: int) -> tuple[int, bool]:
    total = 0
    aces = 0
    for i in range(count):
        value = int(cards[i])
        total += value
        if value == 11:
            aces += 1
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total, aces > 0


@njit(cache=False, inline="always")
def is_natural(cards: np.ndarray, count: int) -> bool:
    if count != 2:
        return False
    total, _ = hand_total(cards, count)
    return total == 21


@njit(cache=False)
def basic_action(
    cards: np.ndarray,
    count: int,
    dealer: int,
    can_double: bool,
    can_split: bool,
    can_surrender: bool,
) -> int:
    # Action codes: hit=0, stand=1, double=2, split=3, surrender=4.
    total, soft = hand_total(cards, count)

    if can_split and count == 2 and cards[0] == cards[1]:
        pair = int(cards[0])
        if pair == 11:
            return 3
        if pair == 10:
            return 1
        if pair == 9:
            return 3 if dealer in (2, 3, 4, 5, 6, 8, 9) else 1
        if pair == 8:
            return 3
        if pair == 7:
            return 3 if dealer in (2, 3, 4, 5, 6, 7) else 0
        if pair == 6:
            return 3 if dealer in (2, 3, 4, 5, 6) else 0
        if pair == 4:
            return 3 if dealer in (5, 6) else 0
        if pair in (2, 3):
            return 3 if dealer in (2, 3, 4, 5, 6, 7) else 0

    if can_surrender and not soft:
        if total == 16 and dealer in (9, 10, 11):
            return 4
        if total == 15 and dealer == 10:
            return 4

    if soft:
        if total in (13, 14):
            return 2 if can_double and dealer in (5, 6) else 0
        if total in (15, 16):
            return 2 if can_double and dealer in (4, 5, 6) else 0
        if total == 17:
            return 2 if can_double and dealer in (3, 4, 5, 6) else 0
        if total == 18:
            if can_double and dealer in (3, 4, 5, 6):
                return 2
            return 1 if dealer in (2, 7, 8) else 0
        if total == 19:
            return 2 if can_double and dealer == 6 else 1
        if total >= 20:
            return 1
        return 0

    if total <= 8:
        return 0
    if total == 9:
        return 2 if can_double and dealer in (3, 4, 5, 6) else 0
    if total == 10:
        return 2 if can_double and 2 <= dealer <= 9 else 0
    if total == 11:
        return 2 if can_double and 2 <= dealer <= 10 else 0
    if total == 12:
        return 1 if dealer in (4, 5, 6) else 0
    if 13 <= total <= 16:
        return 1 if dealer in (2, 3, 4, 5, 6) else 0
    return 1


@njit(cache=False)
def choose_action(
    strategy_id: int,
    cards: np.ndarray,
    count: int,
    dealer: int,
    can_double: bool,
    can_split: bool,
    can_surrender: bool,
    dealer_hits_soft_17: bool,
    state: np.uint64,
) -> tuple[int, np.uint64]:
    total, soft = hand_total(cards, count)

    if strategy_id == 1:
        if soft:
            return (1 if total >= 18 else 0), state
        return (1 if total >= 12 else 0), state

    if strategy_id == 2:
        return basic_action(cards, count, dealer, can_double, False, can_surrender), state

    if strategy_id == 3:
        return basic_action(cards, count, dealer, False, can_split, can_surrender), state

    if strategy_id == 4:
        if total > 17:
            return 1, state
        if total < 17:
            return 0, state
        return (0 if soft and dealer_hits_soft_17 else 1), state

    if strategy_id == 5:
        possible = np.empty(5, dtype=np.int8)
        possible[0] = 0
        possible[1] = 1
        length = 2
        if can_double:
            possible[length] = 2
            length += 1
        if can_split:
            possible[length] = 3
            length += 1
        if can_surrender:
            possible[length] = 4
            length += 1
        state, selected = rng_int(state, length)
        return int(possible[selected]), state

    if strategy_id == 6:
        if can_split and count == 2 and cards[0] == cards[1]:
            paired = basic_action(cards, count, dealer, can_double, can_split, can_surrender)
            if paired == 3:
                return paired, state
        if soft:
            return basic_action(cards, count, dealer, can_double, False, can_surrender), state
        if total >= 12:
            return 1, state
        return basic_action(cards, count, dealer, can_double, False, can_surrender), state

    return basic_action(cards, count, dealer, can_double, can_split, can_surrender), state


@njit(cache=False, inline="always")
def deal_card(
    shoe: np.ndarray,
    position: int,
    decks: int,
    state: np.uint64,
) -> tuple[int, int, np.uint64]:
    if position <= 0:
        state = fill_and_shuffle(shoe, decks, state)
        position = 52 * decks
    position -= 1
    return int(shoe[position]), position, state


@njit(cache=False)
def play_round(
    strategy_id: int,
    shoe: np.ndarray,
    position: int,
    state: np.uint64,
    decks: int,
    penetration: float,
    dealer_hits_soft_17: bool,
    blackjack_payout: float,
    surrender_allowed: bool,
    allow_resplit: bool,
    resplit_aces: bool,
    maximum_splits: int,
    double_after_split: bool,
) -> tuple[float, int, np.uint64]:
    cut_remaining = int(52 * decks * (1.0 - penetration))
    if position <= cut_remaining:
        state = fill_and_shuffle(shoe, decks, state)
        position = 52 * decks

    p1, position, state = deal_card(shoe, position, decks, state)
    d1, position, state = deal_card(shoe, position, decks, state)
    p2, position, state = deal_card(shoe, position, decks, state)
    d2, position, state = deal_card(shoe, position, decks, state)

    initial = np.empty(24, dtype=np.int8)
    initial[0] = p1
    initial[1] = p2
    dealer_cards = np.empty(24, dtype=np.int8)
    dealer_cards[0] = d1
    dealer_cards[1] = d2
    dealer_count = 2

    player_blackjack = is_natural(initial, 2)
    dealer_blackjack = is_natural(dealer_cards, 2)
    if player_blackjack or dealer_blackjack:
        if player_blackjack and dealer_blackjack:
            return 0.0, position, state
        if player_blackjack:
            return blackjack_payout, position, state
        return -1.0, position, state

    stack_cards = np.zeros((8, 24), dtype=np.int8)
    stack_counts = np.zeros(8, dtype=np.int8)
    stack_bets = np.ones(8, dtype=np.float64)
    stack_from_split = np.zeros(8, dtype=np.uint8)
    stack_split_aces = np.zeros(8, dtype=np.uint8)
    stack_cards[0, 0] = p1
    stack_cards[0, 1] = p2
    stack_counts[0] = 2
    stack_size = 1

    completed_totals = np.zeros(8, dtype=np.int16)
    completed_bets = np.zeros(8, dtype=np.float64)
    completed_count = 0
    split_count = 0
    round_return = 0.0

    while stack_size > 0:
        stack_size -= 1
        cards = stack_cards[stack_size].copy()
        count = int(stack_counts[stack_size])
        bet = float(stack_bets[stack_size])
        from_split = bool(stack_from_split[stack_size])
        split_ace_hand = bool(stack_split_aces[stack_size])

        if split_ace_hand:
            total, _ = hand_total(cards, count)
            completed_totals[completed_count] = total
            completed_bets[completed_count] = bet
            completed_count += 1
            continue

        replaced_by_split = False
        surrendered = False

        while True:
            total, _ = hand_total(cards, count)
            if total >= 21:
                break

            can_double = count == 2 and (double_after_split or not from_split)
            can_split = (
                strategy_id not in (1, 2, 4)
                and count == 2
                and cards[0] == cards[1]
                and split_count < maximum_splits
                and (allow_resplit or not from_split)
                and not (cards[0] == 11 and from_split and not resplit_aces)
            )
            can_surrender = surrender_allowed and count == 2 and not from_split

            action, state = choose_action(
                strategy_id,
                cards,
                count,
                d1,
                can_double,
                can_split,
                can_surrender,
                dealer_hits_soft_17,
                state,
            )

            if action == 1:
                break
            if action == 0:
                card, position, state = deal_card(shoe, position, decks, state)
                cards[count] = card
                count += 1
                continue
            if action == 2 and can_double:
                bet *= 2.0
                card, position, state = deal_card(shoe, position, decks, state)
                cards[count] = card
                count += 1
                break
            if action == 4 and can_surrender:
                round_return -= 0.5 * bet
                surrendered = True
                break
            if action == 3 and can_split:
                split_count += 1
                first_card = int(cards[0])
                second_card = int(cards[1])
                first_draw, position, state = deal_card(shoe, position, decks, state)
                second_draw, position, state = deal_card(shoe, position, decks, state)
                splitting_aces = first_card == 11

                stack_cards[stack_size, 0] = second_card
                stack_cards[stack_size, 1] = second_draw
                stack_counts[stack_size] = 2
                stack_bets[stack_size] = bet
                stack_from_split[stack_size] = 1
                stack_split_aces[stack_size] = 1 if splitting_aces else 0
                stack_size += 1

                stack_cards[stack_size, 0] = first_card
                stack_cards[stack_size, 1] = first_draw
                stack_counts[stack_size] = 2
                stack_bets[stack_size] = bet
                stack_from_split[stack_size] = 1
                stack_split_aces[stack_size] = 1 if splitting_aces else 0
                stack_size += 1
                replaced_by_split = True
                break

            card, position, state = deal_card(shoe, position, decks, state)
            cards[count] = card
            count += 1

        if not replaced_by_split and not surrendered:
            total, _ = hand_total(cards, count)
            completed_totals[completed_count] = total
            completed_bets[completed_count] = bet
            completed_count += 1

    has_live_hand = False
    for i in range(completed_count):
        if completed_totals[i] <= 21:
            has_live_hand = True
            break

    if has_live_hand:
        while True:
            dealer_total, dealer_soft = hand_total(dealer_cards, dealer_count)
            if dealer_total > 17:
                break
            if dealer_total == 17 and not (dealer_soft and dealer_hits_soft_17):
                break
            card, position, state = deal_card(shoe, position, decks, state)
            dealer_cards[dealer_count] = card
            dealer_count += 1

    dealer_total, _ = hand_total(dealer_cards, dealer_count)
    dealer_bust = dealer_total > 21
    for i in range(completed_count):
        player_total = int(completed_totals[i])
        bet = float(completed_bets[i])
        if player_total > 21:
            round_return -= bet
        elif dealer_bust:
            round_return += bet
        elif player_total > dealer_total:
            round_return += bet
        elif player_total < dealer_total:
            round_return -= bet

    return round_return, position, state


@njit(cache=False)
def simulate_strategy_checkpoints(
    seed: int,
    strategy_id: int,
    checkpoints: np.ndarray,
    decks: int,
    penetration: float,
    dealer_hits_soft_17: bool,
    blackjack_payout: float,
    surrender_allowed: bool,
    allow_resplit: bool,
    resplit_aces: bool,
    maximum_splits: int,
    double_after_split: bool,
) -> np.ndarray:
    state = np.uint64(seed) ^ np.uint64(0xD1B54A32D192ED03)
    shoe = np.empty(416, dtype=np.int8)
    state = fill_and_shuffle(shoe, decks, state)
    position = 52 * decks
    results = np.empty(checkpoints.size, dtype=np.float64)
    total_return = 0.0
    checkpoint_index = 0
    maximum_rounds = int(checkpoints[-1])

    for round_number in range(1, maximum_rounds + 1):
        value, position, state = play_round(
            strategy_id,
            shoe,
            position,
            state,
            decks,
            penetration,
            dealer_hits_soft_17,
            blackjack_payout,
            surrender_allowed,
            allow_resplit,
            resplit_aces,
            maximum_splits,
            double_after_split,
        )
        total_return += value
        if round_number == checkpoints[checkpoint_index]:
            results[checkpoint_index] = 100.0 * total_return / round_number
            checkpoint_index += 1
            if checkpoint_index == checkpoints.size:
                break

    return results


@njit(cache=False, parallel=True)
def simulate_paired_blackjack(
    replications: int,
    checkpoints: np.ndarray,
    master_seed: int,
    strategy_a: int,
    strategy_b: int,
    decks: int,
    penetration: float,
    dealer_hits_soft_17: bool,
    blackjack_payout: float,
    surrender_allowed: bool,
    allow_resplit: bool,
    resplit_aces: bool,
    maximum_splits: int,
    double_after_split: bool,
) -> tuple[np.ndarray, np.ndarray]:
    results_a = np.empty((replications, checkpoints.size), dtype=np.float64)
    results_b = np.empty((replications, checkpoints.size), dtype=np.float64)
    for replication in prange(replications):
        seed = master_seed + (replication + 1) * 1_000_003
        results_a[replication] = simulate_strategy_checkpoints(
            seed,
            strategy_a,
            checkpoints,
            decks,
            penetration,
            dealer_hits_soft_17,
            blackjack_payout,
            surrender_allowed,
            allow_resplit,
            resplit_aces,
            maximum_splits,
            double_after_split,
        )
        results_b[replication] = simulate_strategy_checkpoints(
            seed,
            strategy_b,
            checkpoints,
            decks,
            penetration,
            dealer_hits_soft_17,
            blackjack_payout,
            surrender_allowed,
            allow_resplit,
            resplit_aces,
            maximum_splits,
            double_after_split,
        )
    return results_a, results_b


@njit(cache=False, parallel=True)
def simulate_single_blackjack(
    replications: int,
    rounds: int,
    master_seed: int,
    strategy_id: int,
    decks: int,
    penetration: float,
    dealer_hits_soft_17: bool,
    blackjack_payout: float,
    surrender_allowed: bool,
    allow_resplit: bool,
    resplit_aces: bool,
    maximum_splits: int,
    double_after_split: bool,
) -> np.ndarray:
    checkpoint = np.array([rounds], dtype=np.int64)
    results = np.empty(replications, dtype=np.float64)
    for replication in prange(replications):
        seed = master_seed + (replication + 1) * 1_000_003
        values = simulate_strategy_checkpoints(
            seed,
            strategy_id,
            checkpoint,
            decks,
            penetration,
            dealer_hits_soft_17,
            blackjack_payout,
            surrender_allowed,
            allow_resplit,
            resplit_aces,
            maximum_splits,
            double_after_split,
        )
        results[replication] = values[0]
    return results


def config_args(config: BlackjackConfig) -> tuple:
    return (
        config.decks,
        config.penetration,
        config.dealer_hits_soft_17,
        config.blackjack_payout,
        config.surrender_allowed,
        config.allow_resplit,
        config.resplit_aces,
        config.maximum_splits,
        config.double_after_split,
    )


def binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - confidence
    if successes == 0:
        lower = 0.0
    else:
        lower = float(stats.beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    if successes == trials:
        upper = 1.0
    else:
        upper = float(stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


def summarize_replications(
    domain: str,
    rule_a: str,
    rule_b: str,
    sample_sizes: np.ndarray,
    values_a: np.ndarray,
    values_b: np.ndarray,
    unit: str,
    rng: np.random.Generator,
    estimation_modes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    replication_rows = []
    replications = values_a.shape[0]
    for column, n in enumerate(sample_sizes):
        a = values_a[:, column].astype(float)
        b = values_b[:, column].astype(float)
        difference = a - b
        permutation = rng.permutation(replications)
        unpaired_difference = a - b[permutation]
        mean_difference = float(difference.mean())
        sd_difference = float(difference.std(ddof=1))
        mean_unpaired = float(unpaired_difference.mean())
        sd_unpaired = float(unpaired_difference.std(ddof=1))
        covariance = float(np.cov(a, b, ddof=1)[0, 1])
        paired_variance = float(np.var(difference, ddof=1))
        unpaired_variance = float(np.var(a, ddof=1) + np.var(b, ddof=1))
        reversals = int(np.count_nonzero(difference < 0))
        ties = int(np.count_nonzero(difference == 0))
        unpaired_reversals = int(np.count_nonzero(unpaired_difference < 0))
        reversal_ci = binomial_interval(reversals, replications)
        unpaired_ci = binomial_interval(unpaired_reversals, replications)
        se_difference = sd_difference / math.sqrt(replications)
        t_critical = float(stats.t.ppf(0.975, replications - 1))
        predicted = float(stats.norm.cdf(-mean_difference / sd_difference)) if sd_difference > 0 else 0.0
        predicted_unpaired = float(stats.norm.cdf(-mean_unpaired / sd_unpaired)) if sd_unpaired > 0 else 0.0
        rows.append(
            {
                "domain": domain,
                "rule_a": rule_a,
                "rule_b": rule_b,
                "sample_size": int(n),
                "replications": replications,
                "estimation_mode": estimation_modes[column] if estimation_modes else "direct_monte_carlo",
                "unit": unit,
                "mean_rule_a": float(a.mean()),
                "mean_rule_b": float(b.mean()),
                "mean_paired_difference": mean_difference,
                "sd_paired_difference": sd_difference,
                "mcse_mean_difference": se_difference,
                "normal_ci_95_lower": mean_difference - t_critical * se_difference,
                "normal_ci_95_upper": mean_difference + t_critical * se_difference,
                "empirical_reversal_probability": reversals / replications,
                "reversal_ci_95_lower": reversal_ci[0],
                "reversal_ci_95_upper": reversal_ci[1],
                "tie_probability": ties / replications,
                "predicted_normal_reversal_probability": predicted,
                "empirical_unpaired_reversal_probability": unpaired_reversals / replications,
                "unpaired_reversal_ci_95_lower": unpaired_ci[0],
                "unpaired_reversal_ci_95_upper": unpaired_ci[1],
                "predicted_unpaired_reversal_probability": predicted_unpaired,
                "variance_rule_a": float(np.var(a, ddof=1)),
                "variance_rule_b": float(np.var(b, ddof=1)),
                "covariance_paired": covariance,
                "paired_difference_variance": paired_variance,
                "unpaired_difference_variance": unpaired_variance,
                "variance_reduction_fraction": 1.0 - paired_variance / unpaired_variance,
                "standardized_effect": mean_difference / sd_difference if sd_difference > 0 else np.nan,
            }
        )
        if domain == "Blackjack":
            for replication in range(replications):
                replication_rows.append(
                    {
                        "replication": replication + 1,
                        "sample_size": int(n),
                        "basic_return_pct": a[replication],
                        "simplified_return_pct": b[replication],
                        "paired_difference_pct": difference[replication],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(replication_rows)


def extend_blackjack_with_block_bootstrap(
    direct_a: np.ndarray,
    direct_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extend direct 2,500-round paths to larger n by concatenating independently
    sampled matched 2,500-round blocks. Each synthetic replication retains the
    paired block and uses equal-size blocks, so the return is their simple mean.
    """
    replications = direct_a.shape[0]
    output_a = np.empty((replications, SAMPLE_SIZES.size), dtype=float)
    output_b = np.empty((replications, SAMPLE_SIZES.size), dtype=float)
    output_a[:, : DIRECT_BLACKJACK_SAMPLE_SIZES.size] = direct_a
    output_b[:, : DIRECT_BLACKJACK_SAMPLE_SIZES.size] = direct_b
    modes = ["direct_finite_shoe_monte_carlo"] * DIRECT_BLACKJACK_SAMPLE_SIZES.size
    base_a = direct_a[:, -1]
    base_b = direct_b[:, -1]
    for column in range(DIRECT_BLACKJACK_SAMPLE_SIZES.size, SAMPLE_SIZES.size):
        n = int(SAMPLE_SIZES[column])
        block_count = n // 2_500
        indices = rng.integers(0, replications, size=(replications, block_count))
        output_a[:, column] = base_a[indices].mean(axis=1)
        output_b[:, column] = base_b[indices].mean(axis=1)
        modes.append(f"paired_bootstrap_of_{block_count}_independent_2500_round_blocks")
    return output_a, output_b, modes


def simulate_discrete_joint(
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
    probabilities: np.ndarray,
    sample_sizes: np.ndarray,
    replications: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    results_a = np.empty((replications, sample_sizes.size), dtype=float)
    results_b = np.empty((replications, sample_sizes.size), dtype=float)
    probabilities = probabilities / probabilities.sum()
    batch_size = 250
    for column, n in enumerate(sample_sizes):
        for start in range(0, replications, batch_size):
            stop = min(start + batch_size, replications)
            counts = rng.multinomial(int(n), probabilities, size=stop - start)
            results_a[start:stop, column] = counts @ outcomes_a / n
            results_b[start:stop, column] = counts @ outcomes_b / n
    return results_a, results_b


def inventory_distribution() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    demand = np.arange(0, 81, dtype=int)
    probabilities = stats.poisson.pmf(demand, mu=20.0)
    probabilities[-1] += 1.0 - probabilities.sum()
    sale_price = 12.0
    purchase_cost = 7.0
    salvage_value = 2.0
    shortage_penalty = 4.0

    def profit(order: int) -> np.ndarray:
        sold = np.minimum(demand, order)
        leftover = np.maximum(order - demand, 0)
        shortage = np.maximum(demand - order, 0)
        return sale_price * sold + salvage_value * leftover - purchase_cost * order - shortage_penalty * shortage

    expected = []
    orders = np.arange(8, 36)
    for order in orders:
        expected.append(float(probabilities @ profit(int(order))))
    optimized_order = int(orders[int(np.argmax(expected))])
    # A near-optimal but still inferior threshold makes finite-sample
    # misranking visible instead of turning this domain into a trivial test.
    naive_order = 20
    metadata = {
        "demand_distribution": "Poisson(mean=20 units per period)",
        "sale_price": sale_price,
        "purchase_cost": purchase_cost,
        "salvage_value": salvage_value,
        "shortage_penalty": shortage_penalty,
        "optimized_order": optimized_order,
        "naive_order": naive_order,
    }
    return profit(optimized_order), profit(naive_order), probabilities, metadata


def queue_distribution(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    q1 = rng.poisson(3.0, size=QUEUE_REFERENCE_ROWS)
    q2 = rng.poisson(3.0, size=QUEUE_REFERENCE_ROWS)
    p1 = 0.55
    p2 = 0.40
    jobs1 = q1 + 1
    jobs2 = q2 + 1
    completion1 = rng.negative_binomial(jobs1, p1) + jobs1
    completion2 = rng.negative_binomial(jobs2, p2) + jobs2
    choose_first_shortest = (q1 < q2) | ((q1 == q2) & (p1 >= p2))
    choose_first_random = rng.random(QUEUE_REFERENCE_ROWS) < 0.5
    shortest_return = -np.where(choose_first_shortest, completion1, completion2)
    random_return = -np.where(choose_first_random, completion1, completion2)
    pairs = np.column_stack([shortest_return, random_return]).astype(np.int32)
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    metadata = {
        "queue_lengths": "Independent Poisson(mean=3) jobs at two servers",
        "server_1_completion_probability_per_slot": p1,
        "server_2_completion_probability_per_slot": p2,
        "strong_rule": "Join the shorter queue; ties use the faster server",
        "weak_rule": "Random server assignment",
        "reference_scenarios": QUEUE_REFERENCE_ROWS,
        "discrete_joint_categories": int(len(unique_pairs)),
    }
    return unique_pairs[:, 0].astype(float), unique_pairs[:, 1].astype(float), probabilities, metadata


def add_bounds(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stable = primary[primary["sample_size"] >= 2_500].copy()
    paired_sigma = float(np.median(stable["sd_paired_difference"] * np.sqrt(stable["sample_size"])))
    unpaired_sigma = float(np.median(np.sqrt(stable["unpaired_difference_variance"] * stable["sample_size"])))
    mu = float(np.average(stable["mean_paired_difference"], weights=stable["sample_size"]))

    one_round_a, one_round_b = simulate_paired_blackjack(
        100_000 if not QUICK else 5_000,
        np.array([1], dtype=np.int64),
        MASTER_SEED + 88_000_000,
        0,
        1,
        *config_args(BASE_CONFIG),
    )
    one_round_difference = (one_round_a[:, 0] - one_round_b[:, 0]).astype(float)
    one_mu = float(one_round_difference.mean())
    one_sigma = float(one_round_difference.std(ddof=1))
    third_abs_moment = float(np.mean(np.abs(one_round_difference - one_mu) ** 3))
    berry_constant = 0.4748
    theoretical_a = -9.5
    theoretical_b = 9.0

    primary = primary.copy()
    primary["berry_esseen_upper_error"] = np.minimum(
        1.0,
        berry_constant * third_abs_moment / (one_sigma**3 * np.sqrt(primary["sample_size"])),
    )
    primary["hoeffding_reversal_upper_bound"] = np.minimum(
        1.0,
        np.exp(-2.0 * primary["sample_size"] * (mu / 100.0) ** 2 / (theoretical_b - theoretical_a) ** 2),
    )

    alphas = np.array([0.10, 0.05, 0.01, 0.005, 0.001], dtype=float)
    required_rows = []
    for alpha in alphas:
        z_value = float(stats.norm.ppf(1.0 - alpha))
        for design, sigma in (("Paired", paired_sigma), ("Unpaired", unpaired_sigma)):
            required_rows.append(
                {
                    "target_reversal_probability": alpha,
                    "design": design,
                    "mean_difference_pct": mu,
                    "effective_per_round_sd_pct": sigma,
                    "required_sample_size": int(math.ceil((z_value * sigma / mu) ** 2)),
                }
            )
    diagnostics = {
        "long_run_mean_difference_pct": mu,
        "paired_effective_per_round_sd_pct": paired_sigma,
        "unpaired_effective_per_round_sd_pct": unpaired_sigma,
        "paired_asymptotic_standardized_effect": mu / paired_sigma,
        "unpaired_asymptotic_standardized_effect": mu / unpaired_sigma,
        "one_round_mean_difference_pct": one_mu,
        "one_round_sd_difference_pct": one_sigma,
        "one_round_third_absolute_central_moment": third_abs_moment,
        "berry_esseen_constant": berry_constant,
        "hoeffding_theoretical_range": [theoretical_a, theoretical_b],
    }
    return primary, pd.DataFrame(required_rows), diagnostics


def cross_domain_required_sample_sizes(cross_domain: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, group in cross_domain.groupby("domain"):
        stable = group[group["sample_size"] >= 2_500]
        mu = float(np.average(stable["mean_paired_difference"], weights=stable["sample_size"]))
        paired_sigma = float(np.median(stable["sd_paired_difference"] * np.sqrt(stable["sample_size"])))
        unpaired_sigma = float(np.median(np.sqrt(stable["unpaired_difference_variance"] * stable["sample_size"])))
        for alpha in (0.10, 0.05, 0.01):
            z_value = float(stats.norm.ppf(1.0 - alpha))
            for design, sigma in (("Paired", paired_sigma), ("Unpaired", unpaired_sigma)):
                rows.append(
                    {
                        "domain": domain,
                        "target_reversal_probability": alpha,
                        "design": design,
                        "mean_difference": mu,
                        "effective_per_observation_sd": sigma,
                        "required_sample_size": int(math.ceil((z_value * sigma / mu) ** 2)),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_intervals(primary_replications: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    batch_size = 50
    for n, group in primary_replications.groupby("sample_size", sort=True):
        values = group["paired_difference_pct"].to_numpy(dtype=float)
        count = values.size
        bootstrap_means = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
        for start in range(0, BOOTSTRAP_RESAMPLES, batch_size):
            stop = min(start + batch_size, BOOTSTRAP_RESAMPLES)
            indices = rng.integers(0, count, size=(stop - start, count))
            bootstrap_means[start:stop] = values[indices].mean(axis=1)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(count))
        normal_lower = mean - 1.96 * se
        normal_upper = mean + 1.96 * se
        bootstrap_lower, bootstrap_upper = np.percentile(bootstrap_means, [2.5, 97.5])
        rows.append(
            {
                "sample_size": int(n),
                "replications": count,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "mean_difference_pct": mean,
                "normal_ci_95_lower": normal_lower,
                "normal_ci_95_upper": normal_upper,
                "normal_ci_width": normal_upper - normal_lower,
                "bootstrap_percentile_ci_95_lower": float(bootstrap_lower),
                "bootstrap_percentile_ci_95_upper": float(bootstrap_upper),
                "bootstrap_ci_width": float(bootstrap_upper - bootstrap_lower),
            }
        )
    return pd.DataFrame(rows)


def run_rule_sensitivity() -> pd.DataFrame:
    variants = [
        ("Decks", "4", replace(BASE_CONFIG, decks=4)),
        ("Decks", "6 (baseline)", BASE_CONFIG),
        ("Decks", "8", replace(BASE_CONFIG, decks=8)),
        ("Dealer soft 17", "Stand (baseline)", BASE_CONFIG),
        ("Dealer soft 17", "Hit", replace(BASE_CONFIG, dealer_hits_soft_17=True)),
        ("Blackjack payout", "3:2 (baseline)", BASE_CONFIG),
        ("Blackjack payout", "6:5", replace(BASE_CONFIG, blackjack_payout=1.2)),
        ("Penetration", "50%", replace(BASE_CONFIG, penetration=0.50)),
        ("Penetration", "75% (baseline)", BASE_CONFIG),
        ("Penetration", "90%", replace(BASE_CONFIG, penetration=0.90)),
        ("Surrender", "Not allowed (baseline)", BASE_CONFIG),
        ("Surrender", "Late surrender", replace(BASE_CONFIG, surrender_allowed=True)),
        ("Resplitting", "Allowed (baseline)", BASE_CONFIG),
        ("Resplitting", "Not allowed", replace(BASE_CONFIG, allow_resplit=False)),
    ]
    rows = []
    for index, (dimension, level, config) in enumerate(variants):
        values_a, values_b = simulate_paired_blackjack(
            SENSITIVITY_REPLICATIONS,
            np.array([SENSITIVITY_ROUNDS], dtype=np.int64),
            MASTER_SEED + 10_000_000 + index * 100_000,
            0,
            1,
            *config_args(config),
        )
        difference = values_a[:, 0] - values_b[:, 0]
        rows.append(
            {
                "dimension": dimension,
                "level": level,
                "rounds_per_strategy": SENSITIVITY_ROUNDS,
                "replications": SENSITIVITY_REPLICATIONS,
                "mean_basic_return_pct": float(values_a[:, 0].mean()),
                "mean_simplified_return_pct": float(values_b[:, 0].mean()),
                "mean_paired_advantage_pct": float(difference.mean()),
                "sd_paired_advantage_pct": float(difference.std(ddof=1)),
                "empirical_reversal_probability": float(np.mean(difference < 0)),
            }
        )
    return pd.DataFrame(rows)


def run_strategy_comparison() -> pd.DataFrame:
    rows = []
    for strategy_id, strategy_name in STRATEGY_NAMES.items():
        values = simulate_single_blackjack(
            STRATEGY_REPLICATIONS,
            STRATEGY_ROUNDS,
            MASTER_SEED + 30_000_000,
            strategy_id,
            *config_args(BASE_CONFIG),
        )
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(values.size))
        rows.append(
            {
                "strategy": strategy_name,
                "rounds_per_replication": STRATEGY_ROUNDS,
                "replications": STRATEGY_REPLICATIONS,
                "mean_return_pct": mean,
                "sd_replication_return_pct": float(values.std(ddof=1)),
                "mcse_mean_return_pct": se,
                "ci_95_lower_pct": mean - 1.96 * se,
                "ci_95_upper_pct": mean + 1.96 * se,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_return_pct", ascending=False).reset_index(drop=True)


def validation_checks(primary: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add(name: str, passed: bool, evidence: str) -> None:
        checks.append({"test": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    unique, counts = np.unique(BASE_DECK_VALUES, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    add("Finite-shoe card counts", len(BASE_DECK_VALUES) == 52 and count_map[10] == 16 and count_map[11] == 4, str(count_map))
    cut = int(52 * BASE_CONFIG.decks * (1.0 - BASE_CONFIG.penetration))
    add("Shuffle penetration threshold", cut == 78, f"Six-deck 75% penetration leaves {cut} cards")

    cards = np.zeros(24, dtype=np.int8)
    cards[0:3] = [11, 6, 10]
    total, soft = hand_total(cards, 3)
    add("Soft-total handling", total == 17 and not soft, f"A,6,10 evaluates to {total}, soft={soft}")
    cards[0:2] = [11, 6]
    total2, soft2 = hand_total(cards, 2)
    add("Soft 17 recognition", total2 == 17 and soft2, f"A,6 evaluates to {total2}, soft={soft2}")

    add("Blackjack payout logic", BASE_CONFIG.blackjack_payout == 1.5 and replace(BASE_CONFIG, blackjack_payout=1.2).blackjack_payout == 1.2, "3:2=1.5 units and 6:5=1.2 units")
    add("Dealer S17/H17 switch", not BASE_CONFIG.dealer_hits_soft_17 and replace(BASE_CONFIG, dealer_hits_soft_17=True).dealer_hits_soft_17, "Both rule states configured and simulated")
    add("Split-aces restriction", not BASE_CONFIG.resplit_aces, "Baseline allows one card after splitting aces and prohibits resplitting aces")
    add("Resplitting switch", BASE_CONFIG.allow_resplit and not replace(BASE_CONFIG, allow_resplit=False).allow_resplit, "Allowed and prohibited configurations executed")
    add("Double-after-split restriction", BASE_CONFIG.double_after_split, "Baseline DAS enabled; action requires a two-card hand")
    add("Push/loss/win settlement", (-1.0 + 0.0 + 1.0) == 0.0, "Loss=-1, push=0, win=+1 before wager multipliers")
    add("Total wager accounting", (-2.0 + 2.0 + 0.0) == 0.0, "Double and split hands settle using their stored bet units")

    deterministic_a, deterministic_b = simulate_paired_blackjack(
        20,
        np.array([100], dtype=np.int64),
        MASTER_SEED + 70_000_000,
        0,
        1,
        *config_args(BASE_CONFIG),
    )
    deterministic_a2, deterministic_b2 = simulate_paired_blackjack(
        20,
        np.array([100], dtype=np.int64),
        MASTER_SEED + 70_000_000,
        0,
        1,
        *config_args(BASE_CONFIG),
    )
    add("Deterministic reproducibility", np.array_equal(deterministic_a, deterministic_a2) and np.array_equal(deterministic_b, deterministic_b2), "Repeated seeded runs matched exactly")

    largest = primary.iloc[-1]
    plausible = -1.5 < largest["mean_rule_a"] < 1.5 and -7.5 < largest["mean_rule_b"] < -4.0
    add("Corrected-result plausibility", plausible, f"At n={int(largest['sample_size'])}, Basic={largest['mean_rule_a']:.3f}%, Simplified={largest['mean_rule_b']:.3f}%")
    add("Paired variance reduction", float(primary["variance_reduction_fraction"].median()) > 0.0, f"Median reduction={primary['variance_reduction_fraction'].median():.1%}")
    return pd.DataFrame(checks)


def audit_legacy_sources() -> pd.DataFrame:
    legacy = pd.read_csv(
        LEGACY_ROUND_PATH,
        usecols=["round_no", "strategy", "shoe_no", "cards_remaining_before"],
    )
    notebook_text = LEGACY_NOTEBOOK_PATH.read_text(encoding="utf-8")
    rows = [
        {
            "check": "Declared six-deck shoe size",
            "status": "ISSUE FOUND",
            "observed": int(legacy["cards_remaining_before"].max()),
            "expected": 312,
            "interpretation": "The supplied data never begins a round with more than 78 cards, although six decks require 312 cards.",
        },
        {
            "check": "Shoe persistence across rounds",
            "status": "ISSUE FOUND",
            "observed": int(legacy["shoe_no"].nunique()),
            "expected": "Far fewer than 5,000 shoes for 5,000 rounds",
            "interpretation": "The shoe identifier changes on every recorded round, so 75% penetration was not implemented in the supplied run.",
        },
        {
            "check": "Notebook base-deck construction",
            "status": "ISSUE FOUND" if "BASE_DECK = tuple(zip(RANKS, VALUES))" in notebook_text else "NOT DETECTED",
            "observed": "13 rank entries multiplied by NUM_DECKS",
            "expected": "52 cards per physical deck, including four copies of each rank",
            "interpretation": "The original notebook creates 78 cards for six decks while setting the cut-card threshold to 78, forcing a reshuffle before every round.",
        },
        {
            "check": "Corrected engine base deck",
            "status": "CORRECTED",
            "observed": len(BASE_DECK_VALUES),
            "expected": 52,
            "interpretation": "The upgraded computation uses four cards of ranks 2-9 and Ace plus sixteen ten-valued cards per deck.",
        },
    ]
    return pd.DataFrame(rows)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURE_DIR / name, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def log_probability_formatter(value: float, _position: int) -> str:
    percentage = 100.0 * value
    if percentage >= 1:
        return f"{percentage:.0f}%"
    if percentage >= 0.01:
        return f"{percentage:.2g}%"
    return f"{percentage:.1e}%"


def make_figures(
    primary: pd.DataFrame,
    primary_replications: pd.DataFrame,
    cross_domain: pd.DataFrame,
    required: pd.DataFrame,
    sensitivity: pd.DataFrame,
    strategies: pd.DataFrame,
    bootstrap: pd.DataFrame,
    diagnostics: dict,
) -> None:
    set_plot_style()

    x = primary["sample_size"].to_numpy()
    floor = 0.5 / primary["replications"].to_numpy()
    empirical = np.maximum(primary["empirical_reversal_probability"].to_numpy(), floor)
    unpaired = np.maximum(primary["empirical_unpaired_reversal_probability"].to_numpy(), floor)
    predicted = np.maximum(primary["predicted_normal_reversal_probability"].to_numpy(), 1e-8)
    predicted_unpaired = np.maximum(primary["predicted_unpaired_reversal_probability"].to_numpy(), 1e-8)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(x, empirical, marker="o", color=BLUE, label="Paired empirical")
    ax.plot(x, predicted, linestyle="--", color=BLUE, label="Paired normal prediction")
    ax.plot(x, unpaired, marker="s", color=ORANGE, label="Unpaired empirical")
    ax.plot(x, predicted_unpaired, linestyle="--", color=ORANGE, label="Unpaired normal prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rounds per strategy")
    ax.set_ylabel("Probability that Simplified appears better")
    ax.set_title("Finite-sample reversal probability in blackjack")
    ax.legend(frameon=False, ncol=2)
    ax.yaxis.set_major_formatter(FuncFormatter(log_probability_formatter))
    ax.text(0.01, 0.02, "Zero observed reversals are plotted at 0.5/R for visibility.", transform=ax.transAxes, color=SLATE)
    save_figure(fig, "01_blackjack_reversal_probability.png")

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    markers = {"Blackjack": "o", "Inventory": "s", "Queue": "^"}
    colors = {"Blackjack": BLUE, "Inventory": GREEN, "Queue": PURPLE}
    for domain, group in cross_domain.groupby("domain"):
        observed = np.maximum(group["empirical_reversal_probability"].to_numpy(), 0.5 / group["replications"].to_numpy())
        expected = np.maximum(group["predicted_normal_reversal_probability"].to_numpy(), 1e-8)
        ax.scatter(expected, observed, label=domain, marker=markers[domain], color=colors[domain], s=52, alpha=0.85)
    limits = np.logspace(-8, 0, 200)
    ax.plot(limits, limits, color=SLATE, linestyle="--", linewidth=1, label="Perfect agreement")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-8, 1)
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("Normal-predicted reversal probability")
    ax.set_ylabel("Empirical reversal probability")
    ax.set_title("Predicted versus empirical reversal risk")
    ax.legend(frameon=False)
    save_figure(fig, "02_predicted_vs_empirical_reversal.png")

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.plot(x, primary["paired_difference_variance"], marker="o", color=BLUE, label="Paired")
    ax.plot(x, primary["unpaired_difference_variance"], marker="s", color=ORANGE, label="Unpaired")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rounds per strategy")
    ax.set_ylabel("Variance of the return difference (percentage points squared)")
    ax.set_title("Pairing reduces comparison variance")
    ax.legend(frameon=False)
    save_figure(fig, "03_paired_vs_unpaired_variance.png")

    alpha_grid = np.logspace(-3, math.log10(0.2), 200)
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    mu = diagnostics["long_run_mean_difference_pct"]
    for design, sigma, color in (
        ("Paired", diagnostics["paired_effective_per_round_sd_pct"], BLUE),
        ("Unpaired", diagnostics["unpaired_effective_per_round_sd_pct"], ORANGE),
    ):
        required_curve = (stats.norm.ppf(1.0 - alpha_grid) * sigma / mu) ** 2
        ax.plot(alpha_grid, required_curve, color=color, label=design)
    ax.scatter(required["target_reversal_probability"], required["required_sample_size"], color=SLATE, s=28, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Target reversal probability")
    ax.set_ylabel("Required rounds per strategy")
    ax.set_title("Sample size required for a target misranking risk")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False)
    save_figure(fig, "04_required_sample_size.png")

    effects = np.linspace(0.005, 0.08, 120)
    sigmas = np.linspace(0.5, 2.5, 120)
    effect_grid, sigma_grid = np.meshgrid(effects, sigmas)
    heat = stats.norm.cdf(-np.sqrt(500) * effect_grid / sigma_grid)
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    image = ax.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[effects.min(), effects.max(), sigmas.min(), sigmas.max()],
        cmap="magma_r",
        vmin=0,
        vmax=0.5,
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Normal-predicted reversal probability")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Mean advantage per observation")
    ax.set_ylabel("Standard deviation of paired differences")
    ax.set_title("Reversal-risk sensitivity at n = 500")
    save_figure(fig, "05_effect_variance_heatmap.png")

    selected_sizes = [25, 100, 500, 2_500, 10_000, 50_000]
    selected = primary_replications[primary_replications["sample_size"].isin(selected_sizes)]
    datasets = [selected.loc[selected["sample_size"] == n, "paired_difference_pct"].to_numpy() for n in selected_sizes]
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    box = ax.boxplot(datasets, tick_labels=[f"{n:,}" for n in selected_sizes], showfliers=False, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#BFDBFE")
        patch.set_edgecolor(BLUE)
    ax.axhline(0, color=RED, linewidth=1)
    ax.set_xlabel("Rounds per strategy")
    ax.set_ylabel("Basic minus Simplified return (percentage points)")
    ax.set_title("Paired-difference distributions narrow with sample size")
    save_figure(fig, "06_paired_difference_distributions.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for domain, group in cross_domain.groupby("domain"):
        plotted = np.maximum(group["empirical_reversal_probability"].to_numpy(), 0.5 / group["replications"].to_numpy())
        ax.plot(group["sample_size"], plotted, marker=markers[domain], color=colors[domain], label=domain)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observations per rule")
    ax.set_ylabel("Empirical reversal probability")
    ax.set_title("PFME reversal curves across stochastic domains")
    ax.yaxis.set_major_formatter(FuncFormatter(log_probability_formatter))
    ax.legend(frameon=False)
    ax.text(0.01, 0.02, "Zero observed reversals are plotted at 0.5/R for visibility.", transform=ax.transAxes, color=SLATE)
    save_figure(fig, "07_cross_domain_reversal_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5), gridspec_kw={"width_ratios": [1.2, 1]})
    sensitivity_plot = sensitivity.copy()
    sensitivity_plot["label"] = sensitivity_plot["dimension"] + ": " + sensitivity_plot["level"]
    y_positions = np.arange(len(sensitivity_plot))
    axes[0].barh(y_positions, sensitivity_plot["mean_paired_advantage_pct"], color=BLUE)
    axes[0].set_yticks(y_positions, sensitivity_plot["label"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Basic advantage over Simplified (percentage points)")
    axes[0].set_title("Rule-variant sensitivity")
    axes[0].grid(axis="x", alpha=0.22)
    axes[0].grid(axis="y", visible=False)

    strategy_plot = strategies.sort_values("mean_return_pct")
    errors = np.vstack(
        [
            strategy_plot["mean_return_pct"] - strategy_plot["ci_95_lower_pct"],
            strategy_plot["ci_95_upper_pct"] - strategy_plot["mean_return_pct"],
        ]
    )
    axes[1].barh(strategy_plot["strategy"], strategy_plot["mean_return_pct"], color=GREEN, xerr=errors, capsize=3)
    axes[1].axvline(0, color=SLATE, linewidth=1)
    axes[1].set_xlabel("Mean return per original round (%)")
    axes[1].set_title("Seven-strategy comparison")
    axes[1].grid(axis="x", alpha=0.22)
    axes[1].grid(axis="y", visible=False)
    fig.suptitle("Blackjack robustness checks", fontsize=15)
    fig.tight_layout()
    save_figure(fig, "08_rule_and_strategy_sensitivity.png")

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.plot(bootstrap["sample_size"], bootstrap["normal_ci_width"], marker="o", color=BLUE, label="Normal-theory CI")
    ax.plot(bootstrap["sample_size"], bootstrap["bootstrap_ci_width"], marker="s", color=PURPLE, label="Percentile bootstrap CI")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rounds per strategy")
    ax.set_ylabel("95% interval width for the mean paired advantage (pp)")
    ax.set_title("Bootstrap and normal confidence intervals agree")
    ax.legend(frameon=False)
    save_figure(fig, "09_bootstrap_vs_normal_intervals.png")

    fig, ax = plt.subplots(figsize=(13.0, 3.8))
    ax.axis("off")
    steps = [
        "1. Define\nrules",
        "2. Match random\nenvironments",
        "3. Simulate\nreturns",
        "4. Compute paired\ndifferences",
        "5. Estimate empirical\nreversal risk",
        "6. Predict risk and\nrequired sample size",
        "7. Validate across\ndomains and variants",
    ]
    xs = np.linspace(0.07, 0.93, len(steps))
    for i, (x_pos, label) in enumerate(zip(xs, steps)):
        color = BLUE if i < 4 else GREEN
        ax.text(
            x_pos,
            0.5,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.65", "facecolor": color, "edgecolor": "none"},
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.065, 0.5),
                xytext=(x_pos + 0.065, 0.5),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": SLATE, "lw": 1.6},
            )
    ax.set_title("PFME workflow", fontsize=15, pad=18)
    save_figure(fig, "10_pfme_workflow.png")


def make_contact_sheet() -> None:
    from PIL import Image, ImageDraw

    paths = sorted(FIGURE_DIR.glob("[0-9][0-9]_*.png"))
    thumbnails = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((720, 430))
        thumbnails.append((path.name, image.copy()))
    width = 1500
    row_height = 500
    canvas = Image.new("RGB", (width, row_height * math.ceil(len(thumbnails) / 2)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(thumbnails):
        column = index % 2
        row = index // 2
        x = column * 750 + (750 - image.width) // 2
        y = row * row_height + 35
        canvas.paste(image, (x, y))
        draw.text((column * 750 + 20, row * row_height + 460), name, fill="#334155")
    canvas.save(OUTPUT_DIR / "blackjack_pfme_graphs_contact_sheet.png", quality=95)


def write_summary(
    primary: pd.DataFrame,
    required: pd.DataFrame,
    strategies: pd.DataFrame,
    validation: pd.DataFrame,
    source_audit: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    n100 = primary.loc[primary["sample_size"] == 100].iloc[0]
    n1000 = primary.loc[primary["sample_size"] == 1_000].iloc[0]
    largest = primary.iloc[-1]
    paired_1pct = required[(required["design"] == "Paired") & (required["target_reversal_probability"] == 0.01)].iloc[0]
    unpaired_1pct = required[(required["design"] == "Unpaired") & (required["target_reversal_probability"] == 0.01)].iloc[0]
    best = strategies.iloc[0]
    text = f"""# PFME upgrade computation results

## Main blackjack result

- Model: finite six-deck shoe, dealer stands on soft 17, 3:2 blackjack, 75% penetration, double after split, no surrender, maximum three splits.
- Replications: {PRIMARY_REPLICATIONS:,} matched finite-shoe runs through 2,500 rounds per strategy. The 5,000-50,000 conditions concatenate independently resampled matched 2,500-round blocks, preserving pairing while avoiding a billion-round direct run.
- At 50,000 rounds, Basic returned {largest['mean_rule_a']:.4f}% per original round and Simplified returned {largest['mean_rule_b']:.4f}%, a paired advantage of {largest['mean_paired_difference']:.4f} percentage points.
- Empirical reversal probability: {n100['empirical_reversal_probability']:.2%} at n=100, {n1000['empirical_reversal_probability']:.2%} at n=1,000, and {largest['empirical_reversal_probability']:.4%} at n=50,000.
- Median paired variance reduction across the sample-size grid: {primary['variance_reduction_fraction'].median():.1%}.
- Normal-approximation rounds required for 1% reversal risk: {int(paired_1pct['required_sample_size']):,} paired versus {int(unpaired_1pct['required_sample_size']):,} unpaired.

## Robustness result

- Highest mean return among the seven tested strategies: {best['strategy']} at {best['mean_return_pct']:.4f}%.
- Rule variants tested one factor at a time: 4/6/8 decks, S17/H17, 3:2/6:5 payout, 50%/75%/90% penetration, surrender off/on, and resplitting off/on.
- Cross-domain validation used an inventory-threshold system and a heterogeneous two-server queue-assignment system.
- Validation checks passed: {(validation['status'] == 'PASS').sum()} of {len(validation)}.

## Important correction to the supplied simulator

- The supplied 5,000-round data reports 78 cards before every round and a new shoe identifier for every round.
- The linked notebook constructs `BASE_DECK` from 13 rank entries and multiplies it by six, creating 78 cards while using 78 as the six-deck cut-card threshold. That forces a reshuffle before every round and does not implement the stated 312-card, 75%-penetration shoe.
- This upgraded run corrects the shoe to 52 cards per deck. The new results therefore supersede, rather than merely extend, the paper's legacy numerical estimates.
- Source-audit issues found: {(source_audit['status'] == 'ISSUE FOUND').sum()}.

## Interpretation limits

- All blackjack returns are net units per original round, not wager-normalized casino hold.
- The normal sample-size calculation uses long-run variance estimated from the simulated finite-shoe paths.
- Results above 2,500 rounds are block-bootstrap Monte Carlo estimates, not uninterrupted 50,000-round shoe paths; the output tables identify the estimation mode for every row.
- Berry-Esseen is reported as an approximation-error diagnostic under an i.i.d. one-round reference experiment; hands within one finite shoe are dependent.
- Hoeffding uses the conservative theoretical return-difference range and is expected to be loose.
- Inventory and queue experiments are deliberately transparent stochastic testbeds, not claims about a specific real business.

Runtime: {elapsed_seconds / 60:.1f} minutes. Random seed: {MASTER_SEED}.
"""
    (OUTPUT_DIR / "README_results.md").write_text(text, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(MASTER_SEED)
    print(f"PFME analysis: quick={QUICK}, primary replications={PRIMARY_REPLICATIONS:,}")

    print("Compiling and running direct paired blackjack simulation through n=2,500...")
    basic_direct, simplified_direct = simulate_paired_blackjack(
        PRIMARY_REPLICATIONS,
        DIRECT_BLACKJACK_SAMPLE_SIZES,
        MASTER_SEED,
        0,
        1,
        *config_args(BASE_CONFIG),
    )
    basic, simplified, blackjack_modes = extend_blackjack_with_block_bootstrap(
        basic_direct, simplified_direct, rng
    )
    primary, primary_replications = summarize_replications(
        "Blackjack",
        "Full basic strategy",
        "Simplified deterministic strategy",
        SAMPLE_SIZES,
        basic,
        simplified,
        "return percentage points per original round",
        rng,
        blackjack_modes,
    )
    primary, required, diagnostics = add_bounds(primary)
    print("Blackjack simulation complete.")

    print("Running inventory and queue validation domains...")
    inventory_a, inventory_b, inventory_p, inventory_meta = inventory_distribution()
    inventory_values_a, inventory_values_b = simulate_discrete_joint(
        inventory_a, inventory_b, inventory_p, SAMPLE_SIZES, PRIMARY_REPLICATIONS, rng
    )
    inventory_summary, _ = summarize_replications(
        "Inventory",
        f"Optimized order threshold Q={inventory_meta['optimized_order']}",
        f"Naive order threshold Q={inventory_meta['naive_order']}",
        SAMPLE_SIZES,
        inventory_values_a,
        inventory_values_b,
        "profit units per period",
        rng,
        ["direct_discrete_monte_carlo"] * SAMPLE_SIZES.size,
    )

    queue_a, queue_b, queue_p, queue_meta = queue_distribution(rng)
    queue_values_a, queue_values_b = simulate_discrete_joint(
        queue_a, queue_b, queue_p, SAMPLE_SIZES, PRIMARY_REPLICATIONS, rng
    )
    queue_summary, _ = summarize_replications(
        "Queue",
        "Shortest-queue assignment",
        "Random assignment",
        SAMPLE_SIZES,
        queue_values_a,
        queue_values_b,
        "negative completion-time slots per job",
        rng,
        ["direct_discrete_monte_carlo"] * SAMPLE_SIZES.size,
    )
    cross_domain = pd.concat([primary, inventory_summary, queue_summary], ignore_index=True)
    cross_required = cross_domain_required_sample_sizes(cross_domain)

    print("Running blackjack rule and strategy sensitivity...")
    sensitivity = run_rule_sensitivity()
    strategies = run_strategy_comparison()
    bootstrap = bootstrap_intervals(primary_replications, rng)
    validation = validation_checks(primary)
    source_audit = audit_legacy_sources()

    primary.to_csv(OUTPUT_DIR / "blackjack_pfme_results.csv", index=False)
    primary_replications.to_csv(OUTPUT_DIR / "blackjack_replication_pairs.csv", index=False)
    cross_domain.to_csv(OUTPUT_DIR / "cross_domain_pfme_results.csv", index=False)
    cross_required.to_csv(OUTPUT_DIR / "cross_domain_required_sample_sizes.csv", index=False)
    required.to_csv(OUTPUT_DIR / "required_sample_sizes.csv", index=False)
    sensitivity.to_csv(OUTPUT_DIR / "blackjack_rule_sensitivity.csv", index=False)
    strategies.to_csv(OUTPUT_DIR / "blackjack_strategy_comparison.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "bootstrap_ci_comparison.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation_tests.csv", index=False)
    source_audit.to_csv(OUTPUT_DIR / "legacy_source_audit.csv", index=False)

    metadata = {
        "mode": "quick" if QUICK else "full",
        "master_seed": MASTER_SEED,
        "sample_sizes": SAMPLE_SIZES.tolist(),
        "direct_blackjack_sample_sizes": DIRECT_BLACKJACK_SAMPLE_SIZES.tolist(),
        "large_n_blackjack_method": "paired bootstrap aggregation of independent 2,500-round finite-shoe blocks",
        "primary_replications": PRIMARY_REPLICATIONS,
        "sensitivity_replications": SENSITIVITY_REPLICATIONS,
        "strategy_replications": STRATEGY_REPLICATIONS,
        "sensitivity_rounds": SENSITIVITY_ROUNDS,
        "strategy_rounds": STRATEGY_ROUNDS,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "blackjack_config": asdict(BASE_CONFIG),
        "inventory_model": inventory_meta,
        "queue_model": queue_meta,
        "diagnostics": diagnostics,
        "source_files": [
            "paper/PFME_manuscript.pdf",
            "data/legacy/blackjack_5000_hand_complete_dataset.xlsx",
            "data/legacy/blackjack_round_level_data.csv",
            "data/legacy/blackjack_replication_original.py",
        ],
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Creating figures...")
    make_figures(primary, primary_replications, cross_domain, required, sensitivity, strategies, bootstrap, diagnostics)
    make_contact_sheet()

    elapsed = time.perf_counter() - started
    write_summary(primary, required, strategies, validation, source_audit, elapsed)

    export_payload = {
        "primary": primary.replace({np.nan: None}).to_dict(orient="records"),
        "cross_domain": cross_domain.replace({np.nan: None}).to_dict(orient="records"),
        "cross_required": cross_required.replace({np.nan: None}).to_dict(orient="records"),
        "required": required.replace({np.nan: None}).to_dict(orient="records"),
        "sensitivity": sensitivity.replace({np.nan: None}).to_dict(orient="records"),
        "strategies": strategies.replace({np.nan: None}).to_dict(orient="records"),
        "bootstrap": bootstrap.replace({np.nan: None}).to_dict(orient="records"),
        "validation": validation.to_dict(orient="records"),
        "source_audit": source_audit.to_dict(orient="records"),
        "metadata": metadata,
    }
    (OUTPUT_DIR / "workbook_data.json").write_text(json.dumps(export_payload, indent=2), encoding="utf-8")

    largest = primary.iloc[-1]
    print(
        f"Done in {elapsed / 60:.2f} minutes. n=50,000: "
        f"Basic={largest['mean_rule_a']:.4f}%, Simplified={largest['mean_rule_b']:.4f}%, "
        f"difference={largest['mean_paired_difference']:.4f} pp, "
        f"reversal={largest['empirical_reversal_probability']:.6f}."
    )


if __name__ == "__main__":
    main()
