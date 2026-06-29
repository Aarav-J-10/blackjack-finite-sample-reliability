import math
import random
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SEED = 20260622

TOTAL_ROUNDS = 5000
BLOCK_SIZE = 100

NUM_DECKS = 6
PENETRATION = 0.75

BLACKJACK_PAYOUT = 1.5

MAX_SPLITS = 3
ALLOW_DOUBLE_AFTER_SPLIT = True
RESPLIT_ACES = False

OUTPUT_FOLDER = Path("blackjack_5000_hand_study")


rng = random.Random(SEED)


RANKS = [
    "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "J", "Q", "K", "A"
]

VALUES = [
    2, 3, 4, 5, 6, 7, 8, 9,
    10, 10, 10, 10, 11
]

BASE_DECK = list(zip(RANKS, VALUES))


class Shoe:
    def __init__(self, num_decks=6, penetration=0.75, rng=None):
        self.num_decks = num_decks
        self.penetration = penetration
        self.rng = rng or random.Random()

        self.initial_size = 52 * num_decks


        self.cut_remaining = int(
            self.initial_size * (1 - penetration)
        )

        self.cards = []
        self.shuffle_count = 0

        self.reshuffle()

    def reshuffle(self):
        self.cards = BASE_DECK * self.num_decks
        self.rng.shuffle(self.cards)
        self.shuffle_count += 1

    def needs_shuffle(self):
        return len(self.cards) <= self.cut_remaining

    def deal(self):
        if not self.cards:
            self.reshuffle()

        return self.cards.pop()


def hand_value(cards):
    total = sum(value for rank, value in cards)
    aces_counted_as_eleven = sum(
        1 for rank, value in cards if value == 11
    )

    while total > 21 and aces_counted_as_eleven > 0:
        total -= 10
        aces_counted_as_eleven -= 1

    soft = aces_counted_as_eleven > 0

    return total, soft


def is_blackjack(cards):
    total, _ = hand_value(cards)
    return len(cards) == 2 and total == 21


def card_label(card):
    return card[0]


def cards_label(cards):
    return " ".join(card_label(card) for card in cards)


def basic_strategy(cards, dealer_upcard, can_double, can_split):
    total, soft = hand_value(cards)
    dealer = dealer_upcard[1]


    if (
        can_split
        and len(cards) == 2
        and cards[0][1] == cards[1][1]
    ):
        pair_value = cards[0][1]


        if pair_value == 11:
            return "split"


        if pair_value == 10:
            return "stand"


        if pair_value == 9:
            if dealer in [2, 3, 4, 5, 6, 8, 9]:
                return "split"
            return "stand"


        if pair_value == 8:
            return "split"


        if pair_value == 7:
            if dealer in [2, 3, 4, 5, 6, 7]:
                return "split"
            return "hit"


        if pair_value == 6:
            if dealer in [2, 3, 4, 5, 6]:
                return "split"
            return "hit"


        if pair_value == 5:
            pass


        elif pair_value == 4:
            if dealer in [5, 6]:
                return "split"
            return "hit"


        elif pair_value in [2, 3]:
            if dealer in [2, 3, 4, 5, 6, 7]:
                return "split"
            return "hit"


    if soft:
        if total in [13, 14]:
            if can_double and dealer in [5, 6]:
                return "double"
            return "hit"


        if total in [15, 16]:
            if can_double and dealer in [4, 5, 6]:
                return "double"
            return "hit"


        if total == 17:
            if can_double and dealer in [3, 4, 5, 6]:
                return "double"
            return "hit"


        if total == 18:
            if can_double and dealer in [3, 4, 5, 6]:
                return "double"

            if dealer in [2, 7, 8]:
                return "stand"

            return "hit"


        if total == 19:
            if can_double and dealer == 6:
                return "double"
            return "stand"


        if total >= 20:
            return "stand"

        return "hit"


    if total <= 8:
        return "hit"

    if total == 9:
        if can_double and dealer in [3, 4, 5, 6]:
            return "double"
        return "hit"

    if total == 10:
        if can_double and dealer in [2, 3, 4, 5, 6, 7, 8, 9]:
            return "double"
        return "hit"

    if total == 11:
        if can_double and dealer in [
            2, 3, 4, 5, 6, 7, 8, 9, 10
        ]:
            return "double"
        return "hit"

    if total == 12:
        if dealer in [4, 5, 6]:
            return "stand"
        return "hit"

    if 13 <= total <= 16:
        if dealer in [2, 3, 4, 5, 6]:
            return "stand"
        return "hit"

    return "stand"


def simplified_strategy(
    cards,
    dealer_upcard,
    can_double,
    can_split
):
    total, soft = hand_value(cards)

    if soft:
        if total >= 18:
            return "stand"
        return "hit"

    if total >= 12:
        return "stand"

    return "hit"


@dataclass
class PlayerHand:
    cards: list
    bet: float = 1.0
    from_split: bool = False
    split_aces: bool = False
    hand_id: int = 1
    parent_hand_id: int = 0


def dealer_play(dealer_cards, shoe):
    actions = []

    while True:
        total, soft = hand_value(dealer_cards)

        if total >= 17:
            break

        dealer_cards.append(shoe.deal())
        actions.append("hit")

    return dealer_cards, actions


def play_round(
    round_number,
    block_number,
    strategy_name,
    shoe
):
    if shoe.needs_shuffle():
        shoe.reshuffle()

    shoe_number = shoe.shuffle_count
    cards_remaining_before = len(shoe.cards)


    player_card_1 = shoe.deal()
    dealer_card_1 = shoe.deal()
    player_card_2 = shoe.deal()
    dealer_card_2 = shoe.deal()

    player_initial = [
        player_card_1,
        player_card_2
    ]

    dealer_cards = [
        dealer_card_1,
        dealer_card_2
    ]

    dealer_upcard = dealer_card_1
    dealer_holecard = dealer_card_2

    player_blackjack = is_blackjack(player_initial)
    dealer_blackjack = is_blackjack(dealer_cards)

    resolved_hand_rows = []


    if player_blackjack or dealer_blackjack:
        if player_blackjack and dealer_blackjack:
            net_return = 0.0
            outcome = "push_blackjack"

        elif player_blackjack:
            net_return = BLACKJACK_PAYOUT
            outcome = "player_blackjack"

        else:
            net_return = -1.0
            outcome = "dealer_blackjack"

        player_total, _ = hand_value(player_initial)
        dealer_total, _ = hand_value(dealer_cards)

        resolved_hand_rows.append({
            "round_no": round_number,
            "block_no": block_number,
            "strategy": strategy_name,
            "shoe_no": shoe_number,

            "hand_id": 1,
            "parent_hand_id": 0,
            "from_split": False,
            "split_aces": False,

            "initial_player_cards": cards_label(player_initial),
            "final_player_cards": cards_label(player_initial),

            "dealer_upcard": card_label(dealer_upcard),
            "dealer_final_cards": cards_label(dealer_cards),

            "player_total": player_total,
            "dealer_total": dealer_total,

            "actions": "blackjack_check",
            "bet_units": 1.0,

            "outcome": outcome,
            "net_return": net_return
        })

        round_row = {
            "round_no": round_number,
            "block_no": block_number,
            "strategy": strategy_name,
            "shoe_no": shoe_number,

            "cards_remaining_before": cards_remaining_before,

            "initial_player_cards": cards_label(player_initial),
            "dealer_upcard": card_label(dealer_upcard),
            "dealer_holecard": card_label(dealer_holecard),
            "dealer_final_cards": cards_label(dealer_cards),
            "dealer_total": dealer_total,

            "num_resolved_hands": 1,
            "num_splits": 0,
            "num_doubles": 0,

            "round_outcome": outcome,
            "net_return": net_return
        }

        return resolved_hand_rows, round_row


    if strategy_name == "Basic":
        strategy_function = basic_strategy
    else:
        strategy_function = simplified_strategy

    pending_hands = deque([
        PlayerHand(
            cards=player_initial.copy(),
            hand_id=1
        )
    ])

    completed_hands = []

    split_count = 0
    double_count = 0
    next_hand_id = 2


    while pending_hands:
        player_hand = pending_hands.popleft()

        initial_hand_label = cards_label(player_hand.cards)
        actions = []


        if player_hand.split_aces:
            actions.append("stand_after_split_aces")

        else:
            while True:
                player_total, soft = hand_value(
                    player_hand.cards
                )

                if player_total >= 21:
                    break

                can_double = (
                    len(player_hand.cards) == 2
                    and (
                        ALLOW_DOUBLE_AFTER_SPLIT
                        or not player_hand.from_split
                    )
                )

                can_split = (
                    strategy_name == "Basic"
                    and len(player_hand.cards) == 2
                    and player_hand.cards[0][1]
                    == player_hand.cards[1][1]
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
                    can_split
                )

                actions.append(action)


                if action == "stand":
                    break


                if action == "hit":
                    player_hand.cards.append(
                        shoe.deal()
                    )
                    continue


                if action == "double" and can_double:
                    player_hand.bet *= 2
                    double_count += 1

                    player_hand.cards.append(
                        shoe.deal()
                    )

                    break


                if action == "split" and can_split:
                    split_count += 1

                    first_card = player_hand.cards[0]
                    second_card = player_hand.cards[1]

                    splitting_aces = (
                        first_card[1] == 11
                    )


                    first_split_cards = [
                        first_card,
                        shoe.deal()
                    ]

                    second_split_cards = [
                        second_card,
                        shoe.deal()
                    ]

                    root_parent = (
                        player_hand.parent_hand_id
                        or player_hand.hand_id
                    )

                    first_split_hand = PlayerHand(
                        cards=first_split_cards,
                        bet=player_hand.bet,
                        from_split=True,
                        split_aces=splitting_aces,
                        hand_id=player_hand.hand_id,
                        parent_hand_id=root_parent
                    )

                    second_split_hand = PlayerHand(
                        cards=second_split_cards,
                        bet=player_hand.bet,
                        from_split=True,
                        split_aces=splitting_aces,
                        hand_id=next_hand_id,
                        parent_hand_id=root_parent
                    )

                    next_hand_id += 1


                    pending_hands.appendleft(
                        second_split_hand
                    )
                    pending_hands.appendleft(
                        first_split_hand
                    )

                    player_hand = None
                    break


                player_hand.cards.append(
                    shoe.deal()
                )

            if player_hand is None:
                continue

        completed_hands.append(
            (
                player_hand,
                initial_hand_label,
                actions
            )
        )


    at_least_one_live_hand = any(
        hand_value(hand.cards)[0] <= 21
        for hand, initial_label, actions
        in completed_hands
    )

    dealer_actions = []

    if at_least_one_live_hand:
        dealer_cards, dealer_actions = dealer_play(
            dealer_cards,
            shoe
        )

    dealer_total, dealer_soft = hand_value(
        dealer_cards
    )

    dealer_bust = dealer_total > 21


    round_net_return = 0.0

    for (
        player_hand,
        initial_hand_label,
        actions
    ) in completed_hands:
        player_total, player_soft = hand_value(
            player_hand.cards
        )

        if player_total > 21:
            outcome = "loss_bust"
            hand_net_return = -player_hand.bet

        elif dealer_bust:
            outcome = "win_dealer_bust"
            hand_net_return = player_hand.bet

        elif player_total > dealer_total:
            outcome = "win"
            hand_net_return = player_hand.bet

        elif player_total < dealer_total:
            outcome = "loss"
            hand_net_return = -player_hand.bet

        else:
            outcome = "push"
            hand_net_return = 0.0

        round_net_return += hand_net_return

        resolved_hand_rows.append({
            "round_no": round_number,
            "block_no": block_number,
            "strategy": strategy_name,
            "shoe_no": shoe_number,

            "hand_id": player_hand.hand_id,
            "parent_hand_id": player_hand.parent_hand_id,
            "from_split": player_hand.from_split,
            "split_aces": player_hand.split_aces,

            "initial_player_cards": initial_hand_label,
            "final_player_cards": cards_label(
                player_hand.cards
            ),

            "dealer_upcard": card_label(
                dealer_upcard
            ),

            "dealer_final_cards": cards_label(
                dealer_cards
            ),

            "player_total": player_total,
            "dealer_total": dealer_total,

            "actions": " > ".join(actions),
            "bet_units": player_hand.bet,

            "outcome": outcome,
            "net_return": hand_net_return
        })


    if round_net_return > 0:
        round_outcome = "net_win"

    elif round_net_return < 0:
        round_outcome = "net_loss"

    else:
        round_outcome = "net_push"

    round_row = {
        "round_no": round_number,
        "block_no": block_number,
        "strategy": strategy_name,
        "shoe_no": shoe_number,

        "cards_remaining_before": cards_remaining_before,

        "initial_player_cards": cards_label(
            player_initial
        ),

        "dealer_upcard": card_label(
            dealer_upcard
        ),

        "dealer_holecard": card_label(
            dealer_holecard
        ),

        "dealer_final_cards": cards_label(
            dealer_cards
        ),

        "dealer_total": dealer_total,

        "num_resolved_hands": len(
            completed_hands
        ),

        "num_splits": split_count,
        "num_doubles": double_count,

        "round_outcome": round_outcome,
        "net_return": round_net_return
    }

    return resolved_hand_rows, round_row


shoe = Shoe(
    num_decks=NUM_DECKS,
    penetration=PENETRATION,
    rng=rng
)

all_hand_rows = []
all_round_rows = []

for round_number in range(
    1,
    TOTAL_ROUNDS + 1
):
    block_number = (
        (round_number - 1) // BLOCK_SIZE
    ) + 1


    if block_number % 2 == 1:
        strategy_name = "Basic"
    else:
        strategy_name = "Simplified"

    hand_rows, round_row = play_round(
        round_number=round_number,
        block_number=block_number,
        strategy_name=strategy_name,
        shoe=shoe
    )

    all_hand_rows.extend(hand_rows)
    all_round_rows.append(round_row)


rounds_df = pd.DataFrame(all_round_rows)
hands_df = pd.DataFrame(all_hand_rows)


rounds_df["strategy_round_number"] = (
    rounds_df.groupby("strategy")
    .cumcount()
    + 1
)

rounds_df["running_return"] = (
    rounds_df.groupby("strategy")["net_return"]
    .cumsum()
)


blocks_df = (
    rounds_df
    .groupby(
        ["block_no", "strategy"],
        as_index=False
    )
    .agg(
        rounds=("round_no", "count"),
        total_return=("net_return", "sum"),
        mean_return_per_round=(
            "net_return",
            "mean"
        ),
        sd_return_per_round=(
            "net_return",
            "std"
        ),
        wins=(
            "round_outcome",
            lambda values:
            (values == "net_win").sum()
        ),
        losses=(
            "round_outcome",
            lambda values:
            (values == "net_loss").sum()
        ),
        pushes=(
            "round_outcome",
            lambda values:
            (values == "net_push").sum()
        ),
        splits=("num_splits", "sum"),
        doubles=("num_doubles", "sum")
    )
)

blocks_df["win_rate"] = (
    blocks_df["wins"]
    / blocks_df["rounds"]
)

blocks_df["loss_rate"] = (
    blocks_df["losses"]
    / blocks_df["rounds"]
)


summary_df = (
    rounds_df
    .groupby(
        "strategy",
        as_index=False
    )
    .agg(
        rounds=("round_no", "count"),
        total_return=("net_return", "sum"),
        mean_return_per_round=(
            "net_return",
            "mean"
        ),
        median_return_per_round=(
            "net_return",
            "median"
        ),
        sd_return_per_round=(
            "net_return",
            "std"
        ),
        wins=(
            "round_outcome",
            lambda values:
            (values == "net_win").sum()
        ),
        losses=(
            "round_outcome",
            lambda values:
            (values == "net_loss").sum()
        ),
        pushes=(
            "round_outcome",
            lambda values:
            (values == "net_push").sum()
        ),
        splits=("num_splits", "sum"),
        doubles=("num_doubles", "sum")
    )
)

summary_df["win_rate"] = (
    summary_df["wins"]
    / summary_df["rounds"]
)

summary_df["loss_rate"] = (
    summary_df["losses"]
    / summary_df["rounds"]
)

summary_df["push_rate"] = (
    summary_df["pushes"]
    / summary_df["rounds"]
)

summary_df["return_percentage"] = (
    100
    * summary_df["total_return"]
    / summary_df["rounds"]
)


def mean_confidence_interval(series, confidence=0.95):
    values = pd.Series(series).dropna()

    n = len(values)
    mean = values.mean()
    sd = values.std(ddof=1)
    standard_error = sd / math.sqrt(n)

    z_critical = 1.96

    lower = mean - z_critical * standard_error
    upper = mean + z_critical * standard_error

    return pd.Series({
        "n": n,
        "mean_return": mean,
        "standard_deviation": sd,
        "standard_error": standard_error,
        "ci_95_lower": lower,
        "ci_95_upper": upper
    })


confidence_intervals_df = (
    rounds_df
    .groupby("strategy")["net_return"]
    .apply(mean_confidence_interval)
    .unstack()
    .reset_index()
)


outcome_distribution_df = (
    rounds_df
    .groupby(
        ["strategy", "round_outcome"]
    )
    .size()
    .reset_index(name="count")
)

outcome_distribution_df["proportion"] = (
    outcome_distribution_df["count"]
    / outcome_distribution_df
      .groupby("strategy")["count"]
      .transform("sum")
)


action_rows = []

for _, row in hands_df.iterrows():
    action_sequence = str(row["actions"])

    for action in action_sequence.split(" > "):
        action_rows.append({
            "strategy": row["strategy"],
            "round_no": row["round_no"],
            "hand_id": row["hand_id"],
            "action": action
        })

actions_df = pd.DataFrame(action_rows)

action_distribution_df = (
    actions_df
    .groupby(
        ["strategy", "action"]
    )
    .size()
    .reset_index(name="count")
)


assert len(rounds_df) == TOTAL_ROUNDS

assert (
    rounds_df["strategy"]
    .value_counts()["Basic"]
    == 2500
)

assert (
    rounds_df["strategy"]
    .value_counts()["Simplified"]
    == 2500
)

assert rounds_df["round_no"].is_unique

assert rounds_df["net_return"].notna().all()
assert hands_df["net_return"].notna().all()


hand_total = hands_df["net_return"].sum()
round_total = rounds_df["net_return"].sum()

assert math.isclose(
    hand_total,
    round_total,
    rel_tol=0,
    abs_tol=1e-9
)

print("All validation checks passed.")


print("\nEXPERIMENT SETTINGS")
print("-" * 60)

print(f"Random seed: {SEED}")
print(f"Total original rounds: {TOTAL_ROUNDS}")
print(f"Rounds per strategy: {TOTAL_ROUNDS // 2}")
print(f"Block size: {BLOCK_SIZE}")
print(f"Number of blocks: {TOTAL_ROUNDS // BLOCK_SIZE}")
print(f"Decks: {NUM_DECKS}")
print(f"Penetration: {PENETRATION:.0%}")
print(f"Shoes used: {shoe.shuffle_count}")

print("\nSTRATEGY SUMMARY")
print("-" * 60)

display(summary_df.round(5))

print("\n95% CONFIDENCE INTERVALS")
print("-" * 60)

display(confidence_intervals_df.round(5))

print("\nFIRST 20 ROUND-LEVEL RECORDS")
print("-" * 60)

display(rounds_df.head(20))

print("\nFIRST 20 RESOLVED-HAND RECORDS")
print("-" * 60)

display(hands_df.head(20))


OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


plt.figure(figsize=(11, 6))

for strategy_name, group in rounds_df.groupby(
    "strategy"
):
    plt.plot(
        group["strategy_round_number"],
        group["running_return"],
        label=strategy_name
    )

plt.axhline(
    y=0,
    linewidth=1
)

plt.xlabel("Round number within strategy")
plt.ylabel("Cumulative return in units")
plt.title(
    "Cumulative Blackjack Return by Strategy"
)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "01_cumulative_return.png",
    dpi=300
)

plt.show()


plt.figure(figsize=(11, 6))

basic_blocks = blocks_df[
    blocks_df["strategy"] == "Basic"
]

simplified_blocks = blocks_df[
    blocks_df["strategy"] == "Simplified"
]

plt.scatter(
    basic_blocks["block_no"],
    basic_blocks["total_return"],
    label="Basic"
)

plt.scatter(
    simplified_blocks["block_no"],
    simplified_blocks["total_return"],
    label="Simplified"
)

plt.axhline(
    y=0,
    linewidth=1
)

plt.xlabel("Block number")
plt.ylabel("Return in 100-round block")
plt.title(
    "Blackjack Return Across 100-Round Blocks"
)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "02_block_returns.png",
    dpi=300
)

plt.show()


plot_ci_df = confidence_intervals_df.copy()

lower_error = (
    plot_ci_df["mean_return"]
    - plot_ci_df["ci_95_lower"]
)

upper_error = (
    plot_ci_df["ci_95_upper"]
    - plot_ci_df["mean_return"]
)

plt.figure(figsize=(8, 6))

plt.errorbar(
    x=plot_ci_df["strategy"],
    y=plot_ci_df["mean_return"],
    yerr=[
        lower_error,
        upper_error
    ],
    fmt="o",
    capsize=6
)

plt.axhline(
    y=0,
    linewidth=1
)

plt.xlabel("Strategy")
plt.ylabel("Mean return per original round")
plt.title(
    "Mean Return and 95% Confidence Intervals"
)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "03_mean_return_confidence_intervals.png",
    dpi=300
)

plt.show()


outcome_pivot = (
    outcome_distribution_df
    .pivot(
        index="strategy",
        columns="round_outcome",
        values="proportion"
    )
    .fillna(0)
)

outcome_pivot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.xlabel("Strategy")
plt.ylabel("Proportion of original rounds")
plt.title(
    "Outcome Distribution by Strategy"
)
plt.xticks(rotation=0)
plt.legend(
    title="Round outcome"
)
plt.grid(
    axis="y",
    alpha=0.25
)
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "04_outcome_distribution.png",
    dpi=300
)

plt.show()


plt.figure(figsize=(10, 6))

for strategy_name, group in blocks_df.groupby(
    "strategy"
):
    plt.hist(
        group["total_return"],
        bins=10,
        alpha=0.5,
        label=strategy_name
    )

plt.axvline(
    x=0,
    linewidth=1
)

plt.xlabel("Return per 100-round block")
plt.ylabel("Frequency")
plt.title(
    "Distribution of 100-Round Block Returns"
)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "05_block_return_distribution.png",
    dpi=300
)

plt.show()


rounds_csv = OUTPUT_FOLDER / "blackjack_round_level_data.csv"
hands_csv = OUTPUT_FOLDER / "blackjack_resolved_hand_data.csv"
blocks_csv = OUTPUT_FOLDER / "blackjack_block_summary.csv"
summary_csv = OUTPUT_FOLDER / "blackjack_strategy_summary.csv"
ci_csv = OUTPUT_FOLDER / "blackjack_confidence_intervals.csv"
outcomes_csv = OUTPUT_FOLDER / "blackjack_outcome_distribution.csv"
actions_csv = OUTPUT_FOLDER / "blackjack_action_distribution.csv"

rounds_df.to_csv(
    rounds_csv,
    index=False
)

hands_df.to_csv(
    hands_csv,
    index=False
)

blocks_df.to_csv(
    blocks_csv,
    index=False
)

summary_df.to_csv(
    summary_csv,
    index=False
)

confidence_intervals_df.to_csv(
    ci_csv,
    index=False
)

outcome_distribution_df.to_csv(
    outcomes_csv,
    index=False
)

action_distribution_df.to_csv(
    actions_csv,
    index=False
)


excel_path = (
    OUTPUT_FOLDER
    / "blackjack_5000_hand_complete_dataset.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:
    rounds_df.to_excel(
        writer,
        sheet_name="Round Level Data",
        index=False
    )

    hands_df.to_excel(
        writer,
        sheet_name="Resolved Hands",
        index=False
    )

    blocks_df.to_excel(
        writer,
        sheet_name="Block Summary",
        index=False
    )

    summary_df.to_excel(
        writer,
        sheet_name="Strategy Summary",
        index=False
    )

    confidence_intervals_df.to_excel(
        writer,
        sheet_name="Confidence Intervals",
        index=False
    )

    outcome_distribution_df.to_excel(
        writer,
        sheet_name="Outcome Distribution",
        index=False
    )

    action_distribution_df.to_excel(
        writer,
        sheet_name="Action Distribution",
        index=False
    )


settings_df = pd.DataFrame({
    "parameter": [
        "random_seed",
        "total_rounds",
        "block_size",
        "number_of_blocks",
        "rounds_per_strategy",
        "number_of_decks",
        "penetration",
        "dealer_soft_17_rule",
        "blackjack_payout",
        "maximum_splits",
        "double_after_split",
        "resplit_aces",
        "surrender",
        "initial_bet_units"
    ],

    "value": [
        SEED,
        TOTAL_ROUNDS,
        BLOCK_SIZE,
        TOTAL_ROUNDS // BLOCK_SIZE,
        TOTAL_ROUNDS // 2,
        NUM_DECKS,
        PENETRATION,
        "stand",
        BLACKJACK_PAYOUT,
        MAX_SPLITS,
        ALLOW_DOUBLE_AFTER_SPLIT,
        RESPLIT_ACES,
        False,
        1
    ]
})

settings_df.to_csv(
    OUTPUT_FOLDER / "experiment_settings.csv",
    index=False
)


zip_base_name = "blackjack_5000_hand_study"

shutil.make_archive(
    zip_base_name,
    "zip",
    OUTPUT_FOLDER
)

zip_path = Path(
    f"{zip_base_name}.zip"
)

print("\nFILES CREATED")
print("-" * 60)

for file_path in sorted(
    OUTPUT_FOLDER.iterdir()
):
    print(file_path)

print(f"\nZIP archive: {zip_path}")


try:
    from google.colab import files

    files.download(
        str(zip_path)
    )

except ImportError:
    print(
        "\nThe script is not running in Google Colab. "
        "The files remain saved in the current directory."
    )
