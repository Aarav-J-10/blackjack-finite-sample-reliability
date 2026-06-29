A reproducible Python simulation comparing blackjack basic strategy with a simplified deterministic player strategy under finite-shoe casino rules.

The project simulates six-deck blackjack gameplay, records round-level and hand-level outcomes, compares strategy performance, and generates visual summaries using pandas and matplotlib.

Overview

This project compares two blackjack strategies:

Basic StrategyA multi-deck blackjack strategy using the player's hand, dealer up-card, doubling options, and splitting options.

Simplified StrategyA deterministic novice-style strategy that ignores the dealer's up-card, never doubles, never splits, and only uses simple hit/stand thresholds.

The simulation is designed to study how strategy quality appears over a finite number of rounds, especially when short-run variance can temporarily make a weaker strategy look better.

Game Rules

The simulator uses the following rules:

Six decks

Dealer stands on soft 17

Blackjack pays 3:2

Flat initial wager of one unit per round

Double after splitting allowed

Maximum of three splits

Split aces receive one additional card each

No surrender

Reshuffle at approximately 75% penetration

Features

Reproducible simulation using a fixed random seed

Round-level and resolved-hand-level data collection

Strategy-level return summaries

Block-level performance summaries

Confidence intervals for mean returns

Action and outcome frequency tracking

Exported CSV files

Generated plots for strategy comparison

Outputs

Running the script creates an output folder containing:

round_level_data.csv

hand_level_data.csv

block_level_summary.csv

strategy_summary.csv

Plot images comparing strategy performance

The exact filenames may depend on the final script version.

Requirements

Install the required Python packages:

pip install -r requirements.txt

Main dependencies:

numpy
pandas
matplotlib

How to Run

Clone the repository:

git clone https://github.com/your-username/blackjack-strategy-comparison.git
cd blackjack-strategy-comparison

Run the simulation:

python blackjack_strategy_comparison.py

The script will generate CSV outputs and plots in the project output folder.

Project Structure

blackjack-strategy-comparison/
├── blackjack_strategy_comparison.py
├── requirements.txt
├── README.md
└── outputs/

Reproducibility

The simulation uses a fixed seed:

SEED = 20260622

Changing the seed will generate a different simulated sequence of blackjack outcomes.

Interpretation

The results should not be interpreted as gambling advice. The purpose of the project is to compare strategy behavior under finite-sample simulation, not to suggest that blackjack can be made profitable.

Even a stronger strategy can underperform over short runs because blackjack outcomes are highly variable. The main value of this project is showing how repeated simulation and statistical summaries provide a clearer comparison than one cumulative-return path.# blackjack-finite-sample-reliability
This project simulates six-deck blackjack gameplay to compare basic strategy against a simplified player strategy, tracking returns, outcomes, actions, block-level variability, and visual summaries.

Possible Extensions

- Future improvements could include:

- Running thousands of independent replications

- Comparing more strategy variants

- Adding wager-normalized return analysis

- Testing different blackjack rule sets

- Adding card counting strategies

- Creating an interactive dashboard

- Adding automated unit tests for hand valuation and game rules
