# Reproducibility notes

## Main blackjack experiment

The main script simulates matched strategy pairs under the same shuffled shoe.
The baseline rules use six decks, 75% penetration, dealer standing on soft 17,
3:2 blackjack payout, double after split, resplitting to four hands, no resplit
aces, and no late surrender.

Direct finite-shoe Monte Carlo is used through 2,500 rounds. For larger sample
sizes, each replication averages independent matched 2,500-round blocks. This
preserves the paired block distribution but is not an uninterrupted 50,000-round
path. The `estimation_mode` column makes this distinction explicit.

## Exact benchmarks

The inventory benchmark uses Poisson demand with mean 20 and compares order
quantities 21 and 20. Its exact paired difference is -5 for demand at most 20
and +9 for demand at least 21.

The Bernoulli benchmark uses an easy/hard environment with equal probability.
Conditional success probabilities are `(0.80, 0.40)` for Rule A and
`(0.70, 0.40)` for Rule B. Outcomes are conditionally independent given the
environment, producing paired difference probabilities 0.19, 0.57, and 0.24
for -1, 0, and +1.

## Environment variables

- `PFME_QUICK=1` selects a reduced smoke-test run.
- `PFME_OUTPUT_DIR=/path` changes the main script's generated-output location.
- `PFME_LEGACY_ROUND_PATH=/path/file.csv` changes the legacy round-level audit
  source.

## Archived versus regenerated results

The versioned CSV files under `results/tables/` are the archived full-run
outputs. A new main run writes to `results/generated/` by default so it does not
silently overwrite the archived evidence.

