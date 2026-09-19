# Finite-Sample Misranking in Stochastic Decision Systems

This repository contains the paper, corrected simulation code, exact benchmark
calculations, processed results, figures, validation tests, and legacy-source
audit for the Paired Finite-Sample Misranking Estimator (PFME).

PFME studies a practical question: when one stochastic decision rule is better
in expectation, how often can a finite experiment rank it below an inferior
rule, and how much data is needed before that risk becomes acceptably small?

## Main results

- Corrected six-deck blackjack mean return: approximately **-0.50%** for full
  basic strategy and **-6.08%** for the simplified strategy.
- The simplified strategy appeared superior in **33.29%** of paired experiments
  at 100 rounds, **9.22%** at 1,000 rounds, and **2.00%** at 2,500 rounds.
- Paired blackjack evaluation reduced comparison variance by a median of
  approximately **19.6%**.
- Normal planning required **3,208 paired rounds** versus **3,974 unpaired
  rounds** for a 1% reversal-risk target.
- Common-demand pairing reduced inventory comparison variance by approximately
  **95.8%**.
- The controlled Bernoulli benchmark provides exact finite-sample probabilities
  against which Monte Carlo and normal approximations can be checked.

## Important correction to the legacy simulator

The earlier blackjack implementation constructed a nominal six-deck shoe from
13 rank entries multiplied by six, producing only 78 cards instead of 312. Its
cut-card threshold was also 78, so it reshuffled before every recorded round.
The corrected implementation explicitly constructs 52 cards per deck and uses
a 312-card six-deck shoe. All corrected results supersede the legacy estimates.

## Repository map

```text
.
|-- data/legacy/            Original supplied data and legacy code (audit only)
|-- docs/                   Methods, reproducibility, and known-issues notes
|-- figures/                Publication figures and contact sheet
|-- paper/                  Current manuscript PDF and LaTeX result section
|-- results/
|   |-- tables/             Computed CSV outputs
|   `-- workbook/           Formatted results workbook
|-- scripts/
|   |-- run_pfme_analysis.py
|   `-- run_exact_benchmarks.py
|-- src/pfme/               Reusable exact-distribution calculations
`-- tests/                  Automated correctness tests
```

## Reproduce the analysis

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the fast smoke-test configuration:

```bash
PFME_QUICK=1 python scripts/run_pfme_analysis.py
```

On PowerShell, use:

```powershell
$env:PFME_QUICK = "1"
python scripts/run_pfme_analysis.py
```

Run the full (R=10{,}000) experiment:

```bash
python scripts/run_pfme_analysis.py
```

The full run is computationally intensive. Blackjack sample sizes through 2,500
are simulated directly. Larger values are produced by aggregating independently
resampled matched 2,500-round blocks and are labeled accordingly in the outputs.

Generate the analytical inventory and Bernoulli tables and figures:

```bash
python scripts/run_exact_benchmarks.py
```

Run the automated tests:

```bash
python -m pytest
```

## Reproducibility controls

- Main simulation seed: `20260904`.
- Exact-benchmark Monte Carlo seed: `20260919`.
- Every result row records its sample size, replication count, and estimation
  mode.
- `results/run_metadata.json` records the configuration used for the archived
  full run.
- `results/tables/validation_tests.csv` contains the 14 simulator checks that
  passed before the reported run.

## Definition of reversal

The formal definition used by the corrected code is a **strict reversal**:

```text
sample mean of Rule A - sample mean of Rule B < 0
```

Ties are reported separately. Because discrete benchmarks can assign meaningful
probability to a tie, exact benchmark outputs also contain the nonpositive
probability (`reversal + tie`). See `docs/KNOWN_ISSUES.md` for the manuscript
table that should be updated before publication.

## Data provenance

`data/legacy/` preserves the originally supplied files so the audit can be
reproduced. Those files are not treated as valid outputs of the corrected
simulator. Corrected processed outputs are under `results/tables/`.

## Citation and license

Citation metadata is supplied in `CITATION.cff`. No public-use license has yet
been selected; see `LICENSE-NOTE.md` before publishing the repository.

