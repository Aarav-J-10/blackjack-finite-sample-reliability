"""Generate exact inventory and Bernoulli benchmark tables and figures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pfme.exact_benchmarks import (  # noqa: E402
    BernoulliParameters,
    bernoulli_difference_probabilities,
    bernoulli_risk_table,
    inventory_moments,
    inventory_risk_table,
)


SAMPLE_SIZES = np.array([25, 50, 96, 100, 191, 250, 500, 1_000], dtype=int)
BERNOULLI_SIZES = np.array([25, 50, 100, 250, 500, 1_000], dtype=int)


def save_inventory_figure(table, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.semilogy(
        table["sample_size"],
        table["exact_strict_reversal_probability"],
        "o-",
        label="Exact paired",
    )
    ax.semilogy(
        table["sample_size"],
        table["normal_reversal_probability"],
        "s--",
        label="Normal approximation",
    )
    ax.axhline(0.05, color="#64748B", linestyle=":", linewidth=1.2, label="5% target")
    ax.axhline(0.01, color="#94A3B8", linestyle=":", linewidth=1.2, label="1% target")
    ax.set_xlabel("Paired demand observations")
    ax.set_ylabel("Strict reversal probability")
    ax.set_title("Inventory-control exact and normal reversal risk")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_bernoulli_figure(table, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = table["sample_size"]
    ax.semilogy(x, table["paired_exact_strict_reversal_probability"], "o-", label="Paired exact")
    ax.semilogy(x, table["paired_monte_carlo_strict_reversal_probability"], "x", markersize=8, label="Paired Monte Carlo")
    ax.semilogy(x, table["paired_normal_reversal_probability"], "s--", label="Paired normal")
    ax.semilogy(x, table["unpaired_exact_strict_reversal_probability"], "^-", label="Unpaired exact")
    ax.set_xlabel("Observations per rule")
    ax.set_ylabel("Strict reversal probability")
    ax.set_title("Controlled Bernoulli exact benchmark")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    table_dir = REPO_ROOT / "results" / "tables"
    figure_dir = REPO_ROOT / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    inventory = inventory_risk_table(SAMPLE_SIZES)
    bernoulli = bernoulli_risk_table(BERNOULLI_SIZES)
    inventory.to_csv(table_dir / "inventory_exact_risk.csv", index=False)
    bernoulli.to_csv(table_dir / "bernoulli_exact_risk.csv", index=False)
    save_inventory_figure(inventory, figure_dir / "11_inventory_exact_validation.png")
    save_bernoulli_figure(bernoulli, figure_dir / "12_bernoulli_exact.png")

    moments = inventory_moments()
    probabilities = bernoulli_difference_probabilities(BernoulliParameters())
    print(
        "Inventory advantage: "
        f"{moments['mean_difference']:.6f}; variance reduction: "
        f"{moments['variance_reduction']:.2%}"
    )
    print(f"Bernoulli difference probabilities: {probabilities}")
    print(f"Wrote {len(inventory)} inventory rows and {len(bernoulli)} Bernoulli rows.")


if __name__ == "__main__":
    main()
