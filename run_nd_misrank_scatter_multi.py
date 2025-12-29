#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 12:49:56 2025

@author: srini
"""

# run_nd_misrank_scatter_multi.py
"""
Combined ND–vs–p(mis-rank) scatter, averaged over seeds/pairs.

Assumes you have already run:
    - run_nd_multi.py        -> results/nd_multi_summary.csv
    - run_misrank_multi.py   -> results/misrank_multi_summary.csv

This script:
    - Reads both summary CSVs.
    - Merges them on (metric, sampler, frac).
    - For each sampling fraction, creates a scatter plot:
        x = nd_mean
        y = p_misrank_mean
      with:
        - color = sampler
        - marker = metric
        - optional error bars:
            xerr = nd_std
            yerr = p_misrank_std
    - Saves figures into results/ as:
        nd_misrank_scatter_frac_0p10.png   (for frac = 0.10, etc.)
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------
# Global plotting style
# ---------------------------------------------------------------------

SAMPLER_COLORS = {
    "random_node":     "tab:blue",
    "snowball":        "tab:orange",
    "degree_weighted": "tab:green",
    "edge_uniform":    "tab:red",
    # add more here if needed
}

METRIC_MARKERS = {
    "clustering":    "o",
    "assortativity": "s",
    "avg_path":      "D",
    "modularity":    "^",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_output_dir(dirname: str = "results") -> Path:
    out_dir = Path(dirname)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def frac_to_str(frac: float) -> str:
    """Turn 0.1 -> '0p10' etc. for filenames."""
    return str(frac).replace(".", "p")


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_nd_vs_misrank_multi(df_nd_sum: pd.DataFrame,
                             df_mis_sum: pd.DataFrame,
                             output_dir: Path) -> None:
    """
    Create ND vs p(mis-rank) scatter plots for each sampling fraction,
    using mean values over graphs/pairs and including error bars.
    """
    # Merge summaries on (metric, sampler, frac)
    merged = df_nd_sum.merge(
        df_mis_sum,
        on=["metric", "sampler", "frac"],
        how="inner",
        suffixes=("_nd", "_mis"),
    )

    if merged.empty:
        print("No overlapping rows between ND and misranking summaries.")
        return

    frac_values = sorted(merged["frac"].unique())

    for frac in frac_values:
        sub = merged[merged["frac"] == frac]
        if sub.empty:
            continue

        fig, ax = plt.subplots()

        for _, row in sub.iterrows():
            metric = row["metric"]
            sampler = row["sampler"]

            nd_mean = row["nd_mean"]
            nd_std  = row.get("nd_std", 0.0)
            pm_mean = row["p_misrank_mean"]
            pm_std  = row.get("p_misrank_std", 0.0)

            color  = SAMPLER_COLORS.get(sampler, None)
            marker = METRIC_MARKERS.get(metric, "o")

            # Error bars + point
            ax.errorbar(
                nd_mean,
                pm_mean,
                xerr=nd_std if pd.notna(nd_std) else 0.0,
                yerr=pm_std if pd.notna(pm_std) else 0.0,
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1,
                capsize=3,
                markersize=6,
                alpha=0.9,
            )

        ax.set_title(f"ND vs p(mis-rank) (mean ± std), frac = {frac:.2f}")
        ax.set_xlabel("ND (mean across graphs)")
        ax.set_ylabel("p(mis-rank) (mean across pairs)")
        ax.grid(True, alpha=0.3)
        #ax.set_ylim(-0.02, 1.02)

        # Legends: samplers (colors) and metrics (markers)
        sampler_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                linestyle="",
                label=sampler,
            )
            for sampler, color in SAMPLER_COLORS.items()
            if sampler in sub["sampler"].unique()
        ]

        metric_handles = [
            Line2D(
                [0],
                [0],
                marker=marker,
                color="black",
                linestyle="",
                label=metric,
            )
            for metric, marker in METRIC_MARKERS.items()
            if metric in sub["metric"].unique()
        ]

        legend1 = ax.legend(
            handles=sampler_handles,
            title="Sampler (color)",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
        ax.add_artist(legend1)

        ax.legend(
            handles=metric_handles,
            title="Metric (marker)",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            borderaxespad=0.0,
        )

        fig.tight_layout()
        fname = output_dir / f"nd_misrank_scatter_frac_{frac_to_str(frac)}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {fname}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    results_dir = ensure_output_dir("results")

    nd_path  = results_dir / "nd_multi_summary.csv"
    mis_path = results_dir / "misrank_multi_summary.csv"

    if not nd_path.exists():
        raise FileNotFoundError(f"ND summary not found: {nd_path}")
    if not mis_path.exists():
        raise FileNotFoundError(f"Mis-ranking summary not found: {mis_path}")

    df_nd_sum  = pd.read_csv(nd_path)
    df_mis_sum = pd.read_csv(mis_path)

    # Basic sanity check: required columns
    for col in ["metric", "sampler", "frac", "nd_mean"]:
        if col not in df_nd_sum.columns:
            raise ValueError(f"Column '{col}' missing in {nd_path}")
    for col in ["metric", "sampler", "frac", "p_misrank_mean"]:
        if col not in df_mis_sum.columns:
            raise ValueError(f"Column '{col}' missing in {mis_path}")

    plot_nd_vs_misrank_multi(df_nd_sum, df_mis_sum, results_dir)
    print(f"Combined ND–p(mis-rank) scatter plots saved in {results_dir.resolve()}")


if __name__ == "__main__":
    main()