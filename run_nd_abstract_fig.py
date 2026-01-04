#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 10:43:48 2026

@author: srini
"""

#!/usr/bin/env python

"""
run_nd_abstract_fig.py

Regenerate ND results for two metrics (clustering, avg_path) for use in the
one-page abstract, and produce a 2x4 panel figure:

Row 1: ND – clustering
Row 2: ND – avg_path
Cols : BA, ER, SBM, WS

Assumes the following local modules exist in the same folder:
    - sampling.py
    - metrics.py
    - nd_diagnostics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sampling import (
    sample_random_nodes,
    sample_snowball,
    sample_degree_weighted_nodes,
    sample_uniform_edges,
)
from metrics import (
    metric_global_clustering,
    metric_avg_shortest_path,
)
from nd_diagnostics import estimate_nd_grid_fast


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

GRAPH_TYPES = ["BA", "ER", "SBM", "WS"]
NODES = 800
TARGET_DEG = 10

N_GRAPHS_PER_TYPE = 5           # number of seeds per graph family
SAMPLE_FRACS = [0.05, 0.10, 0.20, 0.30]
N_REP = 300                     # samples per (sampler, frac) per graph

RESULTS_DIR = Path("results_abstract")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------

def generate_graph(graph_type: str, seed: int) -> nx.Graph:
    """Generate one graph of the requested type and seed."""
    if graph_type == "BA":
        m = TARGET_DEG // 2  # approx avg degree 2m
        return nx.barabasi_albert_graph(n=NODES, m=m, seed=seed)

    if graph_type == "ER":
        p = TARGET_DEG / (NODES - 1)
        return nx.erdos_renyi_graph(n=NODES, p=p, seed=seed)

    if graph_type == "WS":
        k = TARGET_DEG
        p_rewire = 0.1
        return nx.watts_strogatz_graph(n=NODES, k=k, p=p_rewire, seed=seed)

    if graph_type == "SBM":
        # 4 equal communities, relatively strong community structure
        sizes = [NODES // 4] * 4
        p_in = 0.05
        p_out = 0.005
        probs = [
            [p_in if i == j else p_out for j in range(4)]
            for i in range(4)
        ]
        return nx.stochastic_block_model(sizes, probs, seed=seed)

    raise ValueError(f"Unknown graph_type: {graph_type}")


# ---------------------------------------------------------------------
# Samplers & metrics
# ---------------------------------------------------------------------

METRICS = {
    "clustering": metric_global_clustering,
    "avg_path":   metric_avg_shortest_path,
}

SAMPLERS = {
    "random_node":    (sample_random_nodes, {}),
    "snowball":       (sample_snowball, {"n_seeds": 5}),
    "degree_weighted":(sample_degree_weighted_nodes, {}),
    "edge_uniform":   (sample_uniform_edges, {"induced": True}),
}


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

def run_nd_simulation() -> pd.DataFrame:
    """Run ND simulation for all graph types and seeds; return full ND table."""
    all_rows = []

    for gtype in GRAPH_TYPES:
        print(f"=== Graph type: {gtype} ===")
        for seed in range(N_GRAPHS_PER_TYPE):
            print(f"  - seed {seed}")
            G = generate_graph(gtype, seed)

            # Per-graph ND grid
            df_nd = estimate_nd_grid_fast(
                G=G,
                metrics=METRICS,
                samplers=SAMPLERS,
                sample_fracs=SAMPLE_FRACS,
                n_rep=N_REP,
                rng_seed=10_000 + 1000 * GRAPH_TYPES.index(gtype) + seed,
            )

            # Add identifiers
            df_nd["graph_type"] = gtype
            df_nd["seed"] = seed

            all_rows.append(df_nd)

    df_all = pd.concat(all_rows, ignore_index=True)

    # Save raw ND results
    out_csv = RESULTS_DIR / "nd_two_metrics_raw.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"Saved raw ND results to {out_csv}")

    return df_all


def aggregate_nd(df_all: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ND over seeds: mean + std for each (graph_type, metric, sampler, frac)."""

    # Drop NaNs in ND if any
    df = df_all.dropna(subset=["nd"])

    grouped = (
        df
        .groupby(["graph_type", "metric", "sampler", "frac"], as_index=False)
        .agg(nd_mean=("nd", "mean"), nd_std=("nd", "std"))
    )

    out_csv = RESULTS_DIR / "nd_two_metrics_agg.csv"
    grouped.to_csv(out_csv, index=False)
    print(f"Saved aggregated ND results to {out_csv}")

    return grouped


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def make_2x4_figure(df_agg: pd.DataFrame) -> None:
    """
    Make a 2x4 panel figure:

    Row 1: ND – clustering
    Row 2: ND – avg_path
    Cols : BA, ER, SBM, WS
    """

    metrics_row = ["clustering", "avg_path"]
    metric_labels = {
        "clustering": "ND – clustering",
        "avg_path":   "ND – avg shortest path",
    }

    # Fixed sampler order & colors (consistent across panels)
    samplers = ["degree_weighted", "edge_uniform", "random_node", "snowball"]
    colors   = {
        "degree_weighted": "C2",
        "edge_uniform":    "C3",
        "random_node":     "C0",
        "snowball":        "C1",
    }

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(GRAPH_TYPES),
        figsize=(16, 6),
        sharex=True,
        sharey=False,
    )

    for r, metric in enumerate(metrics_row):
        for c, gtype in enumerate(GRAPH_TYPES):
            ax = axes[r, c]
            sub = df_agg[
                (df_agg["metric"] == metric) &
                (df_agg["graph_type"] == gtype)
            ]

            for sampler in samplers:
                s = sub[sub["sampler"] == sampler].sort_values("frac")
                if s.empty:
                    continue

                ax.errorbar(
                    s["frac"].values,
                    s["nd_mean"].values,
                    yerr=s["nd_std"].values,
                    label=sampler,
                    color=colors[sampler],
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                )

            if r == 0:
                ax.set_title(gtype, fontsize=14)

            if c == 0:
                ax.set_ylabel(metric_labels[metric], fontsize=12)

            if r == 1:
                ax.set_xlabel("Sampling fraction", fontsize=12)

            ax.grid(True, alpha=0.3)

    # Shared legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Sampler",
        loc="upper center",
        ncol=len(samplers),
        frameon=False,
    )

    fig.suptitle(
        "ND (mean ± std over 5 seeds × 300 samples) vs sampling fraction",
        fontsize=16,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_png = RESULTS_DIR / "nd_two_metrics_2x4.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved 2x4 ND figure to {out_png}")

    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    df_all = run_nd_simulation()
    df_agg = aggregate_nd(df_all)
    make_2x4_figure(df_agg)


if __name__ == "__main__":
    main()