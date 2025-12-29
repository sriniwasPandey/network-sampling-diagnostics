#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 12:32:10 2025

@author: srini
"""

# run_nd_multi.py
"""
Run ND (normalized distortion) experiments over multiple graph realizations.

- Generates N_GRAPHS independent BA graphs with the same parameters.
- Runs estimate_nd_grid_fast on each graph.
- Adds graph_id and graph_seed to each ND row.
- Saves:
    - results/nd_multi_raw.csv          (all runs concatenated)
    - results/nd_multi_summary.csv      (mean and std over graph_id)
    - results/nd_multi_nd_vs_frac_*.png (ND vs frac with error bars)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sampling import (
    sample_random_nodes,
    sample_snowball,
    sample_degree_weighted_nodes,
    sample_uniform_edges,
)
from metrics import (
    metric_global_clustering,
    metric_degree_assortativity,
    metric_modularity_greedy,
    metric_avg_shortest_path,
)
from nd_diagnostics import estimate_nd_grid_fast


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

# Number of independent graph realizations
N_GRAPHS = 5

# BA graph parameters
N_NODES = 800
BA_M = 3
BA_SEED_BASE = 100   # will use BA_SEED_BASE + graph_id

# Sampling fractions and replicate count inside ND estimator
SAMPLE_FRACS = [0.05, 0.10, 0.20, 0.30]
N_REP = 100

# Random seed base for sampling (inside estimate_nd_grid_fast)
SAMPLING_SEED_BASE = 1000


# Global plotting style for samplers
SAMPLER_COLORS = {
    "random_node":     "tab:blue",
    "snowball":        "tab:orange",
    "degree_weighted": "tab:green",
    "edge_uniform":    "tab:red",
    # add more here if you add more samplers
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_output_dir(dirname: str = "results") -> Path:
    out_dir = Path(dirname)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def sanitize_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def plot_nd_multi(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot ND_mean vs frac with error bars (ND_std) for each metric,
    with one line per sampler.
    """
    metrics = df_summary["metric"].unique()
    samplers = df_summary["sampler"].unique()

    for metric in metrics:
        sub = df_summary[df_summary["metric"] == metric]

        fig, ax = plt.subplots()
        for sampler in samplers:
            s = sub[sub["sampler"] == sampler].sort_values("frac")
            if s.empty:
                continue

            x = s["frac"].values
            y = s["nd_mean"].values
            yerr = s["nd_std"].values

            color = SAMPLER_COLORS.get(sampler, None)

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linestyle="-",
                label=sampler,
                color=color,
                capsize=3,
            )

        ax.set_title(f"ND (mean ± std across graphs) – {metric}")
        ax.set_xlabel("Sampling fraction")
        ax.set_ylabel("ND")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Sampler")
        fig.tight_layout()

        fname = output_dir / f"nd_multi_nd_vs_frac_{sanitize_name(metric)}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------
# Main ND multi-graph runner
# ---------------------------------------------------------------------

def main():
    results_dir = ensure_output_dir("results")

    # 1 - Define metrics (same as in main.py)
    metrics = {
        "clustering":    metric_global_clustering,
        "assortativity": metric_degree_assortativity,
        "avg_path":      metric_avg_shortest_path,
        "modularity":    metric_modularity_greedy,
    }

    # 2 - Define samplers (same as in main.py)
    samplers = {
        "random_node":     (sample_random_nodes, {}),
        "snowball":        (sample_snowball, {"n_seeds": 5}),
        "degree_weighted": (sample_degree_weighted_nodes, {}),
        "edge_uniform":    (sample_uniform_edges, {"induced": True}),
    }

    all_dfs = []

    for graph_id in range(N_GRAPHS):
        ba_seed = BA_SEED_BASE + graph_id
        print(f"Generating BA graph {graph_id} with seed {ba_seed}...")

        G = nx.barabasi_albert_graph(n=N_NODES, m=BA_M, seed=ba_seed)

        rng_seed = SAMPLING_SEED_BASE + graph_id
        print(f"  Running ND diagnostics with rng_seed={rng_seed}...")

        df_nd = estimate_nd_grid_fast(
            G=G,
            metrics=metrics,
            samplers=samplers,
            sample_fracs=SAMPLE_FRACS,
            n_rep=N_REP,
            rng_seed=rng_seed,
        )

        df_nd["graph_id"] = graph_id
        df_nd["graph_seed"] = ba_seed

        all_dfs.append(df_nd)

    # Concatenate all ND results
    df_all = pd.concat(all_dfs, ignore_index=True)

    raw_path = results_dir / "nd_multi_raw.csv"
    df_all.to_csv(raw_path, index=False)
    print(f"Saved raw ND multi-graph results to {raw_path}")

    # Summary: mean and std over graph_id, for each metric-sampler-frac
    group_cols = ["metric", "sampler", "frac"]
    value_cols = ["theta", "mean_sampled", "bias_rel", "var_rel", "nd"]

    summary_mean = df_all.groupby(group_cols)[value_cols].mean().rename(
        columns=lambda c: c + "_mean"
    )
    summary_std = df_all.groupby(group_cols)[value_cols].std(ddof=1).rename(
        columns=lambda c: c + "_std"
    )

    df_summary = summary_mean.join(summary_std).reset_index()

    summary_path = results_dir / "nd_multi_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved ND summary over graphs to {summary_path}")

    # Make plots
    plot_nd_multi(df_summary, results_dir)
    print(f"Saved ND multi-graph plots into {results_dir.resolve()}")


if __name__ == "__main__":
    main()