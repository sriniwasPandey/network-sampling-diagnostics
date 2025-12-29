#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 12:37:06 2025

@author: srini
"""

# run_misrank_multi.py
"""
Run mis-ranking experiments over multiple BA vs WS pairs.

- Generates N_PAIRS independent pairs:
    G1 = BA(n, m) with seed1
    G2 = WS(n, k, p) with seed2
- Runs estimate_misranking_grid on each pair.
- Adds pair_id, seed1, seed2 to each row.
- Saves:
    - results/misrank_multi_raw.csv         (all pairs concatenated)
    - results/misrank_multi_summary.csv     (mean and std over pair_id)
    - results/misrank_multi_pmis_vs_frac_*.png (p_misrank vs frac ± std)
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
from misranking import estimate_misranking_grid


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

# Number of independent BA vs WS pairs
N_PAIRS = 5

# BA parameters
N_NODES = 800
BA_M = 3
BA_SEED_BASE = 200   # BA seed = BA_SEED_BASE + 2 * pair_id

# WS parameters
WS_K = 10
WS_P = 0.1
WS_SEED_BASE = 300   # WS seed = WS_SEED_BASE + 2 * pair_id + 1

# Sampling fractions and replicate count inside misranking estimator
SAMPLE_FRACS = [0.05, 0.10, 0.20, 0.30]
N_REP = 150

# Random seed base for sampling in misranking
SAMPLING_SEED_BASE = 5000


# Global plotting style for samplers
SAMPLER_COLORS = {
    "random_node":     "tab:blue",
    "snowball":        "tab:orange",
    "degree_weighted": "tab:green",
    "edge_uniform":    "tab:red",
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


def plot_misrank_multi(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot p_misrank_mean vs frac with error bars (p_misrank_std) for each metric,
    one line per sampler.
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
            y = s["p_misrank_mean"].values
            yerr = s["p_misrank_std"].values

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

        ax.set_title(f"Mis-ranking (mean ± std across pairs) – {metric}")
        ax.set_xlabel("Sampling fraction")
        ax.set_ylabel("p(mis-rank)")
        #ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Sampler")
        fig.tight_layout()

        fname = output_dir / f"misrank_multi_pmis_vs_frac_{sanitize_name(metric)}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------
# Main mis-ranking multi-pair runner
# ---------------------------------------------------------------------

def main():
    results_dir = ensure_output_dir("results")

    # 1 - Define metrics
    metrics = {
        "clustering":    metric_global_clustering,
        "assortativity": metric_degree_assortativity,
        "avg_path":      metric_avg_shortest_path,
        "modularity":    metric_modularity_greedy,
    }

    # 2 - Define samplers
    samplers = {
        "random_node":     (sample_random_nodes, {}),
        "snowball":        (sample_snowball, {"n_seeds": 5}),
        "degree_weighted": (sample_degree_weighted_nodes, {}),
        "edge_uniform":    (sample_uniform_edges, {"induced": True}),
    }

    all_dfs = []

    for pair_id in range(N_PAIRS):
        seed1 = BA_SEED_BASE + 2 * pair_id
        seed2 = WS_SEED_BASE + 2 * pair_id + 1

        print(f"Generating pair {pair_id}: BA seed={seed1}, WS seed={seed2}...")

        G1 = nx.barabasi_albert_graph(n=N_NODES, m=BA_M, seed=seed1)
        G2 = nx.watts_strogatz_graph(n=N_NODES, k=WS_K, p=WS_P, seed=seed2)

        rng_seed = SAMPLING_SEED_BASE + pair_id
        print(f"  Running mis-ranking diagnostics with rng_seed={rng_seed}...")

        df_mis = estimate_misranking_grid(
            G1=G1,
            G2=G2,
            metrics=metrics,
            samplers=samplers,
            sample_fracs=SAMPLE_FRACS,
            n_rep=N_REP,
            rng_seed=rng_seed,
        )

        df_mis["pair_id"] = pair_id
        df_mis["seed1"] = seed1
        df_mis["seed2"] = seed2

        all_dfs.append(df_mis)

    # Concatenate all mis-ranking results
    df_all = pd.concat(all_dfs, ignore_index=True)

    raw_path = results_dir / "misrank_multi_raw.csv"
    df_all.to_csv(raw_path, index=False)
    print(f"Saved raw mis-ranking multi-pair results to {raw_path}")

    # Summary: mean and std over pair_id, for each metric-sampler-frac
    group_cols = ["metric", "sampler", "frac"]
    value_cols = ["theta1", "theta2", "p_misrank", "n_valid"]

    summary_mean = df_all.groupby(group_cols)[value_cols].mean().rename(
        columns=lambda c: c + "_mean"
    )
    # Only p_misrank really needs std
    summary_std = df_all.groupby(group_cols)[["p_misrank"]].std(ddof=1).rename(
        columns=lambda c: c + "_std"
    )

    df_summary = summary_mean.join(summary_std).reset_index()

    summary_path = results_dir / "misrank_multi_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved mis-ranking summary over pairs to {summary_path}")

    # Make plots
    plot_misrank_multi(df_summary, results_dir)
    print(f"Saved mis-ranking multi-pair plots into {results_dir.resolve()}")


if __name__ == "__main__":
    main()