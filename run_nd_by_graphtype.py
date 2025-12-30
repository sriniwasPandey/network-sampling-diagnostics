#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 15:32:04 2025

@author: srini
"""

# run_nd_by_graphtype.py
"""
ND diagnostics across different graph topologies.

For each graph type (ER, WS, BA, SBM):
  - Generate N_GRAPHS independent realizations.
  - Run estimate_nd_grid_fast with the existing samplers and fractions.
  - Aggregate ND (and components) over graph_id.

Outputs:
  - results/nd_bytype_raw.csv
      All runs, with columns: graph_type, graph_id, graph_seed, ...
  - results/nd_bytype_summary.csv
      Mean and std of ND (and components) for each
      (graph_type, metric, sampler, frac).

Plots:
  1) ND vs sampling fraction, faceted by graph_type, for each metric.
     - results/nd_bytype_nd_vs_frac_<metric>.png

  2) Heatmaps of ND at a fixed fraction (FRAC_FOR_HEATMAP),
     for each graph_type, with metrics × samplers grid.
     - results/nd_bytype_heatmap_frac_<frac_str>_<graph_type>.png
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from networkx.generators.community import stochastic_block_model

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

GraphType = Literal["ER", "WS", "BA", "SBM"]

GRAPH_TYPES: list[GraphType] = ["ER", "WS", "BA", "SBM"]

# Number of independent realizations per graph type
N_GRAPHS = 5

# Common size
N_NODES = 800

# ER parameters
# We roughly target average degree ~ 6-8
ER_P = 0.01   # avg_degree ≈ p * (n-1) ≈ 8

# WS parameters
WS_K = 10
WS_P = 0.1

# BA parameters
BA_M = 3

# SBM parameters (4 equal blocks, stronger intra-block connectivity)
SBM_BLOCK_SIZES = [200, 200, 200, 200]
SBM_PIN = 0.05
SBM_POUT = 0.005

# Base seeds per type (to keep things reproducible but distinct)
SEED_BASE = {
    "ER": 1000,
    "WS": 2000,
    "BA": 3000,
    "SBM": 4000,
}

# Sampling fractions and replicate count inside ND estimator
SAMPLE_FRACS = [0.05, 0.10, 0.20, 0.30]
N_REP = 100

# Random seed base for sampling (inside estimate_nd_grid_fast)
SAMPLING_SEED_BASE = 9000

# Fraction to use for heatmaps
FRAC_FOR_HEATMAP = 0.10


# Global plotting style for samplers (consistent with main.py)
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


def frac_to_str(frac: float) -> str:
    return str(frac).replace(".", "p")


# ---------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------

def generate_graph(graph_type: GraphType, seed: int) -> nx.Graph:
    """
    Generate a graph of the requested type with the given seed.
    """
    if graph_type == "ER":
        G = nx.erdos_renyi_graph(n=N_NODES, p=ER_P, seed=seed)

    elif graph_type == "WS":
        G = nx.watts_strogatz_graph(n=N_NODES, k=WS_K, p=WS_P, seed=seed)

    elif graph_type == "BA":
        G = nx.barabasi_albert_graph(n=N_NODES, m=BA_M, seed=seed)

    elif graph_type == "SBM":
        # 4-block stochastic block model
        k = len(SBM_BLOCK_SIZES)
        p_matrix = [[SBM_POUT] * k for _ in range(k)]
        for i in range(k):
            p_matrix[i][i] = SBM_PIN
        G = stochastic_block_model(
            SBM_BLOCK_SIZES,
            p_matrix,
            seed=seed,
        )

        # Convert to simple Graph (SBM returns a Graph already, but to be safe)
        G = nx.Graph(G)

    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # Ensure we work on the largest connected component if needed
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    return G


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_nd_vs_frac_by_type(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """
    For each metric, create a figure with one subplot per graph_type.
    Each subplot: ND_mean ± ND_std vs frac, one line per sampler.
    """
    metrics = df_summary["metric"].unique()
    samplers = df_summary["sampler"].unique()
    graph_types = df_summary["graph_type"].unique()

    for metric in metrics:
        sub_metric = df_summary[df_summary["metric"] == metric]

        n_types = len(graph_types)
        fig, axes = plt.subplots(
            1,
            n_types,
            figsize=(4 * n_types, 4),
            sharey=True,
        )
        if n_types == 1:
            axes = [axes]

        for ax, gtype in zip(axes, graph_types):
            sub_gt = sub_metric[sub_metric["graph_type"] == gtype]

            for sampler in samplers:
                s = sub_gt[sub_gt["sampler"] == sampler].sort_values("frac")
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

            ax.set_title(f"{gtype}")
            ax.set_xlabel("Sampling fraction")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(f"ND – {metric}")
        # single legend for all samplers
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Sampler",
            loc="upper center",
            ncol=len(samplers),
        )

        fig.suptitle(f"ND (mean ± std over graphs) vs frac – {metric}")
        fig.tight_layout(rect=[0, 0.0, 1, 0.88])

        fname = output_dir / f"nd_bytype_nd_vs_frac_{sanitize_name(metric)}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_nd_heatmaps_by_type(df_summary: pd.DataFrame,
                             frac_value: float,
                             output_dir: Path) -> None:
    """
    For a fixed sampling fraction, create heatmaps of ND_mean
    for each graph_type, with metrics on rows and samplers on columns.
    """
    graph_types = df_summary["graph_type"].unique()
    frac_str = frac_to_str(frac_value)

    # Fix ordering for rows/columns
    metrics = sorted(df_summary["metric"].unique())
    samplers = sorted(df_summary["sampler"].unique())

    for gtype in graph_types:
        sub = df_summary[
            (df_summary["graph_type"] == gtype)
            & (df_summary["frac"] == frac_value)
        ]
        if sub.empty:
            continue

        # Pivot: rows = metric, columns = sampler
        pivot = sub.pivot(
            index="metric",
            columns="sampler",
            values="nd_mean",
        ).reindex(index=metrics, columns=samplers)

        fig, ax = plt.subplots(figsize=(1.5 * len(samplers), 1.5 * len(metrics)))
        im = ax.imshow(pivot.values, aspect="auto")

        ax.set_xticks(np.arange(len(samplers)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(samplers, rotation=45, ha="right")
        ax.set_yticklabels(metrics)

        ax.set_title(f"ND at frac={frac_value:.2f} – {gtype}")
        ax.set_xlabel("Sampler")
        ax.set_ylabel("Metric")

        # Annotate with values
        for i in range(len(metrics)):
            for j in range(len(samplers)):
                val = pivot.values[i, j]
                if np.isnan(val):
                    text = "NA"
                else:
                    text = f"{val:.2f}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if not np.isnan(val) and val > np.nanmean(pivot.values) else "black",
                    fontsize=8,
                )

        fig.colorbar(im, ax=ax, label="ND_mean")
        fig.tight_layout()

        fname = output_dir / f"nd_bytype_heatmap_frac_{frac_str}_{gtype}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def main():
    results_dir = ensure_output_dir("results")

    # Metrics
    metrics = {
        "clustering":    metric_global_clustering,
        "assortativity": metric_degree_assortativity,
        "avg_path":      metric_avg_shortest_path,
        "modularity":    metric_modularity_greedy,
    }

    # Samplers
    samplers = {
        "random_node":     (sample_random_nodes, {}),
        "snowball":        (sample_snowball, {"n_seeds": 5}),
        "degree_weighted": (sample_degree_weighted_nodes, {}),
        "edge_uniform":    (sample_uniform_edges, {"induced": True}),
    }

    all_dfs = []

    for graph_type in GRAPH_TYPES:
        base_seed = SEED_BASE[graph_type]
        print(f"=== Graph type: {graph_type} ===")

        for graph_id in range(N_GRAPHS):
            graph_seed = base_seed + graph_id
            print(f"  Generating {graph_type} graph_id={graph_id} seed={graph_seed}...")

            G = generate_graph(graph_type, graph_seed)

            rng_seed = SAMPLING_SEED_BASE + hash((graph_type, graph_id)) % 10**6
            print(f"    Running ND diagnostics with rng_seed={rng_seed}...")

            df_nd = estimate_nd_grid_fast(
                G=G,
                metrics=metrics,
                samplers=samplers,
                sample_fracs=SAMPLE_FRACS,
                n_rep=N_REP,
                rng_seed=rng_seed,
            )

            df_nd["graph_type"] = graph_type
            df_nd["graph_id"] = graph_id
            df_nd["graph_seed"] = graph_seed

            all_dfs.append(df_nd)

    # Concatenate all ND results
    df_all = pd.concat(all_dfs, ignore_index=True)

    raw_path = results_dir / "nd_bytype_raw.csv"
    df_all.to_csv(raw_path, index=False)
    print(f"Saved raw ND by graph-type results to {raw_path}")

    # Summary: mean and std over graph_id, for each graph_type-metric-sampler-frac
    group_cols = ["graph_type", "metric", "sampler", "frac"]
    value_cols = ["theta", "mean_sampled", "bias_rel", "var_rel", "nd"]

    summary_mean = df_all.groupby(group_cols)[value_cols].mean().rename(
        columns=lambda c: c + "_mean"
    )
    summary_std = df_all.groupby(group_cols)[value_cols].std(ddof=1).rename(
        columns=lambda c: c + "_std"
    )

    df_summary = summary_mean.join(summary_std).reset_index()

    summary_path = results_dir / "nd_bytype_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved ND by graph-type summary to {summary_path}")

    # Plots
    plot_nd_vs_frac_by_type(df_summary, results_dir)
    plot_nd_heatmaps_by_type(df_summary, FRAC_FOR_HEATMAP, results_dir)

    print(f"Saved ND by graph-type plots in {results_dir.resolve()}")


if __name__ == "__main__":
    main()