# main.py
"""
Demo script for network sampling diagnostics.

- Computes ND (normalized distortion) for several metrics
  under different sampling schemes and fractions on one graph.
- Computes mis-ranking probabilities for pairs of graphs.
- Plots ND vs sampling fraction and p(mis-rank) vs sampling fraction.

Requires:
    sampling.py
    metrics.py
    nd_diagnostics.py
    misranking.py
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from sampling import sample_random_nodes, sample_snowball
from metrics import (
    metric_global_clustering,
    metric_degree_assortativity,
    metric_modularity_greedy,
)
from nd_diagnostics import estimate_nd_grid_fast
from misranking import estimate_misranking_grid


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_nd_vs_frac(df_nd):
    """
    For each metric, plot ND vs sampling fraction with one line per sampler.
    """
    metrics = df_nd["metric"].unique()
    samplers = df_nd["sampler"].unique()

    for metric in metrics:
        sub = df_nd[df_nd["metric"] == metric]

        fig, ax = plt.subplots()
        for sampler in samplers:
            s = sub[sub["sampler"] == sampler].sort_values("frac")
            if s.empty:
                continue

            x = s["frac"].values
            y = s["nd"].values

            # drop NaNs
            mask = ~np.isnan(y)
            if not mask.any():
                continue

            ax.plot(x[mask], y[mask], marker="o", label=sampler)

        ax.set_title(f"ND vs sampling fraction – {metric}")
        ax.set_xlabel("Sampling fraction")
        ax.set_ylabel("ND (normalized distortion)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Sampler")
        fig.tight_layout()


def plot_misrank_vs_frac(df_mis):
    """
    For each metric, plot p(mis-rank) vs sampling fraction,
    with one line per sampler.
    """
    metrics = df_mis["metric"].unique()
    samplers = df_mis["sampler"].unique()

    for metric in metrics:
        sub = df_mis[df_mis["metric"] == metric]

        fig, ax = plt.subplots()
        for sampler in samplers:
            s = sub[sub["sampler"] == sampler].sort_values("frac")
            if s.empty:
                continue

            x = s["frac"].values
            y = s["p_misrank"].values

            mask = ~np.isnan(y)
            if not mask.any():
                continue

            ax.plot(x[mask], y[mask], marker="o", label=sampler)

        ax.set_title(f"Mis-ranking probability vs sampling fraction – {metric}")
        ax.set_xlabel("Sampling fraction")
        ax.set_ylabel("p(mis-rank)")
        ax.set_ylim(-0.01, 1.01)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Sampler")
        fig.tight_layout()


def plot_nd_vs_misrank(df_nd, df_mis, frac_value: float):
    """
    Scatter plot of ND vs p(mis-rank) at a fixed sampling fraction.

    Each point is (ND, p_misrank) for a given metric × sampler.
    """
    # restrict to the chosen fraction
    nd_sub = df_nd[df_nd["frac"] == frac_value]
    mis_sub = df_mis[df_mis["frac"] == frac_value]

    # merge on metric + sampler
    merged = nd_sub.merge(
        mis_sub[["metric", "sampler", "p_misrank"]],
        on=["metric", "sampler"],
        how="inner",
    )

    if merged.empty:
        print(f"No merged rows at frac = {frac_value}")
        return

    fig, ax = plt.subplots()

    for (metric, sampler), grp in merged.groupby(["metric", "sampler"]):
        ax.scatter(
            grp["nd"],
            grp["p_misrank"],
            label=f"{metric} / {sampler}",
        )

    ax.set_title(f"ND vs mis-ranking at sampling fraction = {frac_value}")
    ax.set_xlabel("ND (distortion)")
    ax.set_ylabel("p(mis-rank)")
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()


# ---------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------

def main():
    # -----------------------------
    # 1. Define ground-truth graphs
    # -----------------------------
    # Single graph for ND diagnostics
    G = nx.barabasi_albert_graph(n=800, m=3, seed=1)

    # Two graphs for mis-ranking: BA vs WS (or whatever you like)
    G1 = nx.barabasi_albert_graph(n=800, m=3, seed=1)
    G2 = nx.watts_strogatz_graph(n=800, k=10, p=0.1, seed=2)

    # -----------------------------
    # 2. Define metrics and samplers
    # -----------------------------
    metrics = {
        "clustering":    metric_global_clustering,
        "assortativity": metric_degree_assortativity,
        "modularity":    metric_modularity_greedy,
        # you can add avg-path approx later if you want
    }

    from sampling import sample_random_nodes, sample_snowball

    samplers = {
        "random_node": (sample_random_nodes, {}),
        "snowball":    (sample_snowball, {"n_seeds": 5}),
    }

    sample_fracs = [0.1, 0.2, 0.4, 0.6]

    # -----------------------------
    # 3. Compute ND grid on G
    # -----------------------------
    print("Computing ND grid...")
    df_nd = estimate_nd_grid_fast(
        G=G,
        metrics=metrics,
        samplers=samplers,
        sample_fracs=sample_fracs,
        n_rep=120,
        rng_seed=42,
    )
    print(df_nd.head())

    # -----------------------------
    # 4. Compute mis-ranking grid between G1 and G2
    # -----------------------------
    print("Computing mis-ranking grid...")
    df_mis = estimate_misranking_grid(
        G1=G1,
        G2=G2,
        metrics=metrics,
        samplers=samplers,
        sample_fracs=sample_fracs,
        n_rep=200,
        rng_seed=123,
    )
    print(df_mis.head())

    # -----------------------------
    # 5. Plot results
    # -----------------------------
    # ND vs frac
    plot_nd_vs_frac(df_nd)

    # p(mis-rank) vs frac
    plot_misrank_vs_frac(df_mis)

    # ND vs mis-rank at a specific fraction (e.g. 0.2)
    plot_nd_vs_misrank(df_nd, df_mis, frac_value=0.2)

    # Show all figures
    plt.show()


if __name__ == "__main__":
    main()
