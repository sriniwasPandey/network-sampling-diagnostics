# main.py
"""
Demo script for network sampling diagnostics.

- Computes ND (normalized distortion) for several metrics
  under different sampling schemes and fractions on one graph.
- Computes mis-ranking probabilities for pairs of graphs.
- Saves results (CSVs + PNG plots) into a 'results' folder.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.lines import Line2D  # add at top with other imports


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
from misranking import estimate_misranking_grid
# ---------------------------------------------------------------------
# Global plotting style
# ---------------------------------------------------------------------

SAMPLER_COLORS = {
    "random_node":     "tab:blue",
    "snowball":        "tab:orange",
    "degree_weighted": "tab:green",
    "edge_uniform":    "tab:red",
    # If you later add random_walk, just uncomment:
    # "random_walk":     "tab:purple",
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
    """
    Ensure that the output directory exists and return it as a Path.
    """
    out_dir = Path(dirname)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def sanitize_name(name: str) -> str:
    """
    Make a string safe for filenames.
    """
    return name.replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------
# Plotting helpers (now with saving)
# ---------------------------------------------------------------------

def plot_nd_vs_frac(df_nd, output_dir: Path):
    """
    For each metric, plot ND vs sampling fraction with one line per sampler,
    and save each figure as PNG in output_dir.
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

            mask = ~np.isnan(y)
            if not mask.any():
                continue
            color = SAMPLER_COLORS.get(sampler, None)
            ax.plot(x[mask], y[mask], marker="o",color=color, label=sampler)

        ax.set_title(f"ND vs sampling fraction – {metric}")
        ax.set_xlabel("Sampling fraction")
        ax.set_ylabel("ND (normalized distortion)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Sampler")
        fig.tight_layout()

        fname = output_dir / f"nd_vs_frac_{sanitize_name(metric)}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")


def plot_misrank_vs_frac(df_mis, output_dir: Path):
    """
    For each metric, plot p(mis-rank) vs sampling fraction,
    with one line per sampler, and save to output_dir.
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
            color = SAMPLER_COLORS.get(sampler, None)
            ax.plot(x[mask], y[mask], marker="o", color=color,label=sampler)

        ax.set_title(f"Mis-ranking probability vs sampling fraction – {metric}")
        ax.set_xlabel("Sampling fraction")
        ax.set_ylabel("p(mis-rank)")
        #ax.set_ylim(-0.01, 1.01)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Sampler")
        fig.tight_layout()

        fname = output_dir / f"misrank_vs_frac_{sanitize_name(metric)}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")


def plot_nd_vs_misrank(df_nd, df_mis, frac_value: float, output_dir: Path):
    """
    Scatter plot of ND vs p(mis-rank) at a fixed sampling fraction.

    - Color encodes sampler (consistent with other plots).
    - Marker shape encodes metric.

    Saves figure into output_dir.
    """
    nd_sub = df_nd[df_nd["frac"] == frac_value]
    mis_sub = df_mis[df_mis["frac"] == frac_value]

    merged = nd_sub.merge(
        mis_sub[["metric", "sampler", "p_misrank"]],
        on=["metric", "sampler"],
        how="inner",
    )

    if merged.empty:
        print(f"No merged rows at frac = {frac_value}")
        return

    fig, ax = plt.subplots()

    for _, row in merged.iterrows():
        metric = row["metric"]
        sampler = row["sampler"]
        nd_val = row["nd"]
        p_val = row["p_misrank"]

        color = SAMPLER_COLORS.get(sampler, None)
        marker = METRIC_MARKERS.get(metric, "o")

        ax.scatter(
            nd_val,
            p_val,
            color=color,
            marker=marker,
            s=60,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_title(f"ND vs mis-ranking at sampling fraction = {frac_value}")
    ax.set_xlabel("ND (distortion)")
    ax.set_ylabel("p(mis-rank)")
    ax.grid(True, alpha=0.3)

    # --- Build legends: one for samplers (colors), one for metrics (markers) ---
    # Sampler legend
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
        if sampler in merged["sampler"].unique()
    ]

    # Metric legend
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
        if metric in merged["metric"].unique()
    ]

    legend1 = ax.legend(
        handles=sampler_handles,
        title="Sampler (color)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=metric_handles,
        title="Metric (marker)",
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        borderaxespad=0.0,
    )

    fig.tight_layout()

    frac_str = str(frac_value).replace(".", "p")
    fname = output_dir / f"nd_vs_misrank_frac_{frac_str}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------

def main():
    # Where to save CSVs and plots
    results_dir = ensure_output_dir("results")

    # -----------------------------
    # 1. Define ground-truth graphs
    # -----------------------------
    G = nx.barabasi_albert_graph(n=800, m=3, seed=1)

    G1 = nx.barabasi_albert_graph(n=800, m=3, seed=1)
    G2 = nx.watts_strogatz_graph(n=800, k=10, p=0.1, seed=2)

    # -----------------------------
    # 2. Define metrics and samplers
    # -----------------------------
    metrics = {
        "clustering":    metric_global_clustering,
        "assortativity": metric_degree_assortativity,
        "avg_path":      metric_avg_shortest_path,   
        "modularity":    metric_modularity_greedy,
    }

    samplers = {
        "random_node": (sample_random_nodes, {}),
        "snowball":    (sample_snowball, {"n_seeds": 5}),
        "degree_weighted": (sample_degree_weighted_nodes, {}),
        "edge_uniform":    (sample_uniform_edges, {"induced": True}),

    }

    sample_fracs = [0.05, 0.1, 0.2, 0.3]

    # -----------------------------
    # 3. Compute ND grid on G
    # -----------------------------
    print("Computing ND grid...")
    df_nd = estimate_nd_grid_fast(
        G=G,
        metrics=metrics,
        samplers=samplers,
        sample_fracs=sample_fracs,
        n_rep=100,
        rng_seed=42,
    )
    print(df_nd.head())

    # Save ND results
    nd_csv_path = results_dir / "nd_results.csv"
    df_nd.to_csv(nd_csv_path, index=False)
    print(f"Saved ND results to {nd_csv_path}")

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
        n_rep=150,
        rng_seed=123,
    )
    print(df_mis.head())

    # Save mis-ranking results
    mis_csv_path = results_dir / "misranking_results.csv"
    df_mis.to_csv(mis_csv_path, index=False)
    print(f"Saved mis-ranking results to {mis_csv_path}")

    # -----------------------------
    # 5. Plot results and save figs
    # -----------------------------
    plot_nd_vs_frac(df_nd, results_dir)
    plot_misrank_vs_frac(df_mis, results_dir)
    plot_nd_vs_misrank(df_nd, df_mis, frac_value=0.1, output_dir=results_dir)

    print(f"Plots saved into {results_dir.resolve()}")

    # Show all figures (optional; comment out if running headless)
    plt.show()


if __name__ == "__main__":
    main()