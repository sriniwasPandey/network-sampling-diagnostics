# nd_diagnostics.py

from typing import Callable, Dict, Any, List, Tuple
import random

import networkx as nx
import numpy as np
import pandas as pd


def _compute_nd_for_values(
    theta: float,
    values: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Helper: given the true value theta and a 1D array of sampled metric values,
    compute mean, relative bias, relative variability, and ND.

    ND = sqrt( bias_rel^2 + var_rel^2 ), where
        bias_rel = (mean_sampled - theta) / (|theta| + eps)
        var_rel  = std_sampled / (|theta| + eps)
    """
    if values.size == 0:
        return {
            "theta": theta,
            "mean_sampled": np.nan,
            "bias_rel": np.nan,
            "var_rel": np.nan,
            "nd": np.nan,
        }

    mean_sampled = float(values.mean())
    std_sampled = float(values.std(ddof=1)) if values.size > 1 else 0.0

    scale = abs(theta) + eps
    bias_rel = (mean_sampled - theta) / scale
    var_rel = std_sampled / scale
    nd = float(np.sqrt(bias_rel**2 + var_rel**2))

    return {
        "theta": float(theta),
        "mean_sampled": mean_sampled,
        "bias_rel": float(bias_rel),
        "var_rel": float(var_rel),
        "nd": nd,
    }


def estimate_nd_grid_fast(
    G: nx.Graph,
    metrics: Dict[str, Callable[[nx.Graph], float]],
    samplers: Dict[str, Tuple[Callable[..., nx.Graph], Dict[str, Any]]],
    sample_fracs: List[float],
    n_rep: int = 200,
    eps: float = 1e-6,
    rng_seed: int | None = None,
) -> pd.DataFrame:
    """
    Faster ND estimation over:
        metric × sampler × sampling fraction.

    For each (sampler, frac) pair:
      - Generate a batch of sampled graphs ONCE.
      - Reuse this batch for ALL metrics.

    Parameters
    ----------
    G : nx.Graph
        Ground-truth network.
    metrics : dict
        {metric_name: metric_fn}, where metric_fn: G -> float.
    samplers : dict
        {sampler_name: (sampler_fn, base_kwargs)}.
        sampler_fn(G, frac=..., rng=...) -> nx.Graph
        base_kwargs MUST NOT contain 'frac' or 'rng'; those are added here.
    sample_fracs : list of float
        Sampling fractions to evaluate, e.g., [0.1, 0.2, 0.4].
    n_rep : int
        Number of sampling replicates per (sampler, frac).
    eps : float
        Small constant for numerical stability in normalization.
    rng_seed : int or None
        Seed for reproducibility of the sampling.

    Returns
    -------
    pandas.DataFrame with columns:
        metric, sampler, frac, theta, mean_sampled, bias_rel, var_rel, nd
    """
    base_rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
    rows: List[Dict[str, Any]] = []

    # Precompute true metric values on the full graph once
    true_values: Dict[str, float] = {
        m_name: m_fn(G) for m_name, m_fn in metrics.items()
    }

    for sampler_name, (sampler_fn, base_kwargs) in samplers.items():
        for frac in sample_fracs:
            # Fresh RNG for this (sampler, frac) combination
            rng = random.Random(base_rng.randint(0, 10**9))

            # 1) Generate sampled graphs ONCE for this sampler & frac
            sampled_graphs: List[nx.Graph] = []
            for _ in range(n_rep):
                kwargs = dict(base_kwargs)
                kwargs.update({"frac": frac, "rng": rng})
                H = sampler_fn(G, **kwargs)
                if H.number_of_nodes() == 0:
                    continue
                sampled_graphs.append(H)

            # If no valid samples, record NaNs for all metrics
            if not sampled_graphs:
                for metric_name in metrics.keys():
                    rows.append({
                        "metric": metric_name,
                        "sampler": sampler_name,
                        "frac": frac,
                        "theta": true_values[metric_name],
                        "mean_sampled": np.nan,
                        "bias_rel": np.nan,
                        "var_rel": np.nan,
                        "nd": np.nan,
                    })
                continue

            # 2) For each metric, compute ND using this fixed batch
            for metric_name, metric_fn in metrics.items():
                theta = true_values[metric_name]

                vals: List[float] = []
                for H in sampled_graphs:
                    v = metric_fn(H)
                    if np.isnan(v):
                        continue
                    vals.append(v)

                vals_arr = np.asarray(vals, dtype=float)
                stats = _compute_nd_for_values(theta, vals_arr, eps=eps)

                row = {
                    "metric": metric_name,
                    "sampler": sampler_name,
                    "frac": frac,
                }
                row.update(stats)
                rows.append(row)

    return pd.DataFrame(rows)
