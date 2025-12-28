# misranking.py

from typing import Callable, Dict, Any, List, Tuple
import random

import networkx as nx
import numpy as np
import pandas as pd


def estimate_misranking_single(
    G1: nx.Graph,
    G2: nx.Graph,
    metric_fn: Callable[[nx.Graph], float],
    sampler_fn: Callable[..., nx.Graph],
    sampler_kwargs1: Dict[str, Any],
    sampler_kwargs2: Dict[str, Any],
    n_rep: int = 500,
) -> Dict[str, float]:
    """
    Estimate mis-ranking probability for a single (metric, sampler, frac).

    Let:
        theta1 = metric_fn(G1)
        theta2 = metric_fn(G2)

    Define the "true" ordering by sign(theta1 - theta2).
    Then:
        - sample G1 and G2 n_rep times with the same sampling design,
        - compute metric_fn(H1), metric_fn(H2) on the sampled graphs,
        - estimate the probability that the sampled ordering contradicts
          the true ordering.

    Returns
    -------
    dict with keys:
        'theta1', 'theta2' : true metric values on G1, G2
        'sign'             : sign(theta1 - theta2) in {-1, 0, +1}
        'p_misrank'        : estimated mis-ranking probability
        'n_valid'          : number of valid sample pairs used
    """
    theta1 = metric_fn(G1)
    theta2 = metric_fn(G2)

    if np.isnan(theta1) or np.isnan(theta2):
        return {
            "theta1": float(theta1),
            "theta2": float(theta2),
            "sign": 0.0,
            "p_misrank": np.nan,
            "n_valid": 0.0,
        }

    diff = theta1 - theta2
    if diff > 0:
        true_sign = 1
    elif diff < 0:
        true_sign = -1
    else:
        true_sign = 0

    n_bad = 0
    n_valid = 0

    for _ in range(n_rep):
        H1 = sampler_fn(G1, **sampler_kwargs1)
        H2 = sampler_fn(G2, **sampler_kwargs2)

        if H1.number_of_nodes() == 0 or H2.number_of_nodes() == 0:
            continue

        v1 = metric_fn(H1)
        v2 = metric_fn(H2)

        if np.isnan(v1) or np.isnan(v2):
            continue

        n_valid += 1

        if true_sign > 0:
            # True: theta1 > theta2. Mis-rank if v1 <= v2
            if v1 <= v2:
                n_bad += 1
        elif true_sign < 0:
            # True: theta1 < theta2. Mis-rank if v1 >= v2
            if v1 >= v2:
                n_bad += 1
        else:
            # True: theta1 == theta2. If desired, treat any inequality as mis-rank
            if v1 != v2:
                n_bad += 1

    p_mis = np.nan if n_valid == 0 else n_bad / n_valid

    return {
        "theta1": float(theta1),
        "theta2": float(theta2),
        "sign": float(true_sign),
        "p_misrank": float(p_mis),
        "n_valid": float(n_valid),
    }


def estimate_misranking_grid(
    G1: nx.Graph,
    G2: nx.Graph,
    metrics: Dict[str, Callable[[nx.Graph], float]],
    samplers: Dict[str, Tuple[Callable[..., nx.Graph], Dict[str, Any]]],
    sample_fracs: List[float],
    n_rep: int = 500,
    rng_seed: int | None = None,
) -> pd.DataFrame:
    """
    Estimate mis-ranking probability for all combinations of:
        metric × sampler × sampling fraction,
    for a PAIR of networks (G1, G2).

    Parameters
    ----------
    G1, G2 : nx.Graph
        Two ground-truth networks to compare.
    metrics : dict
        {metric_name: metric_fn}, where metric_fn: G -> float.
    samplers : dict
        {sampler_name: (sampler_fn, base_kwargs)}.
        sampler_fn(G, frac=..., rng=...) -> nx.Graph
        base_kwargs MUST NOT contain 'frac' or 'rng'; added internally.
    sample_fracs : list of float
        Sampling fractions to evaluate.
    n_rep : int
        Number of replicate sample pairs per combination.
    rng_seed : int or None
        Seed for reproducibility.

    Returns
    -------
    pandas.DataFrame with columns:
        metric, sampler, frac,
        theta1, theta2, sign, p_misrank, n_valid
    """
    base_rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
    rows: List[Dict[str, Any]] = []

    # Precompute true metric values on full graphs (handy for reporting)
    true_vals1 = {m_name: m_fn(G1) for m_name, m_fn in metrics.items()}
    true_vals2 = {m_name: m_fn(G2) for m_name, m_fn in metrics.items()}

    for sampler_name, (sampler_fn, base_kwargs) in samplers.items():
        for frac in sample_fracs:
            # Fresh RNG seeds per (sampler, frac) for each graph
            rng1 = random.Random(base_rng.randint(0, 10**9))
            rng2 = random.Random(base_rng.randint(0, 10**9))

            for metric_name, metric_fn in metrics.items():
                # Build kwargs for each graph (same frac, different rng)
                kwargs1 = dict(base_kwargs)
                kwargs1.update({"frac": frac, "rng": rng1})

                kwargs2 = dict(base_kwargs)
                kwargs2.update({"frac": frac, "rng": rng2})

                res = estimate_misranking_single(
                    G1=G1,
                    G2=G2,
                    metric_fn=metric_fn,
                    sampler_fn=sampler_fn,
                    sampler_kwargs1=kwargs1,
                    sampler_kwargs2=kwargs2,
                    n_rep=n_rep,
                )

                row = {
                    "metric": metric_name,
                    "sampler": sampler_name,
                    "frac": frac,
                    "theta1": true_vals1[metric_name],
                    "theta2": true_vals2[metric_name],
                }
                row.update(res)
                rows.append(row)

    return pd.DataFrame(rows)
