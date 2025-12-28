# sampling.py

import random
from typing import Any

import networkx as nx


def sample_random_nodes(
    G: nx.Graph,
    frac: float,
    rng: random.Random | None = None,
) -> nx.Graph:
    """
    Random node sampling: choose a fraction of nodes uniformly at random
    and return the induced subgraph.
    """
    if rng is None:
        rng = random

    n = G.number_of_nodes()
    if n == 0:
        return G.copy()

    k = max(1, int(round(frac * n)))
    nodes = rng.sample(list(G.nodes()), min(k, n))
    return G.subgraph(nodes).copy()


def sample_snowball(
    G: nx.Graph,
    frac: float,
    n_seeds: int = 1,
    rng: random.Random | None = None,
) -> nx.Graph:
    """
    Snowball sampling: start from random seed nodes and grow via BFS
    until the target fraction of nodes is reached (or no more neighbors).
    """
    if rng is None:
        rng = random

    N = G.number_of_nodes()
    if N == 0:
        return G.copy()

    target_size = max(1, int(round(frac * N)))
    all_nodes = list(G.nodes())

    seeds = rng.sample(all_nodes, min(n_seeds, N))
    visited = set(seeds)
    frontier = list(seeds)

    while frontier and len(visited) < target_size:
        new_frontier = []
        for u in frontier:
            for v in G.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    new_frontier.append(v)
                    if len(visited) >= target_size:
                        break
            if len(visited) >= target_size:
                break

        frontier = new_frontier

        # If BFS dies out before reaching target, restart from a fresh seed
        if not frontier and len(visited) < target_size:
            remaining = [n for n in all_nodes if n not in visited]
            if not remaining:
                break
            new_seed = rng.choice(remaining)
            visited.add(new_seed)
            frontier = [new_seed]

    return G.subgraph(visited).copy()