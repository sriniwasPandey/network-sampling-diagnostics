# sampling.py

import random
from typing import Any, List

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

def _weighted_sample_without_replacement(
    items: List[Any],
    weights: List[float],
    k: int,
    rng: random.Random,
) -> List[Any]:
    """
    Simple weighted sampling without replacement using cumulative weights.
    Not optimized, but fine for n ~ 10^3.
    """
    chosen: List[Any] = []
    items_local = list(items)
    weights_local = list(weights)

    k = min(k, len(items_local))
    for _ in range(k):
        total_w = sum(weights_local)
        if total_w <= 0:
            # fall back to uniform among remaining
            chosen.extend(rng.sample(items_local, k - len(chosen)))
            break

        r = rng.random() * total_w
        cum = 0.0
        idx = 0
        for i, w in enumerate(weights_local):
            cum += w
            if cum >= r:
                idx = i
                break

        chosen.append(items_local.pop(idx))
        weights_local.pop(idx)

    return chosen


def sample_degree_weighted_nodes(
    G: nx.Graph,
    frac: float,
    rng: random.Random | None = None,
) -> nx.Graph:
    """
    Degree-weighted node sampling: nodes are selected with probability
    proportional to their degree.

    If all nodes have degree zero, falls back to uniform random-node sampling.
    """
    if rng is None:
        rng = random

    n = G.number_of_nodes()
    if n == 0:
        return G.copy()

    k = max(1, int(round(frac * n)))
    nodes = list(G.nodes())
    degrees = [G.degree(u) for u in nodes]

    if all(d == 0 for d in degrees):
        # no edges: just fall back to uniform
        nodes_chosen = rng.sample(nodes, min(k, n))
        return G.subgraph(nodes_chosen).copy()

    nodes_chosen = _weighted_sample_without_replacement(nodes, degrees, k, rng)
    return G.subgraph(nodes_chosen).copy()

def sample_uniform_edges(
    G: nx.Graph,
    frac: float,
    rng: random.Random | None = None,
    induced: bool = True,
) -> nx.Graph:
    """
    Uniform edge sampling:

    - Randomly permute edges.
    - Add edges one by one until the set of incident nodes reaches
      about `frac * |V|` distinct nodes (or we run out of edges).
    - If `induced=True`, return the induced subgraph on those nodes
      (i.e., include *all* edges between the chosen nodes).
      Otherwise, return the graph with only the sampled edges.

    Notes:
        - If the graph has no edges, falls back to copying G.
        - `frac` is interpreted as a *node* fraction, for consistency
          with your other samplers.
    """
    if rng is None:
        rng = random

    N = G.number_of_nodes()
    M = G.number_of_edges()

    if N == 0 or M == 0:
        return G.copy()

    target_nodes = max(1, int(round(frac * N)))

    edges_list: List[Any] = list(G.edges())
    rng.shuffle(edges_list)

    visited_nodes = set()
    sampled_edges: List[tuple[Any, Any]] = []

    for u, v in edges_list:
        sampled_edges.append((u, v))
        visited_nodes.add(u)
        visited_nodes.add(v)
        if len(visited_nodes) >= target_nodes:
            break

    if induced:
        # Use all edges between the nodes we touched
        return G.subgraph(visited_nodes).copy()
    else:
        # Only keep the sampled edges
        H = nx.Graph()
        H.add_nodes_from(visited_nodes)
        H.add_edges_from(sampled_edges)
        return H