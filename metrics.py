# metrics.py

from typing import List

import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities, modularity


def metric_global_clustering(G: nx.Graph) -> float:
    """
    Global clustering coefficient (transitivity).

    Returns NaN for empty graphs.
    """
    if G.number_of_nodes() == 0:
        return np.nan
    return nx.transitivity(G)


def metric_avg_shortest_path(
    G: nx.Graph,
) -> float:
    """
    Average shortest path length.

    - For disconnected graphs, uses the largest connected component.
    - Returns NaN if graph is too small / has no edges.
    """
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return np.nan

    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)

    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc)
    if H.number_of_nodes() < 2:
        return np.nan
    return nx.average_shortest_path_length(H)


def metric_avg_shortest_path_approx(
    G: nx.Graph,
    n_sources: int = 50,
    rng=None,
) -> float:
    """
    Approximate average shortest-path length by:

    - sampling up to n_sources nodes,
    - computing single-source shortest paths from each,
    - averaging all distances.

    Falls back to exact metric_avg_shortest_path for small graphs.
    """
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return np.nan

    if rng is None:
        import random
        rng = random

    n = G.number_of_nodes()
    if n <= n_sources:
        return metric_avg_shortest_path(G)

    nodes = list(G.nodes())
    sources = rng.sample(nodes, n_sources)

    dists: List[int] = []
    for s in sources:
        lengths = nx.single_source_shortest_path_length(G, s)
        dists.extend(lengths.values())

    if len(dists) <= 1:
        return np.nan
    return float(np.mean(dists))


def metric_degree_assortativity(G: nx.Graph) -> float:
    """
    Degree assortativity coefficient.

    Returns NaN if no edges or if NetworkX fails.
    """
    if G.number_of_edges() == 0:
        return np.nan
    try:
        return nx.degree_assortativity_coefficient(G)
    except Exception:
        return np.nan


def metric_modularity_greedy(G: nx.Graph) -> float:
    """
    Modularity of communities detected by a greedy heuristic.

    - Returns NaN for very small graphs or graphs with no edges.
    - Can be relatively expensive for very large graphs.
    """
    if G.number_of_nodes() < 10 or G.number_of_edges() == 0:
        return np.nan
    try:
        comms = greedy_modularity_communities(G)
        return modularity(G, comms)
    except Exception:
        return np.nan
