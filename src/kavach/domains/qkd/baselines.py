"""
baselines.py — Heuristic routing for QKD-secured networks, used as the
comparison baseline once an RL agent is trained (Phase 1).

Key modeling choice: QKD networks are typically built as *trusted-node
relay* networks — each link independently generates a secret key, and
end-to-end keys are relayed hop-by-hop (XOR'd at each trusted node), not
transmitted as a single quantum channel across the whole path. This means:

  - A path is FEASIBLE iff every hop on it is individually feasible
    (QBER < threshold on every link).
  - The achievable end-to-end secret key rate for a path is the MINIMUM
    of the per-hop secret key rates (bottleneck link), not a product or sum.

This makes path selection a "widest path" / maximin problem, not a
shortest-path problem — the standard heuristic in this literature is
exactly this maximin routing, which is what we implement here as the
baseline the RL agent (Phase 1) should learn to beat.
"""

from __future__ import annotations
import heapq
import random
import networkx as nx

from .qkd_channel import compute_link, QKDLinkParams
from .coexistence import compute_link_with_coexistence, RamanCoexistenceParams


def annotate_graph_with_qkd(
    g: nx.Graph,
    params: QKDLinkParams | None = None,
    classical_edges: set[tuple[int, int]] | None = None,
    coexist_params: RamanCoexistenceParams | None = None,
) -> nx.Graph:
    """
    Compute and attach per-link QKD metrics (qber, skr_bps, feasible) to
    every edge. If `classical_edges` is given, those edges get Raman-
    scattering coexistence noise from a co-propagating classical WDM
    channel folded into their QBER/SKR (coexistence.py), instead of the
    quantum-only qkd_channel.compute_link path.
    """
    p = params or QKDLinkParams()
    cp = coexist_params or RamanCoexistenceParams()
    classical_edges = classical_edges or set()
    g2 = g.copy()
    for u, v, data in g2.edges(data=True):
        edge_key = tuple(sorted((u, v)))
        if edge_key in classical_edges:
            link = compute_link_with_coexistence(data["distance_km"], p, cp, classical_active=True)
        else:
            link = compute_link(data["distance_km"], p)
            link["extra_dark_count"] = 0.0
            link["classical_active"] = False
        data.update(link)
    return g2


def select_classical_edges(g: nx.Graph, fraction: float, rng: random.Random) -> set[tuple[int, int]]:
    """
    Randomly select a fraction of edges to carry co-propagating classical
    WDM traffic (Raman coexistence noise). fraction=0.0 -> no edges,
    preserving pure quantum-only behavior.
    """
    if fraction <= 0.0:
        return set()
    edges = [tuple(sorted((u, v))) for u, v in g.edges()]
    n_select = max(1, int(round(len(edges) * fraction)))
    rng.shuffle(edges)
    return set(edges[:n_select])


def widest_path(g: nx.Graph, source: int, target: int) -> tuple[list[int], float] | None:
    """
    Maximin routing: find the path from source to target that maximizes the
    minimum per-hop secret key rate along the path, restricted to feasible
    (QBER-below-threshold) links only.

    Returns (path_as_node_list, bottleneck_skr_bps) or None if unreachable.
    """
    # bottleneck[node] = best (max) bottleneck SKR achievable to reach `node` so far
    bottleneck = {source: float("inf")}
    prev = {}
    # max-heap via negation
    pq = [(-float("inf"), source)]
    visited = set()

    while pq:
        neg_bw, u = heapq.heappop(pq)
        bw = -neg_bw
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break

        for v in g.neighbors(u):
            data = g[u][v]
            if not data.get("feasible", False):
                continue  # link unusable — QBER over threshold
            edge_skr = data["skr_bps"]
            candidate = min(bw, edge_skr)
            if candidate > bottleneck.get(v, -1):
                bottleneck[v] = candidate
                prev[v] = u
                heapq.heappush(pq, (-candidate, v))

    if target not in bottleneck or bottleneck[target] <= 0:
        return None

    # reconstruct path
    path = [target]
    node = target
    while node != source:
        node = prev[node]
        path.append(node)
    path.reverse()
    return path, bottleneck[target]


def shortest_hop_path(g: nx.Graph, source: int, target: int) -> tuple[list[int], float] | None:
    """
    Naive baseline: shortest hop-count path, ignoring key-rate at routing time.
    Feasibility/SKR is only checked *after* the path is chosen — this is the
    "dumb" baseline that a key-rate-aware policy (heuristic widest-path, or
    later the RL agent) should outperform.
    """
    try:
        path = nx.shortest_path(g, source, target)
    except nx.NetworkXNoPath:
        return None

    bottleneck = float("inf")
    for u, v in zip(path[:-1], path[1:]):
        data = g[u][v]
        if not data.get("feasible", False):
            return path, 0.0  # path exists but is infeasible end-to-end
        bottleneck = min(bottleneck, data["skr_bps"])
    return path, bottleneck


def path_metrics(g: nx.Graph, path: list[int]) -> dict:
    """
    Compute bottleneck SKR / feasibility for an arbitrary given path (node
    list). Used to score k-shortest-path candidates for the RL action space —
    unlike widest_path, this does not search, it just evaluates one path.
    """
    bottleneck = float("inf")
    feasible = True
    for u, v in zip(path[:-1], path[1:]):
        data = g[u][v]
        if not data.get("feasible", False):
            feasible = False
            bottleneck = 0.0
            break
        bottleneck = min(bottleneck, data["skr_bps"])
    return {"path": path, "hops": len(path) - 1, "skr_bps": bottleneck, "feasible": feasible}


def k_shortest_candidates(g: nx.Graph, source: int, target: int, k: int) -> list[dict]:
    """
    Generate up to k candidate paths (by hop distance, i.e. cheapest-first
    simple paths) between source and target, each scored via path_metrics.
    This is the standard RL action-space construction for RWA/RSA problems:
    the agent picks among a small fixed number of precomputed candidate
    paths rather than making per-hop decisions over a variable-size graph —
    keeps the action space fixed-size across topologies of different sizes
    (NSFNET vs UBN24).
    """
    try:
        gen = nx.shortest_simple_paths(g, source, target, weight="distance_km")
        candidates = []
        for path in gen:
            candidates.append(path_metrics(g, path))
            if len(candidates) >= k:
                break
        return candidates
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


if __name__ == "__main__":
    from kavach.core.topology import nsfnet_topology, describe

    g = nsfnet_topology()
    g = annotate_graph_with_qkd(g)
    print(describe(g))
    print()

    for src, dst in [(0, 13), (2, 12), (7, 4)]:
        wp = widest_path(g, src, dst)
        sp = shortest_hop_path(g, src, dst)
        print(f"{src} -> {dst}:")
        print(f"  widest-path (key-rate-aware): {wp}")
        print(f"  shortest-hop (naive):         {sp}")
