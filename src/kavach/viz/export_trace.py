"""
export_trace.py — Export a REAL, fully-reproducible episode of the widest-path
heuristic running on the actual NSFNET graph, for honest visualization.

Nothing here is simulated or randomized for display purposes:
- Uses the actual QKDRoutingEnvBuffered class, actual topology.py graph
  construction, actual annotate_graph_with_qkd/coexistence pipeline.
- Uses seed=1000, matching episode 0 of trainer.py's real
  evaluate_heuristic_policy(eval_env, n_episodes=20) loop exactly.
- heuristic_action() is imported directly from trainer.py, not reimplemented.

VISUAL COMPRESSION (documented, not hidden): the real graph has 294 individual
QKD relay hop-edges and ~280 relay nodes (e.g. the 2800km node0-node7 backbone
edge is 35 real hops). For legibility, backbone edges are exported as single
"logical_edges" between the 14 real endpoint nodes, with per-edge aggregate
stats (bottleneck SKR, hop count, fraction of hops carrying classical traffic)
computed directly from the real per-hop annotated graph — not invented.
"""

import json
import sys

from ..core.topology import get_topology, nsfnet_topology
from ..domains.qkd.baselines import annotate_graph_with_qkd, select_classical_edges
from ..envs.qkd_routing_env_buffered import QKDRoutingEnvBuffered
from ..domains.qkd.qkd_channel import QKDLinkParams
from ..domains.qkd.coexistence import RamanCoexistenceParams
from ..training.trainer import heuristic_action

TOPOLOGY = "nsfnet"
CLASSICAL_FRACTION = 0.3
SEED = 1000  # matches trainer.py's evaluate_heuristic_policy episode 0 (seed=1000+ep, ep=0)


def build_logical_edges(raw_graph, annotated_graph):
    """
    Reconstruct the real backbone-edge -> relay-chain mapping the same way
    topology.py built it, then aggregate real per-hop data onto each of the
    22 real backbone (endpoint-to-endpoint) edges for legible visualization.
    """
    endpoints = {n for n, d in raw_graph.nodes(data=True) if d.get("node_type") == "endpoint"}
    visited_backbone = set()
    logical_edges = []

    for u in endpoints:
        for v in raw_graph.neighbors(u):
            # walk the chain until we hit another endpoint
            if raw_graph.nodes[v].get("node_type") == "endpoint":
                chain_nodes = [u, v]
            else:
                chain_nodes = [u]
                prev, cur = u, v
                while raw_graph.nodes[cur].get("node_type") == "relay":
                    chain_nodes.append(cur)
                    nxts = [n for n in raw_graph.neighbors(cur) if n != prev]
                    prev, cur = cur, nxts[0]
                chain_nodes.append(cur)
            endpoint_a, endpoint_b = chain_nodes[0], chain_nodes[-1]
            key = tuple(sorted((endpoint_a, endpoint_b)))
            if key in visited_backbone or endpoint_a == endpoint_b:
                continue
            # only record once per direction pair, but need distinct chains for parallel-ish walks
            hop_edges = list(zip(chain_nodes[:-1], chain_nodes[1:]))
            visited_backbone.add(key)

            total_dist = sum(annotated_graph[a][b]["distance_km"] for a, b in hop_edges)
            n_classical = sum(1 for a, b in hop_edges if annotated_graph[a][b].get("classical_active", False))
            bottleneck_skr = min(annotated_graph[a][b]["skr_bps"] for a, b in hop_edges)
            all_feasible = all(annotated_graph[a][b]["feasible"] for a, b in hop_edges)

            logical_edges.append({
                "u": endpoint_a, "v": endpoint_b,
                "distance_km": round(total_dist, 1),
                "n_hops": len(hop_edges),
                "classical_hop_fraction": round(n_classical / len(hop_edges), 3),
                "bottleneck_skr_bps": round(bottleneck_skr, 1),
                "feasible": all_feasible,
            })
    return logical_edges


def path_to_endpoints(path, raw_graph):
    """Filter a full relay-level path down to just the endpoint node ids it passes through."""
    endpoints = {n for n, d in raw_graph.nodes(data=True) if d.get("node_type") == "endpoint"}
    return [n for n in path if n in endpoints]


def main():
    raw = get_topology(TOPOLOGY)
    params = QKDLinkParams()
    coexist = RamanCoexistenceParams()

    env = QKDRoutingEnvBuffered(
        topology_name=TOPOLOGY,
        classical_traffic_fraction=CLASSICAL_FRACTION,
        channel_params=params,
        coexist_params=coexist,
    )
    obs, _ = env.reset(seed=SEED)

    logical_edges = build_logical_edges(raw, env._base_graph)
    endpoints = sorted([n for n, d in env._base_graph.nodes(data=True) if d.get("node_type") == "endpoint"])

    steps = []
    served, blocked = 0, 0
    done = False
    step_idx = 0

    while not done:
        demand_src, demand_dst, required_bits = env._current_demand
        candidates_summary = [
            {
                "logical_path": path_to_endpoints(c["path"], raw),
                "hops": c["hops"],
                "bottleneck_bits": round(c["skr_bps"], 1),
                "feasible": c["feasible"],
            }
            for c in env._current_candidates[: env.k]
        ]
        action = heuristic_action(env)
        chosen_logical_path = None
        if action < len(env._current_candidates):
            chosen_logical_path = path_to_endpoints(env._current_candidates[action]["path"], raw)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        outcome = "served" if info["served"] > served else "blocked"
        served, blocked = info["served"], info["blocked"]

        steps.append({
            "step_idx": step_idx,
            "demand": {"src": demand_src, "dst": demand_dst, "required_bits": round(required_bits, 1)},
            "candidates": candidates_summary,
            "chosen_action": int(action),
            "chosen_logical_path": chosen_logical_path,
            "outcome": outcome,
            "served_total": served,
            "blocked_total": blocked,
        })
        step_idx += 1

    output = {
        "config": {
            "topology": TOPOLOGY,
            "env_type": "buffered",
            "classical_traffic_fraction": CLASSICAL_FRACTION,
            "seed": SEED,
            "note": "Real, fully-reproducible widest-path heuristic episode. "
                    "Matches episode 0 of trainer.py's actual evaluate_heuristic_policy "
                    "eval loop (seed=1000). 294 real relay hop-edges compressed into 22 "
                    "logical endpoint-to-endpoint edges for visualization; all stats on "
                    "those edges are real aggregates of the underlying per-hop graph, "
                    "not invented.",
        },
        "endpoints": endpoints,
        "logical_edges": logical_edges,
        "steps": steps,
        "final": {
            "served": served,
            "blocked": blocked,
            "blocking_prob": round(blocked / (served + blocked), 4),
        },
    }

    with open("trace_export.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported {len(steps)} real steps. served={served} blocked={blocked} "
          f"blocking_prob={output['final']['blocking_prob']:.2%}")
    print("Written to trace_export.json")


if __name__ == "__main__":
    main()
