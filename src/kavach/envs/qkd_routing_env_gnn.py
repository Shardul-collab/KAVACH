"""
rl_env_gnn.py — QKD routing env with a GRAPH-STRUCTURED observation, for use
with a GNN policy instead of the flat-vector MLP policy in rl_env.py /
rl_env_buffered.py.

Does NOT modify either existing env file. Physics/dynamics here are the
buffered-link model from rl_env_buffered.py (persistent per-edge bit
buffers that drain when served) — copied rather than imported, because the
observation-building step is threaded through the same reset()/step() flow
and diverging it via inheritance would be more fragile than a direct copy
for a file this size. If rl_env_buffered.py's physics changes, this file
needs the same change applied by hand — worth a comment reminder at the
top of both files if you want to guard against drift.

OBSERVATION SPACE (gymnasium.spaces.Dict):
    node_feat:        (max_nodes, 4)   [is_endpoint, is_relay, is_src, is_dst]
    edge_index:       (2, 2*max_edges) static, both directions of every
                       undirected edge, identical every step (topology is
                       fixed for a given topology_name — only the edge/node
                       FEATURES vary step to step, not the graph structure)
    edge_feat:        (2*max_edges, 4) [dist_norm, buffer_level_norm,
                       feasible_flag, classical_active_flag], mirrored for
                       both directions of each edge
    node_mask:         (max_nodes,)    all-ones here (no padding needed —
                       max_nodes is set to the EXACT node count of the
                       chosen topology, not an upper bound). Kept in the
                       space for forward-compatibility if you later want to
                       train one policy across topologies of different size
                       via padding — not needed for a single fixed topology.
    candidate_masks:   (k, max_nodes)  which nodes lie on each of the k
                       candidate paths (for the GNN extractor's per-path
                       pooling)
    demand_feat:       (1,)            required_bits, normalized

Everything else (heuristic_action(env) from trainer.py, evaluate_agent,
evaluate_random_policy) keeps working unmodified against this env: it still
exposes self.k, self._current_demand, self._current_candidates with the
same dict keys (path, edges, hops, skr_bps, feasible) as both existing envs.
"""

from __future__ import annotations
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx

from ..core.topology import get_topology
from ..domains.qkd.baselines import annotate_graph_with_qkd, select_classical_edges
from ..domains.qkd.qkd_channel import QKDLinkParams
from ..domains.qkd.coexistence import RamanCoexistenceParams
from ..domains.qkd.attack_models import (
    apply_jamming, random_jamming_schedule,
    apply_intercept_resend, random_intercept_resend_schedule,
)

REWARD_BLOCK = -1.0
BITS_NORM = 50_000.0
DIST_NORM = 2000.0  # generous ceiling on per-hop distance_km after topology's relay subdivision

NODE_FEAT_DIM = 4
EDGE_FEAT_DIM = 4


class QKDRoutingEnvGNN(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        topology_name: str = "nsfnet",
        k_candidates: int = 4,
        demands_per_episode: int = 50,
        min_bits: float = 2_000.0,
        max_bits: float = 15_000.0,
        n_jamming_events: int = 8,
        n_intercept_resend_events: int = 8,
        channel_params: QKDLinkParams | None = None,
        coexist_params: RamanCoexistenceParams | None = None,
        classical_traffic_fraction: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.topology_name = topology_name
        self.k = k_candidates
        self.demands_per_episode = demands_per_episode
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.n_jamming_events = n_jamming_events
        self.n_intercept_resend_events = n_intercept_resend_events
        self.channel_params = channel_params or QKDLinkParams()
        self.coexist_params = coexist_params or RamanCoexistenceParams()
        self.classical_traffic_fraction = classical_traffic_fraction
        self._rng = random.Random(seed)

        # Probe the topology once to fix graph size/structure. nsfnet_topology()
        # and ubn24_topology() are deterministic (no randomness in edge/node
        # construction), so node/edge COUNT is stable across every reset() —
        # only classical_edges selection and jamming schedules are randomized
        # per-episode, not the graph's structure.
        probe = get_topology(topology_name)
        self.max_nodes = probe.number_of_nodes()
        self._node_list = sorted(probe.nodes())
        self._node_index = {n: i for i, n in enumerate(self._node_list)}
        self._undirected_edges = list(probe.edges())
        self.max_edges = len(self._undirected_edges)

        src = [self._node_index[u] for u, v in self._undirected_edges] + \
              [self._node_index[v] for u, v in self._undirected_edges]
        dst = [self._node_index[v] for u, v in self._undirected_edges] + \
              [self._node_index[u] for u, v in self._undirected_edges]
        self._edge_index = np.array([src, dst], dtype=np.int64)  # (2, 2*max_edges), static

        self.observation_space = spaces.Dict({
            "node_feat": spaces.Box(low=-1.0, high=1.0,
                                     shape=(self.max_nodes, NODE_FEAT_DIM), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=self.max_nodes - 1,
                                      shape=self._edge_index.shape, dtype=np.int64),
            "edge_feat": spaces.Box(low=-1.0, high=1.0,
                                     shape=(self._edge_index.shape[1], EDGE_FEAT_DIM), dtype=np.float32),
            "node_mask": spaces.Box(low=0.0, high=1.0, shape=(self.max_nodes,), dtype=np.float32),
            "candidate_masks": spaces.Box(low=0.0, high=1.0,
                                           shape=(self.k, self.max_nodes), dtype=np.float32),
            "demand_feat": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(self.k + 1)

        self._base_graph = None
        self._jamming_events = None
        self._intercept_resend_events = None
        self._buffers: dict[tuple[int, int], float] = {}
        self._link_skr: dict[tuple[int, int], float] = {}
        self._demand_idx = 0
        self._t = 0.0
        self._current_demand = None
        self._current_candidates = None

        self.episode_served = 0
        self.episode_blocked = 0

    # --- physics: identical to rl_env_buffered.py's QKDRoutingEnvBuffered ---

    def _sample_demand(self):
        endpoints = [n for n, d in self._base_graph.nodes(data=True) if d.get("node_type") == "endpoint"]
        src, dst = self._rng.sample(endpoints, 2)
        bits = self._rng.uniform(self.min_bits, self.max_bits)
        return src, dst, bits

    def _current_link_feasibility(self):
        g = apply_jamming(self._base_graph, self._jamming_events, self._t)
        g = apply_intercept_resend(g, self._intercept_resend_events, self._t,
                                    qber_max=self.channel_params.qber_max)
        return {tuple(sorted((u, v))): d["feasible"] for u, v, d in g.edges(data=True)}

    def _accumulate_buffers(self, dt: float, feasibility: dict):
        for edge, skr in self._link_skr.items():
            if feasibility.get(edge, False):
                self._buffers[edge] += skr * dt

    def _candidates_for(self, src: int, dst: int, feasibility: dict) -> list[dict]:
        try:
            gen = nx.shortest_simple_paths(self._base_graph, src, dst, weight="distance_km")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        candidates = []
        for path in gen:
            edges = [tuple(sorted(e)) for e in zip(path[:-1], path[1:])]
            feasible = all(feasibility.get(e, False) for e in edges)
            bottleneck = min((self._buffers[e] for e in edges), default=0.0) if feasible else 0.0
            candidates.append({
                "path": path, "edges": edges, "hops": len(edges),
                "skr_bps": bottleneck, "feasible": feasible,
            })
            if len(candidates) >= self.k:
                break
        return candidates

    # --- observation building: graph-structured, not flattened ---

    def _build_node_feat(self) -> np.ndarray:
        nf = np.zeros((self.max_nodes, NODE_FEAT_DIM), dtype=np.float32)
        src, dst, _ = self._current_demand
        for n in self._node_list:
            i = self._node_index[n]
            is_relay = 1.0 if self._base_graph.nodes[n].get("node_type") == "relay" else 0.0
            nf[i, 0] = 1.0 - is_relay
            nf[i, 1] = is_relay
            nf[i, 2] = 1.0 if n == src else 0.0
            nf[i, 3] = 1.0 if n == dst else 0.0
        return nf

    def _build_edge_feat(self, feasibility: dict) -> np.ndarray:
        m = self.max_edges
        ef = np.zeros((2 * m, EDGE_FEAT_DIM), dtype=np.float32)
        for idx, (u, v) in enumerate(self._undirected_edges):
            key = tuple(sorted((u, v)))
            d = self._base_graph[u][v]
            dist_norm = min(d.get("distance_km", 0.0) / DIST_NORM, 1.0)
            cap_norm = min(self._buffers.get(key, 0.0) / BITS_NORM, 1.0)
            feas = 1.0 if feasibility.get(key, False) else 0.0
            classical = 1.0 if d.get("classical_active", False) else 0.0
            row = np.array([dist_norm, cap_norm, feas, classical], dtype=np.float32)
            ef[idx] = row
            ef[idx + m] = row  # mirror for the reverse-direction edge
        return ef

    def _build_candidate_masks(self) -> np.ndarray:
        cm = np.zeros((self.k, self.max_nodes), dtype=np.float32)
        for i, cand in enumerate(self._current_candidates[: self.k]):
            for n in cand["path"]:
                cm[i, self._node_index[n]] = 1.0
        return cm

    def _build_observation(self, feasibility: dict) -> dict:
        _, _, bits = self._current_demand
        return {
            "node_feat": self._build_node_feat(),
            "edge_index": self._edge_index.copy(),
            "edge_feat": self._build_edge_feat(feasibility),
            "node_mask": np.ones(self.max_nodes, dtype=np.float32),
            "candidate_masks": self._build_candidate_masks(),
            "demand_feat": np.array([min(bits / self.max_bits, 1.0)], dtype=np.float32),
        }

    # --- gym API ---

    def _advance_to_next_demand(self):
        self._t += 1.0
        self._demand_idx += 1
        feasibility = self._current_link_feasibility()
        self._accumulate_buffers(1.0, feasibility)
        self._current_demand = self._sample_demand()
        src, dst, _ = self._current_demand
        self._current_candidates = self._candidates_for(src, dst, feasibility)
        return feasibility

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        raw_graph = get_topology(self.topology_name)
        classical_edges = select_classical_edges(raw_graph, self.classical_traffic_fraction, self._rng)
        self._base_graph = annotate_graph_with_qkd(
            raw_graph, self.channel_params,
            classical_edges=classical_edges,
            coexist_params=self.coexist_params,
        )

        self._link_skr = {tuple(sorted((u, v))): d["skr_bps"]
                           for u, v, d in self._base_graph.edges(data=True)}
        self._buffers = {edge: 0.0 for edge in self._link_skr}

        self._jamming_events = random_jamming_schedule(
            self._base_graph, n_events=self.n_jamming_events,
            time_horizon=float(self.demands_per_episode),
            seed=self._rng.randint(0, 2**31 - 1),
        )
        self._intercept_resend_events = random_intercept_resend_schedule(
            self._base_graph, n_events=self.n_intercept_resend_events,
            time_horizon=float(self.demands_per_episode),
            seed=self._rng.randint(0, 2**31 - 1),
        )

        self._demand_idx = 0
        self._t = 0.0
        self.episode_served = 0
        self.episode_blocked = 0

        feasibility = self._current_link_feasibility()
        self._current_demand = self._sample_demand()
        src, dst, _ = self._current_demand
        self._current_candidates = self._candidates_for(src, dst, feasibility)

        return self._build_observation(feasibility), {}

    def step(self, action: int):
        _, _, required_bits = self._current_demand
        reject_action = self.k

        if action == reject_action or action >= len(self._current_candidates):
            reward = REWARD_BLOCK
            self.episode_blocked += 1
        else:
            cand = self._current_candidates[action]
            if cand["feasible"] and cand["skr_bps"] >= required_bits:
                for edge in cand["edges"]:
                    self._buffers[edge] -= required_bits
                reward = required_bits / self.max_bits
                self.episode_served += 1
            else:
                reward = REWARD_BLOCK
                self.episode_blocked += 1

        terminated = self._demand_idx + 1 >= self.demands_per_episode
        truncated = False

        if not terminated:
            feasibility = self._advance_to_next_demand()
            obs = self._build_observation(feasibility)
        else:
            obs = {
                "node_feat": np.zeros((self.max_nodes, NODE_FEAT_DIM), dtype=np.float32),
                "edge_index": self._edge_index.copy(),
                "edge_feat": np.zeros((2 * self.max_edges, EDGE_FEAT_DIM), dtype=np.float32),
                "node_mask": np.ones(self.max_nodes, dtype=np.float32),
                "candidate_masks": np.zeros((self.k, self.max_nodes), dtype=np.float32),
                "demand_feat": np.zeros(1, dtype=np.float32),
            }

        info = {"served": self.episode_served, "blocked": self.episode_blocked}
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = QKDRoutingEnvGNN(topology_name="nsfnet", seed=0)
    obs, info = env.reset()
    print(f"max_nodes={env.max_nodes}  max_edges={env.max_edges}  "
          f"edge_index shape={obs['edge_index'].shape}")
    total_reward = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Random-policy episode (GNN env): served={info['served']}, "
          f"blocked={info['blocked']}, total_reward={total_reward:.2f}")
