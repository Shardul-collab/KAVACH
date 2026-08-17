"""
rl_env_buffered.py — QKD routing RL environment WITH persistent link
buffers that actually deplete when a request is served.

WHY THIS FILE EXISTS (read this before using it):
Real testing of rl_env.py's QKDRoutingEnv on actual hardware showed PPO and
DQN converge to match, not beat, a greedy widest-path heuristic (measured:
heuristic 59.2% blocked vs PPO 60.0% / DQN 60.1%, all under 50k timesteps,
identical demand stream). Root cause, confirmed by inspection: that
environment recomputes every candidate's SKR fresh from the physical channel
model at every step — serving a request never reduces what's available to
LATER requests on the same link. The only time-varying element is the
jamming/intercept-resend schedule, which no action can influence. With no
action-dependent state carried between steps, the optimal policy IS the
greedy per-step choice — there's no sequential-planning problem for RL to
have an edge on, so matching the heuristic is the mathematically correct
outcome, not a training failure.

This env fixes that: each link has a persistent bit buffer that fills at
its physical SKR and DRAINS when a request is served through it. Serving a
large request now can starve a DIFFERENT, larger or better-value request a
few steps later on a shared bottleneck link — which is precisely the
FCFS/SJF/LRPT tradeoff already demonstrated standalone in key_assignment.py,
now integrated into the sequential RL setting where an agent could plausibly
learn to hold capacity back for higher-value future demands in a way a
one-step-lookahead heuristic structurally cannot.

Same candidate/observation dict SHAPE as rl_env.py's QKDRoutingEnv (reuses
trainer.py's heuristic_action/evaluate_agent/evaluate_random_policy
unchanged) but candidate['skr_bps'] here means CURRENT BOTTLENECK BUFFER
(bits available right now), not a rate — and demands specify required_bits
(an amount) rather than required_rate (a throughput). Field name kept as
'skr_bps' for code reuse; this is a units/semantics change worth remembering,
not a naming bug.
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
HOP_NORM = 20.0


class QKDRoutingEnvBuffered(gym.Env):
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

        obs_dim = self.k * 3 + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.k + 1)

        self._base_graph = None
        self._endpoints = None
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

    def _sample_demand(self):
        src, dst = self._rng.sample(self._endpoints, 2)
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
                "skr_bps": bottleneck,
                "feasible": feasible,
            })
            if len(candidates) >= self.k:
                break
        return candidates

    def _build_observation(self):
        obs = np.zeros(self.k * 3 + 1, dtype=np.float32)
        for i, cand in enumerate(self._current_candidates[: self.k]):
            obs[i * 3 + 0] = min(cand["skr_bps"] / BITS_NORM, 1.0)
            obs[i * 3 + 1] = min(cand["hops"] / HOP_NORM, 1.0)
            obs[i * 3 + 2] = 1.0 if cand["feasible"] else 0.0
        _, _, bits = self._current_demand
        obs[-1] = min(bits / self.max_bits, 1.0)
        return obs

    def _advance_to_next_demand(self):
        self._t += 1.0
        self._demand_idx += 1
        feasibility = self._current_link_feasibility()
        self._accumulate_buffers(1.0, feasibility)
        self._current_demand = self._sample_demand()
        src, dst, _ = self._current_demand
        self._current_candidates = self._candidates_for(src, dst, feasibility)

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
        self._endpoints = [n for n, d in self._base_graph.nodes(data=True)
                            if d.get("node_type") == "endpoint"]

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

        return self._build_observation(), {}

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
            self._advance_to_next_demand()
            obs = self._build_observation()
        else:
            obs = np.zeros(self.k * 3 + 1, dtype=np.float32)

        info = {"served": self.episode_served, "blocked": self.episode_blocked}
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = QKDRoutingEnvBuffered(topology_name="nsfnet", seed=0)
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Random-policy episode (buffered env): served={info['served']}, "
          f"blocked={info['blocked']}, total_reward={total_reward:.2f}")
