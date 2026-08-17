"""
rl_env.py — Gymnasium environment for RL-based routing in QKD-secured
optical networks.

Matches the problem framing from the reference paper (Sharma/Bhatia 2023):
sequential lightpath (here: QKD key-relay path) requests arrive one at a
time; the agent picks a route for each; reward reflects blocking vs.
successfully served secret key rate. Evaluated on NSFNET and UBN24.

ACTION SPACE DESIGN — k-shortest-candidate selection, not per-hop routing.
The agent chooses among `k_candidates` precomputed simple paths (by hop
distance) between the demand's source and destination, plus one "reject"
action. This keeps the action space FIXED-SIZE regardless of topology size
(NSFNET: 14 endpoints/294 hop-edges vs UBN24: 24 endpoints/393 hop-edges),
which is what makes a single trained policy comparable across topologies —
consistent with how this class of RWA/RSA papers construct the action space.

OBSERVATION SPACE — for each of the k candidates: [bottleneck_skr (norm),
hop_count (norm), feasible_flag], flattened, plus the current demand's
required_rate (norm). Candidates beyond what exists for this demand are
zero-padded with feasible=0.

REWARD:
    served (chosen path's bottleneck SKR >= required_rate): + required_rate / max_rate
    blocked (agent picked reject, or an infeasible/insufficient path):     -1.0
This is the "minimize blocking + maximize secret key rate" objective
confirmed by the research map — exact weighting is a guess pending the
paper's precise formula; adjust REWARD_SERVED_SCALE / REWARD_BLOCK below
once you have it.

DYNAMICS — both jamming and intercept-resend events (attack_models.py) are
active across the episode timeline; each step advances virtual time by 1
unit, so later demands in an episode may find previously-safe links
disrupted, forcing rerouting. The two are applied together each step: an
edge is unusable if EITHER attack is active on it at that time — the agent
sees a single combined "link unusable" signal per candidate path (via the
bottleneck feasibility check), not which attack type caused it. Distinguishing
attack type in the observation (rather than just its effect) would be a
reasonable extension if you want the agent to learn attack-specific responses,
not implemented here.
"""

from __future__ import annotations
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..core.topology import get_topology
from ..domains.qkd.baselines import annotate_graph_with_qkd, k_shortest_candidates, select_classical_edges
from ..domains.qkd.qkd_channel import QKDLinkParams
from ..domains.qkd.coexistence import RamanCoexistenceParams
from ..domains.qkd.attack_models import (
    apply_jamming, random_jamming_schedule,
    apply_intercept_resend, random_intercept_resend_schedule,
)

REWARD_BLOCK = -1.0
SKR_NORM = 5000.0   # bps, normalization ceiling (matches demo.py demand range)
HOP_NORM = 20.0      # generous upper bound on candidate path hop count


class QKDRoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        topology_name: str = "nsfnet",
        k_candidates: int = 4,
        demands_per_episode: int = 50,
        min_rate_bps: float = 100.0,
        max_rate_bps: float = 5000.0,
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
        self.min_rate_bps = min_rate_bps
        self.max_rate_bps = max_rate_bps
        self.n_jamming_events = n_jamming_events
        self.n_intercept_resend_events = n_intercept_resend_events
        self.channel_params = channel_params or QKDLinkParams()
        self.coexist_params = coexist_params or RamanCoexistenceParams()
        self.classical_traffic_fraction = classical_traffic_fraction
        self._rng = random.Random(seed)

        # k candidates * 3 features + 1 (required rate) = observation size
        obs_dim = self.k * 3 + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.k + 1)  # last action = reject

        self._base_graph = None
        self._endpoints = None
        self._jamming_events = None
        self._intercept_resend_events = None
        self._demand_idx = 0
        self._t = 0.0
        self._current_demand = None
        self._current_candidates = None

        # episode-level stats, useful for eval callbacks
        self.episode_served = 0
        self.episode_blocked = 0

    def _sample_demand(self):
        src, dst = self._rng.sample(self._endpoints, 2)
        rate = self._rng.uniform(self.min_rate_bps, self.max_rate_bps)
        return src, dst, rate

    def _current_graph(self):
        g = apply_jamming(self._base_graph, self._jamming_events, self._t)
        g = apply_intercept_resend(g, self._intercept_resend_events, self._t,
                                    qber_max=self.channel_params.qber_max)
        return g

    def _build_observation(self):
        obs = np.zeros(self.k * 3 + 1, dtype=np.float32)
        for i, cand in enumerate(self._current_candidates[: self.k]):
            obs[i * 3 + 0] = min(cand["skr_bps"] / SKR_NORM, 1.0)
            obs[i * 3 + 1] = min(cand["hops"] / HOP_NORM, 1.0)
            obs[i * 3 + 2] = 1.0 if cand["feasible"] else 0.0
        _, _, rate = self._current_demand
        obs[-1] = min(rate / self.max_rate_bps, 1.0)
        return obs

    def _advance_to_next_demand(self):
        self._t += 1.0
        self._demand_idx += 1
        self._current_demand = self._sample_demand()
        src, dst, _ = self._current_demand
        g_now = self._current_graph()
        self._current_candidates = k_shortest_candidates(g_now, src, dst, self.k)

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

        self._jamming_events = random_jamming_schedule(
            self._base_graph,
            n_events=self.n_jamming_events,
            time_horizon=float(self.demands_per_episode),
            seed=self._rng.randint(0, 2**31 - 1),
        )
        self._intercept_resend_events = random_intercept_resend_schedule(
            self._base_graph,
            n_events=self.n_intercept_resend_events,
            time_horizon=float(self.demands_per_episode),
            seed=self._rng.randint(0, 2**31 - 1),
        )

        self._demand_idx = 0
        self._t = 0.0
        self.episode_served = 0
        self.episode_blocked = 0

        self._current_demand = self._sample_demand()
        src, dst, _ = self._current_demand
        g_now = self._current_graph()
        self._current_candidates = k_shortest_candidates(g_now, src, dst, self.k)

        return self._build_observation(), {}

    def step(self, action: int):
        _, _, required_rate = self._current_demand
        reject_action = self.k  # last index

        if action == reject_action or action >= len(self._current_candidates):
            reward = REWARD_BLOCK
            self.episode_blocked += 1
        else:
            cand = self._current_candidates[action]
            if cand["feasible"] and cand["skr_bps"] >= required_rate:
                reward = required_rate / self.max_rate_bps
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

        info = {
            "served": self.episode_served,
            "blocked": self.episode_blocked,
        }
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = QKDRoutingEnv(topology_name="nsfnet", seed=0)
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = env.action_space.sample()  # random policy smoke test
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Random-policy episode: served={info['served']}, blocked={info['blocked']}, "
          f"total_reward={total_reward:.2f}")
