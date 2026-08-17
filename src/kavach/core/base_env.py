"""
core/base_env.py — NetworkRoutingEnv: the domain-agnostic shape shared by
every routing RL problem in this codebase (today: three QKD variants). Pull
this out once, and a future non-QKD network (classical WDM RWA/RSA, generic
SDN routing) reuses the demand loop, candidate-path action space, and gym
plumbing instead of copying rl_env.py's ~250 lines by hand again.

WHAT THIS DOES NOT DO: rewrite the three existing QKD envs to inherit from
this. That's a deliberate, separate follow-up — those three files are
tested and working against your real graph; migrating them onto this base
class blind (without being able to run it against your actual
baselines.py/attack_models.py/coexistence.py) is real regression risk that
doesn't belong bundled into "here's a new abstraction to look at."

USAGE SKETCH for a future domain (not run/tested — illustrative):

    class ClassicalWDMLinkModel(LinkModel):
        def build_base_graph(self, topology_name, rng): ...
        def current_feasibility(self, base_graph, t): ...
        def edge_capacity_field(self): return "bandwidth_mbps"

    env = NetworkRoutingEnv(link_model=ClassicalWDMLinkModel(), topology_name="nsfnet")
"""
from __future__ import annotations
import random
from typing import Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx

from .link_model import LinkModel
from .topology import get_topology

REWARD_BLOCK = -1.0


class NetworkRoutingEnv(gym.Env):
    """
    Domain-agnostic sequential routing environment: demands arrive one at a
    time between two endpoints, the agent picks among k candidate paths (or
    rejects), reward reflects whether the chosen path had enough capacity.

    Everything domain-specific (what makes an edge "feasible", what
    capacity/quality metric matters) is delegated to a LinkModel instance —
    see link_model.py.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        link_model: LinkModel,
        topology_name: str = "nsfnet",
        k_candidates: int = 4,
        demands_per_episode: int = 50,
        min_demand: float = 100.0,
        max_demand: float = 5000.0,
        demand_sampler: Callable[[list[int], random.Random], tuple[int, int, float]] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.link_model = link_model
        self.topology_name = topology_name
        self.k = k_candidates
        self.demands_per_episode = demands_per_episode
        self.min_demand = min_demand
        self.max_demand = max_demand
        self._demand_sampler = demand_sampler or self._default_demand_sampler
        self._rng = random.Random(seed)

        obs_dim = self.k * 3 + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.k + 1)

        self._base_graph: nx.Graph | None = None
        self._endpoints: list[int] = []
        self._demand_idx = 0
        self._t = 0.0
        self._current_demand = None
        self._current_candidates: list[dict] = []
        self.episode_served = 0
        self.episode_blocked = 0

    def _default_demand_sampler(self, endpoints: list[int], rng: random.Random):
        src, dst = rng.sample(endpoints, 2)
        amount = rng.uniform(self.min_demand, self.max_demand)
        return src, dst, amount

    def _candidates_for(self, src: int, dst: int, feasibility: dict) -> list[dict]:
        cap_field = self.link_model.edge_capacity_field()
        try:
            gen = nx.shortest_simple_paths(self._base_graph, src, dst, weight="distance_km")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        candidates = []
        for path in gen:
            edges = [tuple(sorted(e)) for e in zip(path[:-1], path[1:])]
            feasible = all(feasibility.get(e, False) for e in edges)
            if feasible:
                bottleneck = min(
                    self._base_graph[u][v].get(cap_field, 0.0) for u, v in zip(path[:-1], path[1:])
                )
            else:
                bottleneck = 0.0
            candidates.append({"path": path, "edges": edges, "hops": len(edges),
                                "capacity": bottleneck, "feasible": feasible})
            if len(candidates) >= self.k:
                break
        return candidates

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(self.k * 3 + 1, dtype=np.float32)
        for i, cand in enumerate(self._current_candidates[: self.k]):
            obs[i * 3 + 0] = min(cand["capacity"] / max(self.max_demand, 1.0), 1.0)
            obs[i * 3 + 1] = min(cand["hops"] / 20.0, 1.0)
            obs[i * 3 + 2] = 1.0 if cand["feasible"] else 0.0
        _, _, amount = self._current_demand
        obs[-1] = min(amount / self.max_demand, 1.0)
        return obs

    def _advance_to_next_demand(self):
        self._t += 1.0
        self._demand_idx += 1
        feasibility = self.link_model.current_feasibility(self._base_graph, self._t)
        self._current_demand = self._demand_sampler(self._endpoints, self._rng)
        src, dst, _ = self._current_demand
        self._current_candidates = self._candidates_for(src, dst, feasibility)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        self._base_graph = self.link_model.build_base_graph(self.topology_name, self._rng)
        self._endpoints = [n for n, d in self._base_graph.nodes(data=True)
                            if d.get("node_type") == "endpoint"]
        if not self._endpoints:
            self._endpoints = list(self._base_graph.nodes())  # fallback for topologies with no node_type tagging

        self._demand_idx = 0
        self._t = 0.0
        self.episode_served = 0
        self.episode_blocked = 0

        feasibility = self.link_model.current_feasibility(self._base_graph, self._t)
        self._current_demand = self._demand_sampler(self._endpoints, self._rng)
        src, dst, _ = self._current_demand
        self._current_candidates = self._candidates_for(src, dst, feasibility)

        return self._build_observation(), {}

    def step(self, action: int):
        _, _, required = self._current_demand
        reject_action = self.k

        if action == reject_action or action >= len(self._current_candidates):
            reward = REWARD_BLOCK
            self.episode_blocked += 1
        else:
            cand = self._current_candidates[action]
            if cand["feasible"] and cand["capacity"] >= required:
                reward = required / self.max_demand
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
