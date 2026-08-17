"""
core/link_model.py — the interface a domain (QKD, classical optical WDM,
generic SDN routing, whatever) implements to plug into core/base_env.py.

This is the extension point: QKD's physics (qkd_channel.py, coexistence.py,
attack_models.py) becomes ONE implementation of this interface, not baked
into the environment itself. A different network type would implement the
same three methods with its own physics and reuse everything in
base_env.py (demand sampling loop, candidate-path selection, reward
shaping, gym.Env plumbing) unchanged.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import networkx as nx


class LinkModel(ABC):
    """
    Domain-specific physical/logical layer. Everything that's actually
    QKD-specific (or WDM-specific, or whatever) lives behind this interface.
    """

    @abstractmethod
    def build_base_graph(self, topology_name: str, rng) -> nx.Graph:
        """
        Construct the annotated graph for one episode: node/edge structure
        plus whatever per-edge quality attributes this domain needs (SKR,
        distance, capacity, latency — domain's choice). Called once per
        env.reset().
        """
        ...

    @abstractmethod
    def current_feasibility(self, base_graph: nx.Graph, t: float) -> dict[tuple[int, int], bool]:
        """
        Given the base graph and current episode time t, return which
        edges are currently usable — e.g. QKD's jamming/intercept-resend
        checks, or a classical network's congestion/failure model. Keyed
        by sorted (u, v) tuples, matching base_env's edge-key convention.
        """
        ...

    @abstractmethod
    def edge_capacity_field(self) -> str:
        """
        Name of the edge attribute base_env should read as "how much this
        edge can carry" when scoring candidate paths (e.g. 'skr_bps' for
        QKD, could be 'bandwidth_mbps' for a classical network).
        """
        ...
