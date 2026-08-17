"""
attack_models.py — Physical-layer jamming attacks on QKD network links.

Per the research map: jamming is one of the two attack types this lab's
work actually models and mitigates (the other being intercept-resend, added
separately). Unlike intercept-resend, which degrades QBER on a link,
jamming is modeled as making a link TEMPORARILY UNAVAILABLE — a clean,
binary, time-varying signal that forces the routing policy (heuristic or
RL agent) to reroute around it. This is exactly the kind of dynamic
disruption an RL agent needs to see during training to learn robust
routing, as opposed to the static topology used in Phase 0.

Design: jamming events are (edge, start_time, duration) tuples applied to
a QKD-annotated graph. `apply_jamming` mutates a copy of the graph so the
underlying channel model (qkd_channel.py) is untouched — jamming affects
availability, not the physical QBER/SKR model itself.
"""

from __future__ import annotations
from dataclasses import dataclass
import random
import networkx as nx


@dataclass
class JammingEvent:
    u: int
    v: int
    start_time: float
    duration: float

    def active_at(self, t: float) -> bool:
        return self.start_time <= t < self.start_time + self.duration


def apply_jamming(g: nx.Graph, events: list[JammingEvent], t: float) -> nx.Graph:
    """
    Return a copy of `g` where every edge under active jamming at time `t`
    has feasible=False and skr_bps=0 (link unusable), regardless of its
    underlying QBER-based feasibility. Requires the graph to already be
    QKD-annotated (see baselines.annotate_graph_with_qkd).
    """
    g2 = g.copy()
    for ev in events:
        if not ev.active_at(t):
            continue
        if g2.has_edge(ev.u, ev.v):
            g2[ev.u][ev.v]["feasible"] = False
            g2[ev.u][ev.v]["skr_bps"] = 0.0
            g2[ev.u][ev.v]["jammed"] = True
    return g2


def random_jamming_schedule(
    g: nx.Graph,
    n_events: int,
    time_horizon: float,
    min_duration: float = 5.0,
    max_duration: float = 30.0,
    seed: int = 7,
) -> list[JammingEvent]:
    """
    Generate a random jamming schedule over [0, time_horizon) for use as
    training/eval disruption in the RL environment. Only targets real
    QKD hop-edges (works on relay-expanded graphs from topology.py).
    """
    rng = random.Random(seed)
    edges = list(g.edges())
    events = []
    for _ in range(n_events):
        u, v = rng.choice(edges)
        start = rng.uniform(0, time_horizon)
        duration = rng.uniform(min_duration, max_duration)
        events.append(JammingEvent(u, v, start, duration))
    return events


@dataclass
class InterceptResendEvent:
    """
    Intercept-resend eavesdropping attack on a link.

    UNLIKE jamming (a denial-of-service disruption that simply removes a
    link from availability), intercept-resend is an EAVESDROPPING attack:
    Eve measures each photon in a randomly chosen basis and resends a fresh
    photon prepared in the outcome she got. When her basis choice disagrees
    with the sender's (which happens half the time), the resent photon has
    a 50% chance of producing a wrong bit at the receiver — giving Eve's
    presence a signature QBER contribution of ~25% on the affected link,
    independent of and added on top of the link's baseline physical QBER.

    Detection mechanism matters for how this differs operationally from
    jamming: BB84's security proof means the two legitimate parties detect
    intercept-resend via QBER estimation during the sifting/parameter
    estimation step, and abort key generation on that link if QBER exceeds
    threshold — so a *successfully detected* intercept-resend attack has
    the same practical routing consequence as jamming (link becomes
    unusable), but the interpretation differs: jamming denies availability,
    intercept-resend denies *confidentiality* and is only neutralized
    because BB84's QBER monitoring catches it. Modeling both separately
    matters if you later want partial/covert attack scenarios (e.g. an
    attacker who intercepts only a fraction of pulses to stay under the
    detection threshold) — not implemented here, flagged as an extension.
    """
    u: int
    v: int
    start_time: float
    duration: float
    attack_qber_contribution: float = 0.25  # standard BB84 intercept-resend signature

    def active_at(self, t: float) -> bool:
        return self.start_time <= t < self.start_time + self.duration


def apply_intercept_resend(g: nx.Graph, events: list[InterceptResendEvent], t: float,
                            qber_max: float = 0.11) -> nx.Graph:
    """
    Return a copy of `g` where every edge under active intercept-resend
    attack at time `t` has its QBER raised by the attack's signature
    contribution and feasibility/SKR recomputed accordingly. Combines
    additively with the link's baseline physical QBER (both contribute
    errors observed by the legitimate parties).
    """
    g2 = g.copy()
    for ev in events:
        if not ev.active_at(t):
            continue
        if not g2.has_edge(ev.u, ev.v):
            continue
        edge = g2[ev.u][ev.v]
        combined_qber = min(edge.get("qber", 0.0) + ev.attack_qber_contribution, 1.0)
        detected = combined_qber >= qber_max
        edge["qber"] = combined_qber
        edge["feasible"] = not detected
        edge["skr_bps"] = 0.0 if detected else edge.get("skr_bps", 0.0)
        edge["intercept_resend_detected"] = detected
    return g2


def random_intercept_resend_schedule(
    g: nx.Graph,
    n_events: int,
    time_horizon: float,
    min_duration: float = 5.0,
    max_duration: float = 30.0,
    seed: int = 13,
) -> list[InterceptResendEvent]:
    rng = random.Random(seed)
    edges = list(g.edges())
    events = []
    for _ in range(n_events):
        u, v = rng.choice(edges)
        start = rng.uniform(0, time_horizon)
        duration = rng.uniform(min_duration, max_duration)
        events.append(InterceptResendEvent(u, v, start, duration))
    return events


if __name__ == "__main__":
    from .topology import nsfnet_topology
    from .baselines import annotate_graph_with_qkd, widest_path

    g = annotate_graph_with_qkd(nsfnet_topology())

    print("--- Jamming ---")
    events = random_jamming_schedule(g, n_events=5, time_horizon=100.0)
    src, dst = 0, 13
    for t in [0, 20, 40, 60, 80]:
        g_t = apply_jamming(g, events, t)
        result = widest_path(g_t, src, dst)
        n_jammed = sum(1 for _, _, d in g_t.edges(data=True) if d.get("jammed"))
        print(f"  t={t:>3}  jammed links: {n_jammed:>2}  route result: "
              f"{'blocked' if result is None else f'SKR={result[1]:.1f} bps'}")

    print("\n--- Intercept-resend ---")
    ir_events = random_intercept_resend_schedule(g, n_events=5, time_horizon=100.0)
    for t in [0, 20, 40, 60, 80]:
        g_t = apply_intercept_resend(g, ir_events, t)
        result = widest_path(g_t, src, dst)
        n_detected = sum(1 for _, _, d in g_t.edges(data=True) if d.get("intercept_resend_detected"))
        print(f"  t={t:>3}  attacks detected: {n_detected:>2}  route result: "
              f"{'blocked' if result is None else f'SKR={result[1]:.1f} bps'}")
