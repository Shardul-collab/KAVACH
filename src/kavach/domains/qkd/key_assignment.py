"""
key_assignment.py — Secret-key buffer management and assignment ordering.

Distinct problem from routing (baselines.py / rl_env.py). QKD links don't
deliver keys instantaneously on demand — they continuously generate key
material at their SKR into a per-link buffer, and REQUESTS CONSUME FROM
THAT BUFFER. When a link's buffer is fragmented across many small pending
requests, larger requests can starve even though the *cumulative* generation
rate would eventually cover them — this is the fragmentation problem from
the 2024 paper, and the ordering-policy problem from the 2022 paper.

MODEL:
  - Each link has a key buffer (bits) that fills at its SKR (bits/sec).
  - A request consumes `required_bits` from the buffer of every link on its
    (already-chosen) path — bottlenecked by whichever link's buffer is
    smallest, same bottleneck logic as the routing layer.
  - Multiple pending requests compete for buffer on shared links. The
    ORDER in which they're served changes total throughput and blocking,
    even with the routing decision already fixed — this is what "ordering
    policy" optimizes, separately from path selection.

POLICIES implemented:
  - FCFS: serve requests in arrival order (naive baseline)
  - SJF (shortest-job-first): serve smallest `required_bits` first — reduces
    fragmentation by clearing small requests before they get stranded behind
    a large one hogging a shared bottleneck link, at the cost of potentially
    starving large requests (classic SJF tradeoff, same as CPU/disk
    scheduling — worth naming explicitly if asked about this in interview).
  - LRPT (largest-remaining-first): opposite of SJF, included as a contrast
    case to make the ordering effect visible in results, not because it's
    a good policy.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class KeyRequest:
    id: int
    path_edges: list[tuple[int, int]]   # sequence of (u, v) edges on the chosen path
    required_bits: float
    arrival_time: float


@dataclass
class LinkBuffer:
    skr_bps: float
    bits_available: float = 0.0

    def accumulate(self, dt: float):
        self.bits_available += self.skr_bps * dt


class KeyAssignmentSimulator:
    """
    Simulates buffer accumulation + request consumption over a sequence of
    time steps, under a chosen ordering policy, on a FIXED set of paths
    (routing already decided — this module only tests assignment ordering).
    """

    def __init__(self, link_skr: dict[tuple[int, int], float]):
        # normalize edge keys both directions so undirected lookups work
        self.buffers: dict[tuple[int, int], LinkBuffer] = {}
        for (u, v), skr in link_skr.items():
            key = tuple(sorted((u, v)))
            self.buffers[key] = LinkBuffer(skr_bps=skr)

    def _buffer_for(self, edge: tuple[int, int]) -> LinkBuffer:
        return self.buffers[tuple(sorted(edge))]

    def _can_serve(self, req: KeyRequest) -> bool:
        return all(self._buffer_for(e).bits_available >= req.required_bits
                   for e in req.path_edges)

    def _serve(self, req: KeyRequest):
        for e in req.path_edges:
            self._buffer_for(e).bits_available -= req.required_bits

    def run(self, requests: list[KeyRequest], policy: str, dt: float = 1.0,
            time_horizon: float = 100.0) -> dict:
        """
        Time-steps through [0, time_horizon), accumulating buffers each step
        and attempting to serve pending requests in the order the policy
        dictates, re-sorted at every step (since buffer state changes).
        Returns served/blocked counts and average wait time for served requests.
        """
        pending = list(requests)
        served, served_wait_times = [], []

        t = 0.0
        while t < time_horizon and pending:
            for buf in self.buffers.values():
                buf.accumulate(dt)

            arrived = [r for r in pending if r.arrival_time <= t]
            arrived = self._order(arrived, policy)

            for req in arrived:
                if self._can_serve(req):
                    self._serve(req)
                    served.append(req)
                    served_wait_times.append(t - req.arrival_time)
                    pending.remove(req)

            t += dt

        blocked = pending  # never served within time_horizon
        return {
            "served": len(served),
            "blocked": len(blocked),
            "blocking_prob": len(blocked) / len(requests) if requests else 0.0,
            "avg_wait": sum(served_wait_times) / len(served_wait_times) if served_wait_times else None,
        }

    @staticmethod
    def _order(requests: list[KeyRequest], policy: str) -> list[KeyRequest]:
        if policy == "fcfs":
            return sorted(requests, key=lambda r: r.arrival_time)
        elif policy == "sjf":
            return sorted(requests, key=lambda r: r.required_bits)
        elif policy == "lrpt":
            return sorted(requests, key=lambda r: -r.required_bits)
        else:
            raise ValueError(f"Unknown policy '{policy}'. Options: fcfs, sjf, lrpt")


if __name__ == "__main__":
    import random

    # Isolated synthetic scenario, deliberately NOT using NSFNET's 50-hop
    # relay-expanded paths — those conflate the ordering-policy question
    # with routing hop-count effects and absolute deadline feasibility
    # (verified: with NSFNET's ~2600 bps per-hop rate and 50-hop paths,
    # most "blocking" was requests that couldn't fit in the time horizon
    # AT ALL regardless of order, which showed identical results across
    # FCFS/SJF/LRPT and taught nothing about ordering). Here: one shared
    # bottleneck link, several competing requests, sizes chosen so ordering
    # visibly changes which/how-many get served.

    rng = random.Random(0)
    link_skr = {(0, 1): 1000.0}  # single shared bottleneck, 1000 bps

    # Large dt (=6s) so buffer jumps to exactly 6000 bits in a single
    # accumulation step, making three same-arrival-time requests
    # SIMULTANEOUSLY contend for one buffer snapshot within a single round —
    # this is the actual condition under which check order changes the
    # outcome (earlier attempts with continuous small dt increments hit each
    # request's threshold in separate rounds one at a time, so a failed
    # check never blocked anyone and every policy produced identical
    # results — a real finding about this admission-control model, not
    # a bug: order only matters under genuine same-instant contention).
    requests = [
        KeyRequest(id=0, path_edges=[(0, 1)], required_bits=3000, arrival_time=0),
        KeyRequest(id=1, path_edges=[(0, 1)], required_bits=3000, arrival_time=0),
        KeyRequest(id=2, path_edges=[(0, 1)], required_bits=4000, arrival_time=0),
    ]
    # Buffer after one 6s step = 6000 bits. Sum of all three = 10000 (won't
    # all fit). SJF tries the two 3000s first -> both fit (6000 used
    # exactly), 4000 blocked: 2 served. LRPT tries the 4000 first -> fits
    # (buffer left = 2000), neither 3000 fits after: 1 served. Same total
    # demand, same buffer, different SERVED COUNT purely from check order —
    # this is the actual effect the 2022 paper's ordering policy targets.

    print(f"=== Synthetic single-bottleneck ordering demo ({len(requests)} requests, "
          f"link SKR=1000 bps, single contended round) ===\n")
    for policy in ["fcfs", "sjf", "lrpt"]:
        sim = KeyAssignmentSimulator(link_skr)
        stats = sim.run(requests, policy=policy, time_horizon=6.0, dt=6.0)
        wait_str = f"{stats['avg_wait']:.2f}s" if stats["avg_wait"] is not None else "n/a"
        print(f"{policy.upper():<6} served={stats['served']}  blocked={stats['blocked']}  "
              f"blocking_prob={stats['blocking_prob']:.2%}  avg_wait={wait_str}")
