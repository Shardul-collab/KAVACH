"""
fix_imports.py — rewrites the specific import statements that break after
migrate_structure.sh moves files into the src/kavach/ layout.

This is a targeted, explicit find-and-replace list built from the ACTUAL
import statements in your files (not a generic regex sweep) — every
substitution below corresponds to a real "from .xxx import yyy" line that
existed before the move. Run this once, right after migrate_structure.sh,
before you try importing anything.

Usage:
    python3 fix_imports.py            # apply
    python3 fix_imports.py --check    # dry run, show what WOULD change
"""
from __future__ import annotations
import sys
from pathlib import Path

# (file path relative to repo root, old import line, new import line)
REPLACEMENTS: list[tuple[str, str, str]] = [
    # --- src/kavach/envs/qkd_routing_env.py (was kavach/rl_env.py) ---
    ("src/kavach/envs/qkd_routing_env.py",
     "from .topology import get_topology",
     "from ..core.topology import get_topology"),
    ("src/kavach/envs/qkd_routing_env.py",
     "from .baselines import annotate_graph_with_qkd, k_shortest_candidates, select_classical_edges",
     "from ..domains.qkd.baselines import annotate_graph_with_qkd, k_shortest_candidates, select_classical_edges"),
    ("src/kavach/envs/qkd_routing_env.py",
     "from .qkd_channel import QKDLinkParams",
     "from ..domains.qkd.qkd_channel import QKDLinkParams"),
    ("src/kavach/envs/qkd_routing_env.py",
     "from .coexistence import RamanCoexistenceParams",
     "from ..domains.qkd.coexistence import RamanCoexistenceParams"),
    ("src/kavach/envs/qkd_routing_env.py",
     "from .attack_models import (",
     "from ..domains.qkd.attack_models import ("),

    # --- src/kavach/envs/qkd_routing_env_buffered.py (was kavach/rl_env_buffered.py) ---
    ("src/kavach/envs/qkd_routing_env_buffered.py",
     "from .topology import get_topology",
     "from ..core.topology import get_topology"),
    ("src/kavach/envs/qkd_routing_env_buffered.py",
     "from .baselines import annotate_graph_with_qkd, select_classical_edges",
     "from ..domains.qkd.baselines import annotate_graph_with_qkd, select_classical_edges"),
    ("src/kavach/envs/qkd_routing_env_buffered.py",
     "from .qkd_channel import QKDLinkParams",
     "from ..domains.qkd.qkd_channel import QKDLinkParams"),
    ("src/kavach/envs/qkd_routing_env_buffered.py",
     "from .coexistence import RamanCoexistenceParams",
     "from ..domains.qkd.coexistence import RamanCoexistenceParams"),
    ("src/kavach/envs/qkd_routing_env_buffered.py",
     "from .attack_models import (",
     "from ..domains.qkd.attack_models import ("),

    # --- src/kavach/envs/qkd_routing_env_gnn.py (was kavach/rl_env_gnn.py) ---
    ("src/kavach/envs/qkd_routing_env_gnn.py",
     "from .topology import get_topology",
     "from ..core.topology import get_topology"),
    ("src/kavach/envs/qkd_routing_env_gnn.py",
     "from .baselines import annotate_graph_with_qkd, select_classical_edges",
     "from ..domains.qkd.baselines import annotate_graph_with_qkd, select_classical_edges"),
    ("src/kavach/envs/qkd_routing_env_gnn.py",
     "from .qkd_channel import QKDLinkParams",
     "from ..domains.qkd.qkd_channel import QKDLinkParams"),
    ("src/kavach/envs/qkd_routing_env_gnn.py",
     "from .coexistence import RamanCoexistenceParams",
     "from ..domains.qkd.coexistence import RamanCoexistenceParams"),
    ("src/kavach/envs/qkd_routing_env_gnn.py",
     "from .attack_models import (",
     "from ..domains.qkd.attack_models import ("),

    # --- src/kavach/training/trainer.py (was kavach/trainer.py) ---
    ("src/kavach/training/trainer.py",
     "from .rl_env import QKDRoutingEnv",
     "from ..envs.qkd_routing_env import QKDRoutingEnv"),
    ("src/kavach/training/trainer.py",
     "from .rl_env_buffered import QKDRoutingEnvBuffered",
     "from ..envs.qkd_routing_env_buffered import QKDRoutingEnvBuffered"),

    # --- src/kavach/training/train_gnn.py (was top-level train_gnn.py) ---
    ("src/kavach/training/train_gnn.py",
     "from kavach.rl_env_gnn import QKDRoutingEnvGNN",
     "from ..envs.qkd_routing_env_gnn import QKDRoutingEnvGNN"),
    ("src/kavach/training/train_gnn.py",
     "from kavach.trainer import heuristic_action",
     "from .trainer import heuristic_action"),
    ("src/kavach/training/train_gnn.py",
     "from gnn_extractor import GNNFeaturesExtractor",
     "from ..policies.gnn_extractor import GNNFeaturesExtractor"),

    # --- src/kavach/viz/export_trace.py (was kavach/export_trace.py) ---
    ("src/kavach/viz/export_trace.py",
     "from .topology import get_topology, nsfnet_topology",
     "from ..core.topology import get_topology, nsfnet_topology"),
    ("src/kavach/viz/export_trace.py",
     "from .baselines import annotate_graph_with_qkd, select_classical_edges",
     "from ..domains.qkd.baselines import annotate_graph_with_qkd, select_classical_edges"),
    ("src/kavach/viz/export_trace.py",
     "from .rl_env_buffered import QKDRoutingEnvBuffered",
     "from ..envs.qkd_routing_env_buffered import QKDRoutingEnvBuffered"),
    ("src/kavach/viz/export_trace.py",
     "from .qkd_channel import QKDLinkParams",
     "from ..domains.qkd.qkd_channel import QKDLinkParams"),
    ("src/kavach/viz/export_trace.py",
     "from .coexistence import RamanCoexistenceParams",
     "from ..domains.qkd.coexistence import RamanCoexistenceParams"),
    ("src/kavach/viz/export_trace.py",
     "from .trainer import heuristic_action",
     "from ..training.trainer import heuristic_action"),
]


def main():
    check_only = "--check" in sys.argv
    root = Path(__file__).resolve().parent
    applied, missing = 0, []

    for rel_path, old, new in REPLACEMENTS:
        path = root / rel_path
        if not path.exists():
            missing.append(f"{rel_path} (file not found — was migrate_structure.sh run first?)")
            continue
        text = path.read_text()
        if old not in text:
            if new in text:
                continue  # already applied, fine
            missing.append(f"{rel_path}: expected line not found:\n    {old}")
            continue
        if check_only:
            print(f"[would change] {rel_path}\n  - {old}\n  + {new}\n")
        else:
            path.write_text(text.replace(old, new, 1))
        applied += 1

    verb = "would apply" if check_only else "applied"
    print(f"\n{applied}/{len(REPLACEMENTS)} replacements {verb}.")
    if missing:
        print(f"\n{len(missing)} NOT matched — check these by hand:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)


if __name__ == "__main__":
    main()
