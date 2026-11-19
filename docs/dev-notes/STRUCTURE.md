# Kavach — proposed repo structure

```
Kavach/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
│
├── src/kavach/
│   ├── __init__.py
│   │
│   ├── core/                    # GENERIC network-sim framework — domain-agnostic
│   │   ├── __init__.py
│   │   ├── base_env.py          # NetworkRoutingEnv: abstract base any network-routing
│   │   │                        #   RL problem can subclass (QKD today; classical WDM
│   │   │                        #   RWA/RSA, SDN routing, etc. tomorrow — same shape:
│   │   │                        #   graph + sequential demands + candidate-path actions)
│   │   ├── link_model.py        # LinkModel ABC — physical/logical link-quality interface
│   │   └── topology.py          # generic weighted-graph builder (moved from kavach/topology.py,
│   │                            #   NSFNET/UBN24/ring builders unchanged — see note below)
│   │
│   ├── domains/
│   │   └── qkd/                 # QKD is now a PLUGIN implementing core's interfaces,
│   │       ├── __init__.py      #   not the framework itself
│   │       ├── qkd_channel.py           # unchanged from kavach/qkd_channel.py
│   │       ├── coexistence.py           # unchanged
│   │       ├── attack_models.py         # unchanged
│   │       ├── baselines.py             # unchanged
│   │       ├── key_assignment.py        # unchanged
│   │       └── qkd_link_model.py        # NEW — adapts qkd_channel.py to core.LinkModel
│   │
│   ├── envs/                    # concrete Gymnasium envs (today: QKD only)
│   │   ├── __init__.py
│   │   ├── qkd_routing_env.py           # = old rl_env.py
│   │   ├── qkd_routing_env_buffered.py  # = old rl_env_buffered.py
│   │   └── qkd_routing_env_gnn.py       # = old rl_env_gnn.py
│   │
│   ├── policies/
│   │   ├── __init__.py
│   │   └── gnn_extractor.py     # = old gnn_extractor.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # = old trainer.py (MLP policy)
│   │   └── train_gnn.py         # = old train_gnn.py
│   │
│   ├── workhandler/
│   │   ├── __init__.py
│   │   ├── handler.py           # unchanged
│   │   ├── cli.py               # unchanged
│   │   └── mcp_server.py        # NEW — exposes handler.py via MCP for Claude Desktop
│   │
│   └── viz/
│       ├── __init__.py
│       ├── export_trace.py      # unchanged
│       └── render_showcase.py   # unchanged
│
├── examples/
│   └── cifar_classifier/        # the CNN exercise — explicitly separated: NOT part of
│       ├── train_cifarnet.py    #   the QKD framework, kept only as a standalone,
│       └── eval_confusion.py    #   documented "how training/eval loops work" reference
│
├── scripts/                     # thin CLI wrappers people actually run
│   ├── train_mlp.sh
│   ├── train_gnn.sh
│   └── smoke_test.sh
│
├── data/                        # gitignored — CIFAR/torch downloads land here
├── checkpoints/                 # gitignored
├── runs/                        # gitignored — workhandler run logs
├── workhandler.db               # gitignored
│
└── tests/
    └── (placeholder — see note below)
```

## Notes on scope of this pass

- **Files marked "unchanged"** are moved via `git mv`, not edited. Their logic,
  imports of each other, and behavior are untouched — this pass is a
  reorganization, not a rewrite, for everything already proven to work.
- **`core/base_env.py` and `core/link_model.py` are new** — an abstraction layer
  the three existing envs *could* migrate onto, so a future non-QKD network
  (classical optical RWA, generic SDN routing, whatever) can reuse the
  sequential-demand / candidate-path-action / graph-observation machinery
  without copying rl_env.py's structure by hand again. The three existing
  envs are NOT rewritten to inherit from this in this pass — that's real
  regression risk on working, tested code that I can't run against your
  actual imports. Treat this as the extension point, not a completed
  migration.
- **`tests/` is a placeholder.** None of this has real unit tests yet — worth
  flagging honestly rather than pretending otherwise. Once the repo
  structure is stable, this is the natural next investment (pytest fixtures
  around `core/base_env.py` particularly, since that's the piece meant to
  be reused).
