# Kavach — restructure + MCP connector

Everything needed to reorganize the existing, working repo into a clean,
publishable layout, plus a Work Handler connector for Claude Desktop.

## What's here

| File | What it does | Tested how |
|---|---|---|
| `STRUCTURE.md` | The target directory layout and rationale | — (reference doc) |
| `migrate_structure.sh` | Moves your existing files into that layout via `git mv` | Full dry run against reconstructed real file contents — see below |
| `fix_imports.py` | Rewrites the ~26 import statements that break after the move | Same dry run — all 26 matched and applied correctly |
| `pyproject.toml` | Makes the repo `pip install -e .`-able from the new `src/` layout | Verified: installed package imports cleanly from the repo root |
| `mcp_server.py` | Exposes the Work Handler (`submit_job`, `get_run_status`, `list_runs`, `list_artifacts`, `get_run_config`) to Claude Desktop via MCP | Full round-trip tested: real background job submitted and tracked through the actual MCP tool-call protocol layer, not just imported |
| `claude_desktop_config.example.json` | The config snippet to register the connector | Verified against current real-world working WSL+Claude Desktop configs (web search), not written from memory |
| `src_new/core/link_model.py`, `src_new/core/base_env.py` | Generic, domain-agnostic routing-RL base class | Tested end-to-end against a deliberately non-QKD dummy domain to confirm the abstraction is real |

## How I validated this without access to your actual machine

I reconstructed your real files (from what you'd pasted earlier in this
conversation) in a local mirror, ran the full migration → import-fix →
`pip install -e .` → import pipeline against it, and fixed two real bugs
that surfaced only by actually running it:

1. **`workhandler/README.md` missing** in an early test run caused the
   script to hard-fail with `set -e` rather than explain what was
   missing — not a script bug, just revealed the script wasn't defensive
   about optional files. (Your actual repo has this file, so this
   specifically won't hit you — but the underlying fragility was worth
   knowing about.)
2. **The real one**: the original `kavach/__init__.py` was never removed
   after everything inside `kavach/` was moved out. A stale, near-empty
   `kavach/` directory was left at the repo root, and because Python
   puts the current directory first on `sys.path`, running anything from
   the repo root silently imported the OLD stale package instead of the
   new `src/kavach/` — `kavach.envs` and everything else would fail with
   a confusing "module not found" that had nothing obviously to do with
   the real cause. Fixed in the script; confirmed fixed by re-running the
   exact scenario that broke.

What I could **not** test: the actual migration against your real
`Kavach/` directory, since I don't have access to your WSL filesystem —
only the reconstructed mirror. Run `git status` before and after, and if
anything looks different from what `STRUCTURE.md` describes, stop and
paste the output rather than proceeding.

## Steps to run this on your machine

```bash
cd /mnt/c/Users/deman/Desktop/Kavach

# 1. Copy these files into the repo root
#    (migrate_structure.sh, fix_imports.py, pyproject.toml, mcp_server.py,
#     claude_desktop_config.example.json, and src_new/ -> merge its
#     contents into src/core/ after migration, see step 4)

# 2. Confirm a clean-ish git state first
git status

# 3. Run the migration
bash migrate_structure.sh
# review the output — it stops before committing, on purpose

# 4. Fix imports
python3 fix_imports.py

# 5. Add the new generic base-env files (not part of migration — brand new)
mkdir -p src/kavach/core
cp src_new/core/link_model.py src/kavach/core/
cp src_new/core/base_env.py src/kavach/core/

# 6. Install in editable mode
pip install -e . --break-system-packages

# 7. Smoke-test the actual imports before committing anything
python3 -c "
import kavach.envs.qkd_routing_env
import kavach.envs.qkd_routing_env_buffered
import kavach.envs.qkd_routing_env_gnn
import kavach.training.trainer
import kavach.training.train_gnn
import kavach.viz.export_trace
import kavach.policies.gnn_extractor
import kavach.workhandler.handler
import kavach.workhandler.cli
print('ALL IMPORTS OK')
"

# 8. Only once step 7 passes clean, commit
git add -A && git commit -m "restructure: src/kavach layout + generic base env + MCP connector"

# 9. Set up the MCP connector
pip install mcp --break-system-packages
python3 mcp_server.py   # Ctrl+C once it's just sitting there waiting — confirms it starts without error

# 10. Register with Claude Desktop
#     - Open Claude Desktop -> Settings -> Developer -> Edit Config
#     - Merge in the "mcpServers" block from claude_desktop_config.example.json
#       (edit the path inside it if your repo isn't at
#       /mnt/c/Users/deman/Desktop/Kavach)
#     - Fully restart Claude Desktop (not just close the window)
```

## After that

Once the connector's registered, you should be able to ask Claude Desktop
things like "submit a GNN training run on nsfnet for 50k timesteps" or
"what's the status of my last run" and have it actually call these tools
against your real run registry — that was the original point of building
the Work Handler as an MCP server in the first place, back at the start of
this whole project.

## What's still explicitly NOT done here

- The three working env files are not migrated onto `core/base_env.py` —
  see `STRUCTURE.md`'s notes section for why.
- No unit tests exist yet (`tests/` is an empty placeholder in the
  structure doc).
- `pyproject.toml`'s dependency version floors are reasonable guesses, not
  pinned to exactly what's on your machine — worth tightening before
  anyone else tries to install this from a public GitHub repo.
- The MCP server only exposes the Work Handler. It doesn't let Claude
  Desktop directly poke at training internals (env state, live metrics) —
  that'd be a reasonable v2 if useful.
