#!/bin/bash
# migrate_structure.sh — reorganizes Kavach into the src/kavach/ layout in
# STRUCTURE.md. Uses `git mv` throughout so file history is preserved and
# the whole thing is a normal, revertable commit (or `git reset --hard` if
# something looks wrong before you commit).
#
# SAFETY: run `git status` first — this expects a clean or near-clean tree.
# Run from the Kavach repo root.

set -euo pipefail

if [ ! -d .git ]; then
  echo "error: run this from the Kavach repo root (no .git found here)"; exit 1
fi

echo "=== creating new directory structure ==="
mkdir -p src/kavach/core src/kavach/domains/qkd src/kavach/envs \
         src/kavach/policies src/kavach/training src/kavach/workhandler \
         src/kavach/viz examples/cifar_classifier scripts tests

touch src/kavach/__init__.py src/kavach/core/__init__.py \
      src/kavach/domains/__init__.py src/kavach/domains/qkd/__init__.py \
      src/kavach/envs/__init__.py src/kavach/policies/__init__.py \
      src/kavach/training/__init__.py src/kavach/workhandler/__init__.py \
      src/kavach/viz/__init__.py

echo "=== moving QKD domain files (physics — unchanged) ==="
git mv kavach/qkd_channel.py     src/kavach/domains/qkd/qkd_channel.py
git mv kavach/coexistence.py     src/kavach/domains/qkd/coexistence.py
git mv kavach/attack_models.py   src/kavach/domains/qkd/attack_models.py
git mv kavach/baselines.py       src/kavach/domains/qkd/baselines.py
git mv kavach/key_assignment.py  src/kavach/domains/qkd/key_assignment.py

echo "=== moving topology (generic — unchanged) ==="
git mv kavach/topology.py src/kavach/core/topology.py

echo "=== moving envs ==="
git mv kavach/rl_env.py          src/kavach/envs/qkd_routing_env.py
git mv kavach/rl_env_buffered.py src/kavach/envs/qkd_routing_env_buffered.py
git mv kavach/rl_env_gnn.py      src/kavach/envs/qkd_routing_env_gnn.py

echo "=== moving policies ==="
git mv gnn_extractor.py src/kavach/policies/gnn_extractor.py

echo "=== moving training scripts ==="
git mv kavach/trainer.py src/kavach/training/trainer.py
git mv train_gnn.py      src/kavach/training/train_gnn.py

echo "=== moving workhandler ==="
git mv workhandler/handler.py src/kavach/workhandler/handler.py
git mv workhandler/cli.py     src/kavach/workhandler/cli.py
git mv workhandler/README.md  src/kavach/workhandler/README.md
rm -f workhandler/__init__.py  # replaced by src/kavach/workhandler/__init__.py above
rmdir workhandler 2>/dev/null || true

echo "=== moving visualization ==="
git mv kavach/export_trace.py  src/kavach/viz/export_trace.py
git mv render_showcase.py      src/kavach/viz/render_showcase.py

echo "=== moving the CIFAR exercise into examples/ (explicitly not part of the QKD framework) ==="
git mv train_cifarnet.py  examples/cifar_classifier/train_cifarnet.py
git mv eval_confusion.py  examples/cifar_classifier/eval_confusion.py

echo "=== removing __pycache__ / leftover empty dirs ==="
find . -name "__pycache__" -type d -not -path "./.git/*" -exec git rm -r --cached -q {} + 2>/dev/null || true
find . -name "__pycache__" -type d -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
# The original kavach/ dir's __init__.py was never targeted by a git mv above
# (everything IN it was moved out individually) — it's leftover, and if left
# behind it silently shadows the real package: Python puts cwd first on
# sys.path, so running anything from the repo root would import this stale,
# near-empty kavach/ instead of src/kavach/, and every subpackage import
# would fail with a confusing "No module named kavach.envs" even though the
# real files are right there. Remove it outright — everything worth keeping
# was already git mv'd out above.
if [ -d kavach ]; then
  git rm -r --cached -q kavach 2>/dev/null || true
  rm -rf kavach
fi

echo ""
echo "=== done. NOT yet committed — review before committing: ==="
git status
echo ""
echo "Next steps (manual, on purpose):"
echo "  1. Fix internal imports — every moved file's 'from .xxx import yyy' or"
echo "     'from kavach.xxx import yyy' needs updating to the new module paths."
echo "     This script does NOT rewrite import statements — see the accompanying"
echo "     fix_imports.py for an automated first pass, then check by hand."
echo "  2. python3 -m pip install -e .   (after pyproject.toml is in place)"
echo "  3. Run smoke tests before committing: scripts/smoke_test.sh"
echo "  4. git add -A && git commit -m 'restructure: src/kavach layout'"
