# Work Handler — Kavach MVP

Drop the `workhandler/` folder into your Kavach root, next to `kavach/`:

    Kavach/
    ├── kavach/
    ├── workhandler/        <- this
    │   ├── __init__.py
    │   ├── handler.py
    │   └── cli.py
    ├── trainer.py etc.

First run creates `workhandler.db` (SQLite) and a `runs/` directory
(per-run stdout/stderr/exitcode files) next to `workhandler/`. Add both
to `.gitignore`.

## Recommended first step

You're not in a git repo yet:

    git init && git add -A && git commit -m "baseline"

Without this, run records fall back to a source-hash of `kavach/*.py`
instead of a commit hash — still usable, but less precise, and it can't
tell you "this run predates that code change."

## Usage

Submit a real training run (note the `--` separator):

    python3 -m workhandler.cli submit \
      --label ppo-nsfnet-cf0.3 \
      --config topology=nsfnet --config timesteps=50000 --config classical_fraction=0.3 \
      -- python3 -m kavach.trainer --topology nsfnet --timesteps 50000 --classical-fraction 0.3

Returns a run_id immediately (does not block).

Check status any time — derived fresh each call, never a stale flag:

    python3 -m workhandler.cli status <run_id>

List runs, optionally filtered by config:

    python3 -m workhandler.cli list
    python3 -m workhandler.cli list --status running
    python3 -m workhandler.cli list --config topology=nsfnet
    python3 -m workhandler.cli list --config classical_fraction__gt=0   # via config.<field>__gt in Python API

See what a run produced:

    python3 -m workhandler.cli artifacts <run_id>

See exactly what was run and with what environment:

    python3 -m workhandler.cli config <run_id>

## What this does NOT do (by design, for this MVP)

- Does not touch trainer.py / export_trace.py / render_showcase.py.
  Artifacts are found by diffing the output directory before/after the
  run, not by trainer.py telling the handler what it wrote.
- Does not auto-retry, auto-kill, or auto-resume anything. It reports
  state; you decide what to do about it.
- Does not yet know about mid-run checkpoints (e.g. stable-baselines3's
  own periodic-save callback, if you add one later) — only what exists in
  the output directory at completion time. Resumability tracking (item 4
  in the original spec) is the natural next increment once this is
  proven on real runs.

## Known gap to fix in trainer.py separately (not part of this handler)

`ppo_model.save(f"ppo_{topology}_{env_type}_cf{classical_fraction}")` has
no run_id or timestamp — two runs with the same config will silently
overwrite each other's checkpoint on disk, whether or not the Work
Handler is tracking them. Worth a small follow-up patch to trainer.py
independent of this.
