"""
mcp_server.py — exposes kavach.workhandler.handler as MCP tools, so Claude
Desktop (or Claude Code, or any other MCP client) can submit training runs,
check status, and inspect artifacts directly through natural language,
grounded in the actual SQLite-backed run registry rather than Claude's
memory of the conversation.

This was surface #2 of the original Work Handler design (see the
"Next Session Prompt" doc this whole project started from): "Standalone MCP
server: usable outside BRAHM entirely... exposing tools like submit_job,
get_run_status, list_artifacts, get_run_config." This is that.

Run directly for local testing:
    python3 mcp_server.py

Register with Claude Desktop by adding an entry to claude_desktop_config.json
(see claude_desktop_config.example.json alongside this file) pointing at this
script with an absolute path and the right working directory.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure `kavach` is importable whether or not this is pip-installed;
# falls back to a path insert if run straight out of the repo.
try:
    from kavach.workhandler import handler
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from kavach.workhandler import handler

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(
    name="kavach-workhandler",
    instructions=(
        "Tools for submitting and tracking Kavach training/simulation runs. "
        "Always check get_run_status before assuming a run finished — status "
        "is derived fresh each call, not cached. submit_job launches in the "
        "background and returns immediately; poll get_run_status to know when "
        "it's done."
    ),
)


@mcp.tool()
def submit_job(command: list[str], config: dict[str, object], output_dir: str = ".",
                label: str | None = None) -> str:
    """
    Launch a command as a background job and track it. Returns a run_id
    immediately — this does NOT wait for the job to finish.

    Args:
        command: the command to run, as a list of argv tokens, e.g.
            ["python3", "train_gnn.py", "--topology", "nsfnet", "--timesteps", "50000"]
        config: resolved config to record against this run, e.g.
            {"topology": "nsfnet", "timesteps": "50000"} — used later for
            filtering with list_runs.
        output_dir: directory to watch for artifacts this run produces
            (defaults to the current directory).
        label: optional human-readable label for this run.

    Returns:
        run_id (str) — use this with get_run_status / list_artifacts / get_run_config.
    """
    return handler.submit_job(command, config, output_dir=output_dir, label=label)


@mcp.tool()
def get_run_status(run_id: str) -> dict[str, object]:
    """
    Get a run's current status, re-derived fresh (never a stale cached flag).
    Status is one of: running, completed, failed, unknown ("unknown" means
    the process died without writing an exit code — report this honestly,
    don't guess whether it succeeded).
    """
    return handler.get_run_status(run_id)


@mcp.tool()
def list_runs(status: str | None = None, config_filter: dict[str, object] | None = None) -> list[dict[str, object]]:
    """
    List tracked runs, most recent first. Optionally filter by status
    (e.g. "completed", "running") and/or by resolved config values, e.g.
    config_filter={"topology": "nsfnet"} to find every run trained on NSFNET.
    """
    filt = {}
    if status:
        filt["status"] = status
    if config_filter:
        for k, v in config_filter.items():
            filt[f"config.{k}"] = v
    return handler.list_runs(filt)


@mcp.tool()
def list_artifacts(run_id: str) -> list[dict[str, object]]:
    """
    List files a run produced (checkpoints, traces, logs, reports), each
    with a content hash and size. Automatically re-checks whether the run
    has completed first, so this is safe to call even immediately after
    submit_job without a status check in between.
    """
    return handler.list_artifacts(run_id)


@mcp.tool()
def get_run_config(run_id: str) -> dict[str, object]:
    """
    Get everything needed to know what a run actually was: the resolved
    config, the exact command, git commit (or source hash if there was no
    git repo at submit time), Python version, and key dependency versions.
    """
    return handler.get_run_config(run_id)


if __name__ == "__main__":
    mcp.run()
