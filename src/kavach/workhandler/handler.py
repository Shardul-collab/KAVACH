"""
workhandler.handler — core Work Handler logic, scoped to Kavach.

Research-grade job tracking for long-running training/export runs:

- submit_job() launches a command as a fully-detached background process
  and returns a run_id immediately. It does NOT wait() on the child, so
  the exact silent-failure class from `wait $PID` (job already exited by
  the time wait() was called, nothing surfaced it) is structurally
  impossible here — there is no wait() in this design at all.
- Instead, the launched command is wrapped in a shell snippet that writes
  its own exit code to a file once it finishes. Status is derived at query
  time from that file (or PID liveness if the file doesn't exist yet),
  never trusted as a stale DB column on its own.
- Config, git state (or an honest fallback when there's no git repo),
  dependency versions, and the exact command are captured at submission
  time — enough to answer "could someone else reproduce this run".
- Artifacts are found by diffing the output directory's contents before
  and after the run, hashed, and linked to the run_id that produced them —
  no filename convention to remember or mistype.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Optional

# --- paths -------------------------------------------------------------

_HANDLER_DIR = Path(__file__).resolve().parent
DB_PATH = _HANDLER_DIR.parent / "workhandler.db"
RUNS_DIR = _HANDLER_DIR.parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    label TEXT,
    command TEXT NOT NULL,
    config_json TEXT NOT NULL,
    cwd TEXT NOT NULL,
    git_commit TEXT,
    git_dirty INTEGER,
    repo_state_note TEXT,
    source_hash TEXT,
    python_version TEXT,
    key_deps_json TEXT,
    pid INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    exit_code INTEGER,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    stdout_log_path TEXT NOT NULL,
    stderr_log_path TEXT NOT NULL,
    exitcode_path TEXT NOT NULL,
    output_dir TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    kind TEXT,
    registered_at TEXT NOT NULL,
    UNIQUE(run_id, path, content_hash)
);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with closing(_conn()) as conn, conn:
        conn.executescript(SCHEMA)


_init_db()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _git_state(cwd: Path) -> dict:
    def run(args):
        try:
            return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=5)
        except Exception:
            return None

    rev = run(["git", "rev-parse", "HEAD"])
    if rev is None or rev.returncode != 0:
        return {
            "git_commit": None,
            "git_dirty": None,
            "repo_state_note": "not a git repository at submit time — "
                                "reproducibility relies on source_hash only",
        }

    commit = rev.stdout.strip()
    status = run(["git", "status", "--porcelain"])
    dirty = bool(status.stdout.strip()) if status is not None else None
    if dirty is True:
        note = "uncommitted changes present at submit time — source_hash may not match this commit"
    elif dirty is False:
        note = "clean working tree"
    else:
        note = "could not determine working tree state"
    return {"git_commit": commit, "git_dirty": dirty, "repo_state_note": note}


def _source_hash(cwd: Path, package_dir: str = "kavach") -> Optional[str]:
    """
    Fallback (and cross-check) reproducibility signal: a stable hash of
    every .py file's content under package_dir. Matters most when there's
    no git repo at all, but is cheap enough to always compute.
    """
    pkg = cwd / package_dir
    if not pkg.exists():
        return None
    h = hashlib.sha256()
    for p in sorted(pkg.rglob("*.py")):
        h.update(str(p.relative_to(cwd)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _key_deps() -> dict:
    import importlib.metadata as im
    deps = {}
    for name in ["torch", "stable-baselines3", "gymnasium", "numpy"]:
        try:
            deps[name] = im.version(name)
        except im.PackageNotFoundError:
            deps[name] = None
    return deps


_INTERNAL_PATHS = {DB_PATH, DB_PATH.with_suffix(".db-journal"),
                   DB_PATH.with_suffix(".db-wal"), DB_PATH.with_suffix(".db-shm")}


def _snapshot_dir(output_dir: Path) -> dict[str, str]:
    """
    Maps path -> sha256 content hash for every file in output_dir.

    Deliberately NOT mtime/size based: on WSL's /mnt/c (DrvFs via the 9p
    protocol), stat() timestamps are not reliably precise between two
    calls, so unchanged files can spuriously appear "changed". Content
    hashing costs more per run but is the only correct signal here.
    """
    snap = {}
    if not output_dir.exists():
        return snap
    for p in output_dir.iterdir():
        if not p.is_file():
            continue
        if p.resolve() in _INTERNAL_PATHS or RUNS_DIR in p.resolve().parents:
            continue  # never register the handler's own db/log files as a run's artifact
        try:
            snap[str(p)] = _sha256(p)
        except OSError:
            continue  # file disappeared/locked mid-scan; skip rather than crash the run
    return snap


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


KIND_RULES = [
    (lambda name: name.startswith(("ppo_", "dqn_")), "checkpoint"),
    (lambda name: name == "trace_export.json", "trace"),
    (lambda name: name.endswith(".log"), "log"),
    (lambda name: name.endswith(".html"), "report"),
]


def _classify(name: str) -> str:
    for pred, kind in KIND_RULES:
        if pred(name):
            return kind
    return "other"


# --- public API ----------------------------------------------------------

def submit_job(command: list[str], config: dict, output_dir: str = ".",
               label: Optional[str] = None) -> str:
    """
    Launch `command` as a detached background process. Returns a run_id
    immediately without blocking on completion.

    No wait() is ever called on the child from this process. The command is
    wrapped so it writes its own exit code to a file when done; status is
    reconstructed from that file (or PID liveness) at query time — see
    reconcile_status(). This is what replaces `nohup ... & echo $PID` plus
    a manually-typed `wait $PID`.
    """
    run_id = uuid.uuid4().hex[:12]
    cwd = Path.cwd()
    out_dir = Path(output_dir).resolve()

    stdout_path = RUNS_DIR / f"{run_id}.stdout.log"
    stderr_path = RUNS_DIR / f"{run_id}.stderr.log"
    exitcode_path = RUNS_DIR / f"{run_id}.exitcode"

    git_info = _git_state(cwd)
    src_hash = _source_hash(cwd)
    deps = _key_deps()

    pre_snapshot = _snapshot_dir(out_dir)

    inner_cmd = " ".join(shlex.quote(c) for c in command)
    wrapped = (
        f"{{ {inner_cmd} ; }} >'{stdout_path}' 2>'{stderr_path}'; "
        f"echo $? > '{exitcode_path}'"
    )

    proc = subprocess.Popen(
        ["/bin/sh", "-c", wrapped],
        cwd=str(cwd),
        start_new_session=True,  # detach: survives this process exiting, no controlling tty
    )

    with closing(_conn()) as conn, conn:
        conn.execute(
            """INSERT INTO runs
               (run_id, label, command, config_json, cwd, git_commit, git_dirty,
                repo_state_note, source_hash, python_version, key_deps_json, pid,
                status, started_at, stdout_log_path, stderr_log_path, exitcode_path,
                output_dir)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, label, inner_cmd, json.dumps(config), str(cwd),
             git_info["git_commit"], git_info["git_dirty"], git_info["repo_state_note"],
             src_hash, sys.version, json.dumps(deps), proc.pid, "running", _now(),
             str(stdout_path), str(stderr_path), str(exitcode_path), str(out_dir)),
        )

    (RUNS_DIR / f"{run_id}.pre_snapshot.json").write_text(json.dumps(pre_snapshot))
    return run_id


def _register_artifacts(run_id: str) -> None:
    with closing(_conn()) as conn:
        row = conn.execute("SELECT output_dir FROM runs WHERE run_id=?", (run_id,)).fetchone()
    out_dir = Path(row["output_dir"])
    snap_path = RUNS_DIR / f"{run_id}.pre_snapshot.json"
    before = json.loads(snap_path.read_text()) if snap_path.exists() else {}
    after = _snapshot_dir(out_dir)  # path -> content_hash

    # Only genuinely new or content-changed files, never mtime-only differences.
    changed = {p: h for p, h in after.items() if before.get(p) != h}

    with closing(_conn()) as conn, conn:
        for p, digest in changed.items():
            path = Path(p)
            if not path.exists():
                continue
            conn.execute(
                """INSERT OR IGNORE INTO artifacts
                   (artifact_id, run_id, path, content_hash, size_bytes, kind, registered_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (uuid.uuid4().hex[:12], run_id, p, digest, path.stat().st_size,
                 _classify(path.name), _now()),
            )


def reconcile_status(run_id: str) -> dict:
    """
    The single source of truth for a run's status. Always re-derives it
    from the exitcode file / PID liveness rather than trusting the stored
    `status` column — this is the direct fix for SHANI's "multiple Stage
    rows, unclear which reflects true state" failure mode, applied here.
    """
    with closing(_conn()) as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"no such run: {run_id}")
    row = dict(row)

    if row["status"] in ("completed", "failed"):
        return row

    exitcode_path = Path(row["exitcode_path"])
    if exitcode_path.exists():
        code = int(exitcode_path.read_text().strip())
        new_status = "completed" if code == 0 else "failed"
        with closing(_conn()) as conn, conn:
            conn.execute("UPDATE runs SET status=?, exit_code=?, ended_at=? WHERE run_id=?",
                         (new_status, code, _now(), run_id))
        _register_artifacts(run_id)
        row.update(status=new_status, exit_code=code, ended_at=_now())
        return row

    if _pid_alive(row["pid"]):
        return row  # still running

    # PID is dead but no exitcode file exists: the wrapper shell itself was
    # killed before it could write one (OOM-killed, machine slept, etc).
    # This is reported honestly as "unknown", never silently as success.
    with closing(_conn()) as conn, conn:
        conn.execute("UPDATE runs SET status='unknown', ended_at=? WHERE run_id=?",
                     (_now(), run_id))
    row.update(status="unknown", ended_at=_now())
    return row


def get_run_status(run_id: str) -> dict:
    return reconcile_status(run_id)


def get_run_config(run_id: str) -> dict:
    with closing(_conn()) as conn:
        row = conn.execute(
            """SELECT config_json, command, git_commit, git_dirty, repo_state_note,
                      source_hash, python_version, key_deps_json
               FROM runs WHERE run_id=?""", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"no such run: {run_id}")
    d = dict(row)
    d["config"] = json.loads(d.pop("config_json"))
    d["key_deps"] = json.loads(d.pop("key_deps_json"))
    return d


def list_artifacts(run_id: str) -> list[dict]:
    reconcile_status(run_id)  # ensure completion (and artifact registration) isn't stale
    with closing(_conn()) as conn:
        rows = conn.execute("SELECT * FROM artifacts WHERE run_id=? ORDER BY registered_at",
                             (run_id,)).fetchall()
    return [dict(r) for r in rows]


def list_runs(filter_dict: Optional[dict] = None) -> list[dict]:
    """
    filter_dict supports equality filters on top-level run columns
    (e.g. {"status": "completed"}) and dotted lookups into config, with
    optional __gt / __lt suffixes, e.g.
    {"config.topology": "nsfnet", "config.classical_traffic_fraction__gt": 0}
    """
    filter_dict = filter_dict or {}
    with closing(_conn()) as conn:
        ids = [r["run_id"] for r in conn.execute("SELECT run_id, status FROM runs").fetchall()
               if r["status"] not in ("completed", "failed")]
    for rid in ids:
        reconcile_status(rid)  # refresh anything non-terminal before listing

    with closing(_conn()) as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC").fetchall()]
    for r in rows:
        r["config"] = json.loads(r["config_json"])

    def matches(r, key, val):
        if key.startswith("config."):
            field = key[len("config."):]
            if field.endswith("__gt"):
                return r["config"].get(field[:-4], float("-inf")) > val
            if field.endswith("__lt"):
                return r["config"].get(field[:-4], float("inf")) < val
            return str(r["config"].get(field)) == str(val)
        return str(r.get(key)) == str(val)

    for k, v in filter_dict.items():
        rows = [r for r in rows if matches(r, k, v)]
    return rows
