"""
workhandler.cli — command-line entry point.

Usage:
    python3 -m workhandler.cli submit --label ppo-nsfnet --config topology=nsfnet \
        --config timesteps=50000 -- python3 -m kavach.trainer --topology nsfnet --timesteps 50000

    python3 -m workhandler.cli status <run_id>
    python3 -m workhandler.cli list [--status running] [--config topology=nsfnet]
    python3 -m workhandler.cli artifacts <run_id>
    python3 -m workhandler.cli config <run_id>
"""
import argparse
import json
import sys

from . import handler


def cmd_submit(args, command):
    if not command:
        print("error: give the command to run after `--`, e.g.\n"
              "  workhandler submit --label foo -- python3 -m kavach.trainer --timesteps 1000",
              file=sys.stderr)
        sys.exit(1)
    config = dict(kv.split("=", 1) for kv in (args.config or []))
    run_id = handler.submit_job(command, config, output_dir=args.output_dir, label=args.label)
    print(run_id)


def cmd_status(args, _):
    print(json.dumps(handler.get_run_status(args.run_id), indent=2, default=str))


def cmd_list(args, _):
    filt = {}
    if args.status:
        filt["status"] = args.status
    for kv in (args.config or []):
        k, v = kv.split("=", 1)
        filt[f"config.{k}"] = v
    runs = handler.list_runs(filt)
    if not runs:
        print("no runs match")
        return
    for r in runs:
        print(f"{r['run_id']}  {r['status']:<10} {(r['label'] or '-'):<20} "
              f"{r['started_at']}  {r['command'][:60]}")


def cmd_artifacts(args, _):
    print(json.dumps(handler.list_artifacts(args.run_id), indent=2, default=str))


def cmd_config(args, _):
    print(json.dumps(handler.get_run_config(args.run_id), indent=2, default=str))


def main():
    argv = sys.argv[1:]
    command = None
    if "--" in argv:
        idx = argv.index("--")
        command = argv[idx + 1:]
        argv = argv[:idx]

    ap = argparse.ArgumentParser(prog="workhandler")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--label")
    p_submit.add_argument("--output-dir", default=".")
    p_submit.add_argument("--config", action="append", help="key=value, repeatable")
    p_submit.set_defaults(func=cmd_submit)

    p_status = sub.add_parser("status")
    p_status.add_argument("run_id")
    p_status.set_defaults(func=cmd_status)

    p_list = sub.add_parser("list")
    p_list.add_argument("--status")
    p_list.add_argument("--config", action="append", help="key=value")
    p_list.set_defaults(func=cmd_list)

    p_art = sub.add_parser("artifacts")
    p_art.add_argument("run_id")
    p_art.set_defaults(func=cmd_artifacts)

    p_cfg = sub.add_parser("config")
    p_cfg.add_argument("run_id")
    p_cfg.set_defaults(func=cmd_config)

    args = ap.parse_args(argv)
    args.func(args, command)


if __name__ == "__main__":
    main()
