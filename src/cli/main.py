from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.traceback import install as install_rich_traceback

from simple_rag_writer.config.loader import load_config
from simple_rag_writer.cli.plan import run_planning_mode
from simple_rag_writer.cli.run import run_automated_mode
from simple_rag_writer.cli.replay import run_replay_mode

console = Console()
install_rich_traceback(show_locals=False)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="srw",
    description="Simple Rag Writer (planning, run, and replay modes).",
  )

  parser.add_argument(
    "-c",
    "--config",
    required=True,
    help="Path to config.yaml",
  )

  subparsers = parser.add_subparsers(dest="command", required=True)

  plan_p = subparsers.add_parser("plan", help="Interactive planning chat")
  plan_p.add_argument(
    "-m",
    "--model",
    help="Initial model id (overrides default_model)",
  )

  run_p = subparsers.add_parser("run", help="Run YAML writing tasks")
  run_p.add_argument(
    "tasks",
    nargs="+",
    help="Paths or globs to YAML task files",
  )

  replay_p = subparsers.add_parser("replay", help="Replay a planning log turn")
  replay_p.add_argument(
    "--log",
    required=True,
    help="Path to planning log Markdown file",
  )
  replay_p.add_argument(
    "--turn",
    required=True,
    type=int,
    help="Turn index (1-based) to reconstruct",
  )
  replay_p.add_argument(
    "--show-prompt",
    action="store_true",
    help="Print reconstructed prompt to stdout",
  )
  replay_p.add_argument(
    "--run-model",
    help=(
      "Optional model id to re-run the reconstructed prompt against. "
      "If omitted, no model call is made."
    ),
  )

  return parser


def main(argv: list[str] | None = None) -> int:
  if argv is None:
    argv = sys.argv[1:]

  parser = build_parser()
  args = parser.parse_args(argv)

  cfg_path = Path(args.config)
  if not cfg_path.exists():
    console.print(f"[red]Config not found:[/red] {cfg_path}")
    return 1

  config = load_config(cfg_path)

  if args.command == "plan":
    return run_planning_mode(config, initial_model=args.model)
  if args.command == "run":
    return run_automated_mode(config, args.tasks)
  if args.command == "replay":
    return run_replay_mode(
      config,
      Path(args.log),
      args.turn,
      args.show_prompt,
      args.run_model,
    )

  parser.print_help()
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
