import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from simple_rag_writer.cli.main import build_parser, main


def _get_subparser_choices(parser: argparse.ArgumentParser) -> set[str]:
  """Helper to inspect configured subparsers in a stable way."""
  if not parser._subparsers:
    return set()
  # _subparsers stores a list with a single _SubParsersAction containing choices
  return set(parser._subparsers._group_actions[0].choices.keys())


def test_build_parser_registers_expected_subcommands():
  parser = build_parser()
  choices = _get_subparser_choices(parser)
  assert {"plan", "run", "replay"} <= choices


def test_main_reports_missing_config(capsys, tmp_path: Path):
  cfg_path = tmp_path / "missing.yaml"
  result = main(["-c", str(cfg_path), "plan"])
  assert result == 1
  out = capsys.readouterr().out
  assert "Config not found" in out


def test_srw_script_displays_help():
  script = Path(__file__).resolve().parents[1] / "srw"
  assert script.exists()

  proc = subprocess.run(
    [sys.executable, str(script), "--help"],
    capture_output=True,
    text=True,
  )
  assert proc.returncode == 0
  assert "Simple Rag Writer" in proc.stdout


def test_main_loads_dotenv_from_config_dir(monkeypatch, tmp_path: Path):
  env_path = tmp_path / ".env"
  env_path.write_text("DOTENV_KEY=from-dotenv\n", encoding="utf-8")

  cfg_path = tmp_path / "config.yaml"
  cfg_path.write_text(
    (
      "default_model: openai:gpt-4.1-mini\n"
      "providers:\n"
      "  openai:\n"
      "    type: openai\n"
      "    api_key_env: DOTENV_KEY\n"
      "models:\n"
      "  - id: openai:gpt-4.1-mini\n"
      "    provider: openai\n"
      "    model_name: gpt-4.1-mini\n"
    ),
    encoding="utf-8",
  )

  monkeypatch.delenv("DOTENV_KEY", raising=False)

  fake_planner_calls: dict[str, str | None] = {}

  def fake_run_planning_mode(*args, **kwargs):
    fake_planner_calls["env_var"] = os.environ.get("DOTENV_KEY")
    return 0

  monkeypatch.setattr(
    "simple_rag_writer.cli.main.run_planning_mode",
    fake_run_planning_mode,
  )

  offsite_dir = tmp_path / "elsewhere"
  offsite_dir.mkdir()
  monkeypatch.chdir(offsite_dir)

  result = main(["-c", str(cfg_path), "plan"])
  assert result == 0
  assert fake_planner_calls["env_var"] == "from-dotenv"
