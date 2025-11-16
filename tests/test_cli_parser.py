import argparse
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
