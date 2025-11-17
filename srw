#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import litellm
litellm._turn_on_debug()

def _bootstrap_path() -> None:
  root = Path(__file__).resolve().parent
  src = root / "src"
  if str(src) not in sys.path:
    sys.path.insert(0, str(src))


def main() -> int:
  _bootstrap_path()
  from simple_rag_writer.cli.main import main as cli_main

  return cli_main()


if __name__ == "__main__":
  raise SystemExit(main())
