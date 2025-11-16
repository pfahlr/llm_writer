from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, List

import yaml

from .models import TaskSpec


def load_task(path: Path) -> TaskSpec:
  with path.open("r", encoding="utf-8") as f:
    raw = yaml.safe_load(f) or {}
  return TaskSpec(**raw)


def expand_task_paths(paths: Iterable[Path]) -> List[Path]:
  result: List[Path] = []
  for p in paths:
    s = str(p)
    if any(ch in s for ch in ["*", "?", "["]):
      for g in glob.glob(s):
        result.append(Path(g))
    else:
      result.append(p)
  unique = sorted({p.resolve() for p in result})
  return unique
