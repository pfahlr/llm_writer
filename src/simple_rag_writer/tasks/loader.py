from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, List, Set

import yaml

from .models import TaskSpec


def load_task(path: Path) -> TaskSpec:
  with path.open("r", encoding="utf-8") as f:
    raw = yaml.safe_load(f) or {}
  return TaskSpec(**raw)


def expand_task_paths(paths: Iterable[Path]) -> List[Path]:
  ordered: List[Path] = []
  seen: Set[Path] = set()
  for p in paths:
    s = str(p)
    candidates = (
      sorted(Path(g) for g in glob.glob(s))
      if any(ch in s for ch in ["*", "?", "["])
      else [Path(p)]
    )
    for candidate in candidates:
      resolved = candidate.resolve()
      if resolved in seen:
        continue
      seen.add(resolved)
      ordered.append(resolved)
  return ordered
