from pathlib import Path
from typing import Optional

import yaml

from .outline_models import Outline


def load_outline(path: Path) -> Outline:
  """Load and validate an outline YAML file."""
  if not path.exists():
    raise FileNotFoundError(f"Outline file not found: {path}")

  with path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

  if not data:
    raise ValueError(f"Outline file is empty: {path}")

  return Outline(**data)


def load_outline_safe(path: Optional[Path]) -> Optional[Outline]:
  """Load outline with error handling; returns None on failure."""
  if not path:
    return None
  try:
    return load_outline(path)
  except Exception:  # noqa: BLE001
    return None
