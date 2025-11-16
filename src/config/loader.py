from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .models import AppConfig


def load_config(path: Path) -> AppConfig:
  with path.open("r", encoding="utf-8") as f:
    raw: Dict[str, Any] = yaml.safe_load(f) or {}
  cfg = AppConfig(**raw)
  return cfg.with_path(path)
