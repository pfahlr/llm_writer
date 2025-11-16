from __future__ import annotations

from pathlib import Path
from typing import Iterable

from rich.console import Console

from simple_rag_writer.config.models import AppConfig
from simple_rag_writer.llm.registry import ModelRegistry
from simple_rag_writer.runner.run_tasks import run_tasks_for_paths

console = Console()


def run_automated_mode(config: AppConfig, task_paths: Iterable[str]) -> int:
  registry = ModelRegistry(config)
  task_paths_resolved = [Path(p) for p in task_paths]
  result = run_tasks_for_paths(config, registry, task_paths_resolved)
  if not result.ok:
    console.print("[red]Some tasks failed[/red]")
    return 1
  return 0
