from __future__ import annotations

from rich.console import Console

from simple_rag_writer.config import PromptsFile, load_prompts_config
from simple_rag_writer.config.models import AppConfig
from simple_rag_writer.llm.registry import ModelRegistry
from simple_rag_writer.logging.planning_log import PlanningLogWriter
from simple_rag_writer.planning.repl import PlanningRepl

_console = Console()


def _load_prompts_for_config(config: AppConfig) -> PromptsFile:
  cfg_path = config.config_path
  if not cfg_path:
    return PromptsFile.empty()
  prompts_path = cfg_path.with_name("prompts.yaml")
  if not prompts_path.exists():
    return PromptsFile.empty()
  try:
    return load_prompts_config(prompts_path)
  except Exception as exc:  # noqa: BLE001 - report but continue
    _console.print(f"[yellow]Failed to load prompts.yaml:[/yellow] {exc}")
    return PromptsFile.empty()


def run_planning_mode(config: AppConfig, initial_model: str | None = None) -> int:
  registry = ModelRegistry(config)
  if initial_model:
    registry.set_current(initial_model)

  log_writer = PlanningLogWriter.from_config(
    config,
    config_path=config.config_path,
    default_model_id=registry.current_id,
  )

  prompts = _load_prompts_for_config(config)

  repl = PlanningRepl(
    config=config,
    model_registry=registry,
    log_writer=log_writer,
    prompts=prompts,
  )
  repl.run()
  return 0
