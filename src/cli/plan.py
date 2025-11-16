from __future__ import annotations

from simple_rag_writer.llm.registry import ModelRegistry
from simple_rag_writer.planning.repl import PlanningRepl
from simple_rag_writer.logging.planning_log import PlanningLogWriter
from simple_rag_writer.config.models import AppConfig


def run_planning_mode(config: AppConfig, initial_model: str | None = None) -> int:
  registry = ModelRegistry(config)
  if initial_model:
    registry.set_current(initial_model)

  log_writer = PlanningLogWriter.from_config(
    config,
    config_path=config.config_path,
    default_model_id=registry.current_id,
  )

  repl = PlanningRepl(
    config=config,
    model_registry=registry,
    log_writer=log_writer,
  )
  repl.run()
  return 0
