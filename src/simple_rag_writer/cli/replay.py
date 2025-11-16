from __future__ import annotations

from pathlib import Path

from rich.console import Console

from simple_rag_writer.config.models import AppConfig
from simple_rag_writer.llm.registry import ModelRegistry
from simple_rag_writer.replay.reconstruct import (
  reconstruct_prompt_for_turn,
  run_replay_prompt,
)

console = Console()


def run_replay_mode(
  config: AppConfig,
  log_path: Path,
  turn_index: int,
  show_prompt: bool,
  run_model_id: str | None,
) -> int:
  prompt, meta = reconstruct_prompt_for_turn(log_path, turn_index)

  if show_prompt:
    console.rule(f"Reconstructed Prompt for Turn {turn_index}")
    console.print(prompt)

  if run_model_id:
    registry = ModelRegistry(config)
    registry.set_current(run_model_id)
    output = run_replay_prompt(registry, prompt, meta)
    console.rule("Model Output")
    console.print(output)

  return 0
