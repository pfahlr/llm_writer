from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from simple_rag_writer.llm.registry import ModelRegistry


@dataclass
class ReplayMeta:
  turn_index: int


def reconstruct_prompt_for_turn(log_path: Path, turn_index: int) -> Tuple[str, ReplayMeta]:
  """Very rough placeholder: returns full log as prompt."""
  text = log_path.read_text(encoding="utf-8")
  return text, ReplayMeta(turn_index=turn_index)


def run_replay_prompt(registry: ModelRegistry, prompt: str, meta: ReplayMeta) -> str:
  return registry.complete(prompt)
