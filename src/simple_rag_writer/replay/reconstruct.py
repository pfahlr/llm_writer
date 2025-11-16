from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from simple_rag_writer.llm.registry import ModelRegistry
from simple_rag_writer.prompts.planning import DEFAULT_HISTORY_WINDOW, build_planning_prompt


@dataclass
class ReplayMeta:
  turn_index: int
  user_text: str
  history_turns: Tuple[int, ...]
  mcp_reference_count: int = 0


@dataclass
class _TurnRecord:
  index: int
  user_text: str
  assistant_text: str
  mcp_yaml: Optional[str] = None


def reconstruct_prompt_for_turn(log_path: Path, turn_index: int) -> Tuple[str, ReplayMeta]:
  if turn_index < 1:
    raise ValueError("turn_index must be >= 1")

  turns = _parse_turns(log_path.read_text(encoding="utf-8"))
  if not turns:
    raise ValueError("No turns found in log")

  target = next((t for t in turns if t.index == turn_index), None)
  if target is None:
    raise ValueError(f"Turn {turn_index} not found in log")

  prior_turns = [t for t in turns if t.index < turn_index]
  history_slice = prior_turns[-DEFAULT_HISTORY_WINDOW:]
  history_pairs = [(t.user_text, t.assistant_text) for t in history_slice]
  mcp_context, reference_count = _prepare_mcp_context(target.mcp_yaml)

  prompt = build_planning_prompt(
    history_pairs,
    target.user_text,
    mcp_context,
    history_window=DEFAULT_HISTORY_WINDOW,
    mcp_query_history=None,
  )
  meta = ReplayMeta(
    turn_index=turn_index,
    user_text=target.user_text,
    history_turns=tuple(t.index for t in history_slice),
    mcp_reference_count=reference_count,
  )
  return prompt, meta


def run_replay_prompt(registry: ModelRegistry, prompt: str, meta: ReplayMeta) -> str:
  return registry.complete(prompt)


def _parse_turns(text: str) -> List[_TurnRecord]:
  turns: List[_TurnRecord] = []
  current: Optional[dict] = None
  block: Optional[str] = None
  in_fence = False
  in_mcp_section = False

  def finalize() -> None:
    nonlocal current
    if not current:
      return
    turns.append(
      _TurnRecord(
        index=current["index"],
        user_text="\n".join(current["user"]).strip(),
        assistant_text="\n".join(current["assistant"]).strip(),
        mcp_yaml="\n".join(current["mcp"]).strip() or None,
      )
    )
    current = None

  for line in text.splitlines():
    stripped = line.strip()
    if stripped.startswith("## Turn "):
      finalize()
      index_text = stripped[len("## Turn ") :].strip()
      try:
        idx = int(index_text)
      except ValueError:
        continue
      current = {"index": idx, "user": [], "assistant": [], "mcp": []}
      block = None
      in_fence = False
      in_mcp_section = False
      continue

    if current is None:
      continue

    if stripped.startswith("**User:**"):
      block = "user"
      continue
    if stripped.startswith("**Assistant:**"):
      block = "assistant"
      continue
    if stripped == "### MCP References Injected":
      block = None
      in_mcp_section = True
      continue
    if stripped.startswith("```mcp-yaml"):
      block = "mcp"
      in_fence = True
      continue
    if in_fence and stripped.startswith("```"):
      in_fence = False
      block = None
      in_mcp_section = False
      continue

    if block == "mcp" and in_fence:
      current["mcp"].append(line)
      continue
    if block == "user":
      current["user"].append(line)
      continue
    if block == "assistant":
      current["assistant"].append(line)
      continue

    if in_mcp_section:
      continue

  finalize()
  return turns


def _prepare_mcp_context(mcp_yaml: Optional[str]) -> Tuple[Optional[str], int]:
  if not mcp_yaml:
    return None, 0

  raw = mcp_yaml.strip()
  if not raw:
    return None, 0

  try:
    payload = yaml.safe_load(raw)
  except yaml.YAMLError:
    return raw, 0

  normalized = yaml.safe_dump(payload, sort_keys=False).strip()
  ref_count = 0
  if isinstance(payload, dict):
    references = payload.get("references")
    if isinstance(references, list):
      ref_count = len(references)

  context_text = (normalized or raw).strip()
  if not context_text:
    return None, ref_count
  return context_text, ref_count
