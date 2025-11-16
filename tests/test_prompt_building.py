from __future__ import annotations

from simple_rag_writer.prompts.planning import build_planning_prompt
from simple_rag_writer.prompts.task_prompt import build_task_prompt
from simple_rag_writer.tasks.models import TaskSpec


def test_planning_prompt_includes_sections_and_limits_history() -> None:
  history = [(f"user {i}", f"assistant {i}") for i in range(1, 8)]
  prompt = build_planning_prompt(
    history=history,
    user_message="Help me plan a task about research summaries.",
    mcp_context="Context from MCP server",
  )

  assert "You are a planning assistant" in prompt
  assert "## Context from external sources" in prompt
  assert "Context from MCP server" in prompt
  assert "## Conversation so far" in prompt
  assert "## New message" in prompt
  assert "Help me plan a task about research summaries." in prompt

  assert "User: user 1" not in prompt
  assert "Assistant: assistant 1" not in prompt
  assert "User: user 2" not in prompt
  assert "Assistant: assistant 2" not in prompt

  for i in range(3, 8):
    assert f"User: user {i}" in prompt
    assert f"Assistant: assistant {i}" in prompt


def test_task_prompt_includes_outline_style_and_references() -> None:
  task = TaskSpec(
    title="Write section on evaluation methods",
    id="eval-methods",
    description="Summarize evaluation approaches for retrieval-augmented generation.",
    output="drafts/eval.md",
    style="Friendly but precise",
  )
  outline_context = "Focus on comparison of automatic metrics vs. qualitative review."
  reference_blobs = [
    "Reference A details background on benchmarks.",
    "Reference B includes expert interview quotes.",
  ]

  prompt = build_task_prompt(task, outline_context, reference_blobs)

  assert "# Task: Write section on evaluation methods" in prompt
  assert "## Description" in prompt
  assert task.description in prompt
  assert "Style: Friendly but precise" in prompt
  assert "## Outline context" in prompt
  assert outline_context in prompt
  assert "## Reference material" in prompt
  for idx, blob in enumerate(reference_blobs, start=1):
    assert f"### Reference {idx}" in prompt
    assert blob in prompt
  assert "## Instructions" in prompt
  assert "Write a complete draft in Markdown" in prompt
