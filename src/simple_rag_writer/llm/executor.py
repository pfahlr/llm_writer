from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from simple_rag_writer.llm.registry import ModelRegistry
from simple_rag_writer.mcp.client import McpClient


class LlmCompletionError(Exception):
  """Raised when a completion repeatedly fails and cannot be recovered."""

  def __init__(self, message: str):
    super().__init__(message)
    self.message = message


def _default_feedback_builder(base_prompt: str, error_message: str) -> str:
  sanitized_prompt = base_prompt.rstrip()
  sanitized_error = error_message.strip()
  feedback = (
    "SYSTEM FEEDBACK:\n"
    "The previous LLM completion attempt failed with the following error:\n"
    f"{sanitized_error}\n"
    "Please adjust your response formatting and retry."
  )
  return f"{sanitized_prompt}\n\n{feedback}" if sanitized_prompt else feedback


def run_completion_with_feedback(
  registry: ModelRegistry,
  prompt: str,
  *,
  model_id: Optional[str] = None,
  task_params: Optional[Dict[str, Any]] = None,
  mcp_client: Optional[McpClient] = None,
  max_attempts: int = 2,
  system_prompt: Optional[str] = None,
  feedback_builder: Callable[[str, str], str] = _default_feedback_builder,
  on_attempt_failure: Optional[Callable[[int, str, bool], None]] = None,
) -> str:
  """Run `ModelRegistry.complete` while feeding errors back into the prompt."""
  attempt = 0
  base_prompt = prompt
  current_prompt = prompt
  while True:
    attempt += 1
    try:
      return registry.complete(
        current_prompt,
        model_id=model_id,
        task_params=task_params,
        mcp_client=mcp_client,
        system_prompt=system_prompt,
      )
    except Exception as exc:  # noqa: BLE001 - we want to reframe any exception
      message = str(exc).strip() or exc.__class__.__name__
      should_retry = attempt < max_attempts
      if on_attempt_failure:
        on_attempt_failure(attempt, message, should_retry)
      if not should_retry:
        raise LlmCompletionError(message) from exc
      current_prompt = feedback_builder(base_prompt, message)
