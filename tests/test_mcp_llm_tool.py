from __future__ import annotations

from typing import Any, Dict

import pytest

from simple_rag_writer.config.models import (
  AppConfig,
  LlmToolConfig,
  ModelConfig,
  ProviderConfig,
)
from simple_rag_writer.mcp.llm_tool import LlmToolHandler


class _FakeRegistry:
  def __init__(self) -> None:
    self.calls: list[Dict[str, Any]] = []

  def complete(self, prompt: str, model_id: str | None = None, task_params=None, mcp_client=None) -> str:
    self.calls.append({"prompt": prompt, "model_id": model_id, "task_params": task_params})
    return f"{model_id}:{prompt}"


def _make_config() -> AppConfig:
  return AppConfig(
    default_model="reason-model",
    providers={
      "local": ProviderConfig(type="openai", api_key="dummy"),
    },
    models=[
      ModelConfig(id="reason-model", provider="local", model_name="reason"),
      ModelConfig(id="summarize-model", provider="local", model_name="summary"),
    ],
    llm_tool=LlmToolConfig(
      id="llm",
      tool_name="llm-complete",
      title="LLM skill completions",
      description="Invoke a tuned model skill.",
      skills={
        "reason": "reason-model",
        "summarize": "summarize-model",
      },
      default_skill="reason",
      max_tokens_limit=512,
    ),
  )


def test_call_tool_uses_configured_skill_and_task_params() -> None:
  config = _make_config()
  registry = _FakeRegistry()
  handler = LlmToolHandler(config, registry=registry)

  result = handler.call_tool(
    {"prompt": "Explain trends", "skill": "summarize", "max_tokens": 128}
  )

  assert registry.calls == [
    {"prompt": "Explain trends", "model_id": "summarize-model", "task_params": {"max_tokens": 128}}
  ]
  assert result["structuredContent"]["items"][0]["metadata"]["model"] == "summarize-model"
  assert "summarize-model" in result["structuredContent"]["items"][0]["metadata"]["model"]
  assert result["content"][0]["text"].startswith("summarize-model:")


def test_call_tool_uses_default_skill_when_missing() -> None:
  config = _make_config()
  registry = _FakeRegistry()
  handler = LlmToolHandler(config, registry=registry)

  handler.call_tool({"prompt": "Describe future work"})

  assert registry.calls[0]["model_id"] == "reason-model"


def test_call_tool_rejects_unknown_skill() -> None:
  config = _make_config()
  handler = LlmToolHandler(config, registry=_FakeRegistry())

  with pytest.raises(ValueError):
    handler.call_tool({"prompt": "Test", "skill": "unknown"})


def test_list_tools_exposes_skill_enum() -> None:
  config = _make_config()
  handler = LlmToolHandler(config, registry=_FakeRegistry())

  tools = handler.list_tools()
  assert len(tools) == 1
  schema = tools[0]["inputSchema"]
  assert schema["properties"]["skill"]["enum"] == ["reason", "summarize"]
  assert schema["properties"]["skill"]["default"] == "reason"


def test_call_tool_retries_with_feedback() -> None:
  class _FlakyRegistry(_FakeRegistry):
    def __init__(self) -> None:
      super().__init__()
      self.attempt = 0

    def complete(self, prompt: str, model_id: str | None = None, task_params=None, mcp_client=None) -> str:
      self.attempt += 1
      if self.attempt == 1:
        raise RuntimeError("bad format")
      return super().complete(prompt, model_id=model_id, task_params=task_params, mcp_client=mcp_client)

  config = _make_config()
  registry = _FlakyRegistry()
  handler = LlmToolHandler(config, registry=registry)

  handler.call_tool({"prompt": "Need update", "skill": "reason"})

  assert registry.attempt == 2
  assert len(registry.calls) == 1
  prompt = registry.calls[0]["prompt"]
  assert "SYSTEM FEEDBACK" in prompt
  assert "bad format" in prompt


def test_call_tool_returns_error_after_retries() -> None:
  class _FailingRegistry(_FakeRegistry):
    def complete(self, prompt: str, model_id: str | None = None, task_params=None, mcp_client=None) -> str:
      raise RuntimeError("network timeout")

  config = _make_config()
  handler = LlmToolHandler(config, registry=_FailingRegistry())

  result = handler.call_tool({"prompt": "Retry please"})

  assert result["isError"] is True
  assert "network timeout" in result["content"][0]["text"]
  assert result["structuredContent"]["items"] == []
