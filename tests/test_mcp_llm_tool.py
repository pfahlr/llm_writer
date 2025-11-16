from __future__ import annotations

from typing import Any, Dict

import pytest

from simple_rag_writer.config import PromptsFile, PromptDefinition, SkillLibrary
from simple_rag_writer.config.models import (
  AppConfig,
  LlmToolConfig,
  ModelConfig,
  ProviderConfig,
  SkillConfig,
)
from simple_rag_writer.mcp.llm_tool import LlmToolHandler


class _FakeRegistry:
  def __init__(self) -> None:
    self.calls: list[Dict[str, Any]] = []

  def complete(
    self,
    prompt: str,
    model_id: str | None = None,
    task_params=None,
    mcp_client=None,
    system_prompt: str | None = None,
  ) -> str:
    self.calls.append(
      {
        "prompt": prompt,
        "model_id": model_id,
        "task_params": task_params,
        "system_prompt": system_prompt,
      }
    )
    return f"{model_id}:{prompt}"


def _make_prompts() -> PromptsFile:
  return PromptsFile(
    spec_version="1.0.0",
    validate_prompts=True,
    prompt_guidelines_url=None,
    prompts={
      "reasoner": PromptDefinition(
        id="reasoner",
        label="Reasoned Analyst",
        description="Performs analytical reasoning.",
        tags=("reasoning",),
        category="analysis",
        model_hint=None,
        system_prompt="You are Reasoner Alpha.",
        template_vars=(),
      ),
      "summarizer": PromptDefinition(
        id="summarizer",
        label="Summarizer",
        description="Summarize content.",
        tags=("summary",),
        category="summaries",
        model_hint=None,
        system_prompt="You are Summary Guide.",
        template_vars=(),
      ),
    },
  )


def _make_config() -> AppConfig:
  return AppConfig(
    default_model="reason-model",
    providers={
      "local": ProviderConfig(type="openai", api_key="dummy"),
    },
    models=[
      ModelConfig(id="reason-model", provider="local", model_name="reason"),
      ModelConfig(id="summarize-model", provider="local", model_name="summary"),
      ModelConfig(
        id="summarize-model",
        provider="local",
        model_name="summary",
        system_prompt="Model summary default.",
      ),
    ],
    llm_tool=LlmToolConfig(
      id="llm",
      tool_name="llm-complete",
      title="LLM skill completions",
      description="Invoke a tuned model skill.",
      skills=[
        SkillConfig(
          id="reason",
          label="Reason",
          description="Reason about complex problems.",
          model_id="reason-model",
          prompt_id="reasoner",
          max_output_tokens=512,
          temperature=0.3,
        ),
        SkillConfig(
          id="summarize",
          label="Summarize",
          description="Summaries with defaults.",
          model_id="summarize-model",
          prompt_id="summarizer",
        ),
        SkillConfig(
          id="fallback",
          label="Fallback",
          description="Uses model-only prompt.",
          model_id="summarize-model",
          prompt_id=None,
        ),
      ],
      default_skill="reason",
      max_tokens_limit=512,
    ),
  )


def test_call_tool_uses_configured_skill_and_task_params() -> None:
  config = _make_config()
  registry = _FakeRegistry()
  handler = LlmToolHandler(config, prompts=_make_prompts(), registry=registry)

  result = handler.call_tool(
    {"prompt": "Explain trends", "skill": "summarize", "max_tokens": 128}
  )

  assert registry.calls == [
    {
      "prompt": "Explain trends",
      "model_id": "summarize-model",
      "task_params": {"max_tokens": 128},
      "system_prompt": "You are Summary Guide.",
    }
  ]
  assert result["structuredContent"]["items"][0]["metadata"]["model"] == "summarize-model"
  assert "summarize-model" in result["structuredContent"]["items"][0]["metadata"]["model"]
  assert result["content"][0]["text"].startswith("summarize-model:")


def test_call_tool_uses_default_skill_when_missing() -> None:
  config = _make_config()
  registry = _FakeRegistry()
  handler = LlmToolHandler(config, prompts=_make_prompts(), registry=registry)

  handler.call_tool({"prompt": "Describe future work"})

  assert registry.calls[0]["model_id"] == "reason-model"


def test_call_tool_rejects_unknown_skill() -> None:
  config = _make_config()
  handler = LlmToolHandler(config, prompts=_make_prompts(), registry=_FakeRegistry())

  with pytest.raises(ValueError):
    handler.call_tool({"prompt": "Test", "skill": "unknown"})


def test_list_tools_exposes_skill_enum() -> None:
  config = _make_config()
  handler = LlmToolHandler(config, prompts=_make_prompts(), registry=_FakeRegistry())

  tools = handler.list_tools()
  assert len(tools) == 1
  schema = tools[0]["inputSchema"]
  assert schema["properties"]["skill"]["enum"] == ["reason", "summarize", "fallback"]
  assert schema["properties"]["skill"]["default"] == "reason"


def test_call_tool_retries_with_feedback() -> None:
  class _FlakyRegistry(_FakeRegistry):
    def __init__(self) -> None:
      super().__init__()
      self.attempt = 0

    def complete(
      self,
      prompt: str,
      model_id: str | None = None,
      task_params=None,
      mcp_client=None,
      system_prompt: str | None = None,
    ) -> str:
      self.attempt += 1
      if self.attempt == 1:
        raise RuntimeError("bad format")
      return super().complete(
        prompt,
        model_id=model_id,
        task_params=task_params,
        mcp_client=mcp_client,
        system_prompt=system_prompt,
      )

  config = _make_config()
  registry = _FlakyRegistry()
  handler = LlmToolHandler(config, prompts=_make_prompts(), registry=registry)

  handler.call_tool({"prompt": "Need update", "skill": "reason"})

  assert registry.attempt == 2
  assert len(registry.calls) == 1
  prompt = registry.calls[0]["prompt"]
  assert "SYSTEM FEEDBACK" in prompt
  assert "bad format" in prompt


def test_call_tool_returns_error_after_retries() -> None:
  class _FailingRegistry(_FakeRegistry):
    def complete(
      self,
      prompt: str,
      model_id: str | None = None,
      task_params=None,
      mcp_client=None,
      system_prompt: str | None = None,
    ) -> str:
      raise RuntimeError("network timeout")

  config = _make_config()
  handler = LlmToolHandler(config, prompts=_make_prompts(), registry=_FailingRegistry())

  result = handler.call_tool({"prompt": "Retry please"})

  assert result["isError"] is True
  assert "network timeout" in result["content"][0]["text"]
  assert result["structuredContent"]["items"] == []


def test_call_tool_uses_model_system_prompt_when_prompt_missing() -> None:
  config = _make_config()
  registry = _FakeRegistry()
  prompts = _make_prompts()
  handler = LlmToolHandler(config, prompts=prompts, registry=registry)

  handler.call_tool({"prompt": "Fallback call", "skill": "fallback"})

  assert registry.calls[0]["system_prompt"] == "Model summary default."


def test_call_tool_applies_skill_default_params() -> None:
  config = _make_config()
  registry = _FakeRegistry()
  prompts = _make_prompts()
  handler = LlmToolHandler(config, prompts=prompts, registry=registry)

  handler.call_tool({"prompt": "Reason with defaults", "skill": "reason"})

  assert registry.calls[0]["task_params"] == {"max_tokens": 512, "temperature": 0.3}
