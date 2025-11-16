from __future__ import annotations

from simple_rag_writer.config.models import (
  AppConfig,
  McpPromptPolicy,
  ModelConfig,
  ProviderConfig,
  RawCappedPolicy,
  SummaryPolicy,
)
from simple_rag_writer.mcp.prompt_policy import apply_prompt_policy
from simple_rag_writer.mcp.types import NormalizedItem
from simple_rag_writer.tasks.models import McpReference


class FakeModelRegistry:
  def __init__(self, response: str = "SUMMARY OUTPUT") -> None:
    self.response = response
    self.calls = []

  def complete(self, prompt: str, model_id: str | None = None, task_params=None) -> str:
    self.calls.append(
      {
        "prompt": prompt,
        "model_id": model_id,
        "task_params": task_params,
      }
    )
    return self.response


def _make_config() -> AppConfig:
  return AppConfig(
    default_model="writer-model",
    providers={
      "local": ProviderConfig(type="openai", api_key="dummy"),
    },
    models=[
      ModelConfig(id="writer-model", provider="local", model_name="writer"),
      ModelConfig(id="summary-model", provider="local", model_name="summary"),
    ],
    mcp_prompt_policy=McpPromptPolicy(
      default_mode="raw_capped",
      raw_capped=RawCappedPolicy(
        max_items_per_reference=2,
        max_chars_per_item=5,
        max_total_chars=9,
      ),
      summary=SummaryPolicy(
        summarizer_model="summary-model",
        max_items_per_reference=3,
        summary_max_tokens=128,
        default_prompt="Default summary prompt.",
        per_type_prompts={
          "case": "Summarize the legal case evidence.",
        },
      ),
    ),
  )


def test_apply_prompt_policy_raw_capped_enforces_limits_and_defaults():
  config = _make_config()
  registry = FakeModelRegistry()
  reference = McpReference(server="notes", tool="search")
  items = [
    NormalizedItem(body="AAAAA11111"),
    NormalizedItem(body="BBBBB22222"),
    NormalizedItem(body="CCCCC33333"),
  ]

  text = apply_prompt_policy(config, items, reference, registry)

  assert text == "AAAAA\n\nBBBB"
  assert registry.calls == []


def test_apply_prompt_policy_summary_uses_per_type_prompt_and_limits_items():
  config = _make_config()
  registry = FakeModelRegistry(response="  SUMMARY TEXT  ")
  reference = McpReference(
    server="notes",
    tool="search",
    prompt_mode="summary",
    item_type="case",
    summary_max_tokens=42,
    max_items=1,
  )
  items = [
    NormalizedItem(title="Doc 1", body="Alpha body", url="https://example.com/a"),
    NormalizedItem(title="Doc 2", body="Beta body"),
  ]

  text = apply_prompt_policy(config, items, reference, registry)

  assert text == "SUMMARY TEXT"
  assert len(registry.calls) == 1
  call = registry.calls[0]
  assert call["model_id"] == "summary-model"
  assert call["task_params"] == {"max_tokens": 42}
  assert "Summarize the legal case evidence." in call["prompt"]
  assert "Doc 1" in call["prompt"]
  assert "Doc 2" not in call["prompt"]
