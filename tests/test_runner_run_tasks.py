from __future__ import annotations

from io import StringIO
from pathlib import Path
from textwrap import dedent

from rich.console import Console

from simple_rag_writer.config.models import (
  AppConfig,
  McpPromptPolicy,
  ModelConfig,
  ProviderConfig,
  RawCappedPolicy,
  SummaryPolicy,
)
from simple_rag_writer.mcp.types import McpToolResult
from simple_rag_writer.runner.run_tasks import run_tasks_for_paths


class DummyRegistry:
  def __init__(self) -> None:
    self.calls = []

  def complete(
    self,
    prompt: str,
    model_id: str | None = None,
    task_params=None,
    mcp_client=None,
  ) -> str:
    call = {
      "prompt": prompt,
      "model_id": model_id,
      "task_params": task_params,
    }
    self.calls.append(call)
    if model_id == "summary-model":
      return "URL SUMMARY"
    return "FINAL OUTPUT"


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
        max_items_per_reference=5,
        max_chars_per_item=200,
        max_total_chars=2000,
      ),
      summary=SummaryPolicy(
        summarizer_model="summary-model",
        default_prompt="Summarize the material.",
        summary_max_tokens=128,
      ),
    ),
  )


def _write_task(tmp_path: Path, body: str) -> Path:
  path = tmp_path / "task.yaml"
  path.write_text(dedent(body), encoding="utf-8")
  return path


def test_run_tasks_includes_url_reference_text(tmp_path, monkeypatch):
  config = _make_config()
  registry = DummyRegistry()

  class FakeMcpClient:
    def __init__(self, _config):
      self.calls = []

    def call_tool(self, server: str, tool: str, params: dict) -> McpToolResult:
      self.calls.append((server, tool, params))
      return McpToolResult(
        server_id=server,
        tool_name=tool,
        payload=[
          {
            "title": "Notebook entry",
            "body": "Alpha body",
          }
        ],
      )

  monkeypatch.setattr("simple_rag_writer.runner.run_tasks.McpClient", FakeMcpClient)
  monkeypatch.setattr(
    "simple_rag_writer.runner.run_tasks.fetch_url_text",
    lambda url: "URL page text for " + url,
  )

  output_path = tmp_path / "drafts" / "chapter.md"
  task_path = _write_task(
    tmp_path,
    f"""
    title: "Chapter"
    id: "ch01"
    description: "Write the chapter"
    output: "{output_path}"
    references:
      - type: "mcp"
        label: "Notes"
        server: "notes"
        tool: "search"
      - type: "url"
        label: "Spec"
        url: "https://example.com/spec"
        prompt_mode: "summary"
    """,
  )

  result = run_tasks_for_paths(config, registry, [task_path])

  assert result.ok is True
  assert len(registry.calls) == 2
  summary_call, final_call = registry.calls
  assert summary_call["model_id"] == "summary-model"
  assert "URL page text" in summary_call["prompt"]
  assert final_call["model_id"] == "writer-model"
  assert "URL SUMMARY" in final_call["prompt"]
  assert "Notebook entry" in final_call["prompt"]
  assert output_path.exists()
  assert output_path.read_text(encoding="utf-8") == "FINAL OUTPUT"


def test_run_tasks_warns_and_continues_on_url_error(tmp_path, monkeypatch):
  config = _make_config()
  registry = DummyRegistry()

  class FakeMcpClient:
    def __init__(self, _config):
      pass

    def call_tool(self, server: str, tool: str, params: dict) -> McpToolResult:
      return McpToolResult(
        server_id=server,
        tool_name=tool,
        payload=[{"title": "Doc", "body": "Reference body"}],
      )

  console_output = StringIO()
  monkeypatch.setattr("simple_rag_writer.runner.run_tasks.McpClient", FakeMcpClient)
  monkeypatch.setattr(
    "simple_rag_writer.runner.run_tasks.fetch_url_text",
    lambda url: (_ for _ in ()).throw(RuntimeError("boom")),
  )
  monkeypatch.setattr(
    "simple_rag_writer.runner.run_tasks.console",
    Console(file=console_output, force_terminal=False, color_system=None),
  )

  output_path = tmp_path / "drafts" / "chapter.md"
  task_path = _write_task(
    tmp_path,
    f"""
    title: "Chapter"
    id: "ch01"
    description: "Write the chapter"
    output: "{output_path}"
    references:
      - type: "url"
        label: "Spec"
        url: "https://example.com/spec"
      - type: "mcp"
        label: "Notes"
        server: "notes"
        tool: "search"
    """,
  )

  result = run_tasks_for_paths(config, registry, [task_path])

  assert result.ok is True
  assert len(registry.calls) == 1
  assert "Reference body" in registry.calls[0]["prompt"]
  log_text = console_output.getvalue()
  assert "URL" in log_text and "failed" in log_text.lower()
  assert output_path.read_text(encoding="utf-8") == "FINAL OUTPUT"
