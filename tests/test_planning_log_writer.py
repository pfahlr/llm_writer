from __future__ import annotations

from pathlib import Path

import yaml

from simple_rag_writer.config.models import (
  AppConfig,
  LoggingConfig,
  ModelConfig,
  PlanningLoggingConfig,
  ProviderConfig,
)
from simple_rag_writer.logging.planning_log import McpLogItem, PlanningLogWriter


def _make_config(
  tmp_path: Path,
  *,
  include_mcp_events: bool = True,
  mcp_inline: bool = True,
) -> AppConfig:
  planning_cfg = PlanningLoggingConfig(
    enabled=True,
    dir=str(tmp_path / "logs"),
    include_mcp_events=include_mcp_events,
    mcp_inline=mcp_inline,
  )
  return AppConfig(
    default_model="writer-model",
    providers={"local": ProviderConfig(type="openai", api_key="dummy")},
    models=[
      ModelConfig(id="writer-model", provider="local", model_name="writer"),
      ModelConfig(id="editor-model", provider="local", model_name="editor"),
    ],
    logging=LoggingConfig(planning=planning_cfg),
  )


def _read_log_text(log_dir: Path) -> str:
  files = list((log_dir).glob("plan-*.md"))
  assert len(files) == 1, "expected exactly one planning log file"
  return files[0].read_text(encoding="utf-8")


def test_planning_log_writer_writes_header_and_turns(tmp_path: Path) -> None:
  config = _make_config(tmp_path)
  config_path = tmp_path / "config.yaml"
  writer = PlanningLogWriter.from_config(
    config,
    config_path=config_path,
    default_model_id="writer-model",
  )

  writer.log_model_used("writer-model")
  writer.log_model_used("editor-model")
  writer.start_turn(1, "Need help outlining the chapter.")
  writer.end_turn(1, "Let's start with an outline.")
  writer.close()

  log_text = _read_log_text(tmp_path / "logs")
  assert log_text.startswith("# Planning Session — ")
  assert f"- Config: {config_path}" in log_text
  assert "- Default model: writer-model" in log_text
  assert "- Models used: editor-model, writer-model" in log_text
  assert "## Turn 1" in log_text
  assert "**User:**" in log_text and "Need help outlining the chapter." in log_text
  assert "**Assistant:**" in log_text and "Let's start with an outline." in log_text


def test_planning_log_writer_logs_mcp_references_when_enabled(tmp_path: Path) -> None:
  config = _make_config(tmp_path, include_mcp_events=True, mcp_inline=True)
  writer = PlanningLogWriter.from_config(
    config,
    config_path=tmp_path / "cfg.yaml",
    default_model_id="writer-model",
  )

  items = [
    McpLogItem(
      idx=1,
      server="notes",
      tool="search",
      label="Spec",
      normalized_id="notes#1",
      title="Notebook entry",
      type="document",
      snippet="First few lines",
      body="Full body",
      url="https://example.com/spec",
      metadata={"score": 0.9},
    )
  ]
  writer.start_turn(1, "Use the spec.")
  writer.log_mcp_injection(1, items)
  writer.end_turn(1, "Referencing the spec now.")
  writer.close()

  log_text = _read_log_text(tmp_path / "logs")
  assert "### MCP References Injected" in log_text
  assert "| 1 | notes | search | Spec | notes#1 | Notebook entry |" in log_text

  fenced_block = log_text.split("```mcp-yaml", 1)[1].split("```", 1)[0]
  data = yaml.safe_load(fenced_block)
  assert data == {
    "references": [
      {
        "idx": 1,
        "server": "notes",
        "tool": "search",
        "label": "Spec",
        "normalized_id": "notes#1",
        "title": "Notebook entry",
        "type": "document",
        "snippet": "First few lines",
        "body": "Full body",
        "url": "https://example.com/spec",
        "metadata": {"score": 0.9},
      }
    ]
  }


def test_planning_log_writer_skips_mcp_logging_when_disabled(tmp_path: Path) -> None:
  config = _make_config(tmp_path, include_mcp_events=False, mcp_inline=True)
  writer = PlanningLogWriter.from_config(
    config,
    config_path=tmp_path / "cfg.yaml",
    default_model_id="writer-model",
  )
  writer.start_turn(1, "Hello")
  writer.log_mcp_injection(
    1,
    [
      McpLogItem(
        idx=1,
        server="notes",
        tool="search",
        label=None,
        normalized_id=None,
        title=None,
        type=None,
        snippet=None,
        body=None,
        url=None,
      )
    ],
  )
  writer.end_turn(1, "World")
  writer.close()

  log_text = _read_log_text(tmp_path / "logs")
  assert "MCP References Injected" not in log_text
