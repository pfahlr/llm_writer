from __future__ import annotations

import sys
from typing import List

from simple_rag_writer.config.models import (
  AppConfig,
  McpServerConfig,
  ModelConfig,
  ProviderConfig,
)
from simple_rag_writer.mcp.client import McpClient


def _make_config(command: List[str]) -> AppConfig:
  return AppConfig(
    default_model="writer",
    providers={"local": ProviderConfig(type="openai", api_key="dummy")},
    models=[ModelConfig(id="writer", provider="local", model_name="Writer Model")],
    mcp_servers=[McpServerConfig(id="notes", command=command)],
  )


def _server_command() -> List[str]:
  return [sys.executable, "-m", "tests.mcp_fixtures.notes_server"]


def test_mcp_client_lists_tools() -> None:
  client = McpClient(_make_config(_server_command()))

  tools = client.list_tools("notes")

  names = {tool["name"] for tool in tools}
  assert "search" in names
  assert "recent" in names
  search_tool = next(tool for tool in tools if tool["name"] == "search")
  assert search_tool["description"]
  assert "synthetic notes" in search_tool["description"]


def test_mcp_client_calls_tool_and_returns_payload() -> None:
  client = McpClient(_make_config(_server_command()))

  result = client.call_tool("notes", "search", {"query": "outline ideas", "limit": 1})

  assert result.server_id == "notes"
  assert result.tool_name == "search"
  assert isinstance(result.payload, list)
  assert len(result.payload) == 1
  assert result.payload[0]["title"] == "Notebook entry"

