import json
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from simple_rag_writer.config.models import (
  AppConfig,
  ModelConfig,
  McpServerConfig,
  ProviderConfig,
)
from simple_rag_writer.llm.registry import ModelRegistry
from simple_rag_writer.mcp.types import McpToolResult


def test_model_registry_sets_and_switches_current_model():
  cfg = AppConfig(
    default_model="openai:gpt-4.1-mini",
    providers={
      "openai": ProviderConfig(type="openai", api_key="test-key"),
    },
    models=[
      ModelConfig(
        id="openai:gpt-4.1-mini",
        provider="openai",
        model_name="gpt-4.1-mini",
      ),
      ModelConfig(
        id="openai:gpt-4o-mini",
        provider="openai",
        model_name="gpt-4o-mini",
      ),
    ],
  )
  registry = ModelRegistry(cfg)
  assert registry.current_id == "openai:gpt-4.1-mini"
  registry.set_current("openai:gpt-4o-mini")
  assert registry.current_id == "openai:gpt-4o-mini"
  assert [m.id for m in registry.list_models()] == [
    "openai:gpt-4.1-mini",
    "openai:gpt-4o-mini",
  ]


def test_complete_merges_params_and_returns_response(monkeypatch):
  called = {}

  def fake_completion(*, messages, **kwargs):
    called["messages"] = messages
    called["kwargs"] = kwargs
    return SimpleNamespace(
      choices=[SimpleNamespace(message=SimpleNamespace(content="done"))]
    )

  monkeypatch.setattr(
    "simple_rag_writer.llm.registry.litellm",
    SimpleNamespace(completion=fake_completion),
  )
  monkeypatch.setenv("OPENAI_API_KEY", "env-key")

  cfg = AppConfig(
    default_model="openai:gpt-4.1-mini",
    model_defaults={"max_tokens": 200, "temperature": 0.3},
    providers={
      "openai": ProviderConfig(
        type="openai",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.example.com/v1",
      ),
    },
    models=[
      ModelConfig(
        id="openai:gpt-4.1-mini",
        provider="openai",
        model_name="gpt-4.1-mini",
        params={"temperature": 0.8, "top_p": 0.5},
      ),
    ],
  )

  registry = ModelRegistry(cfg)
  result = registry.complete(
    prompt="Hello",
    task_params={"temperature": 0.1, "presence_penalty": 0.6},
  )

  assert result == "done"
  assert called["messages"] == [{"role": "user", "content": "Hello"}]
  assert called["kwargs"] == {
    "model": "gpt-4.1-mini",
    "api_key": "env-key",
    "api_base": "https://api.example.com/v1",
    "max_tokens": 200,
    "temperature": 0.1,
    "top_p": 0.5,
    "presence_penalty": 0.6,
    "tools": None,
  }


def test_complete_missing_provider_raises(monkeypatch):
  monkeypatch.setattr(
    "simple_rag_writer.llm.registry.litellm",
    SimpleNamespace(completion=lambda **_: None),
  )
  cfg = AppConfig(
    default_model="openai:gpt-4.1-mini",
    providers={},
    models=[
      ModelConfig(
        id="openai:gpt-4.1-mini",
        provider="openai",
        model_name="gpt-4.1-mini",
      )
    ],
  )
  registry = ModelRegistry(cfg)
  with pytest.raises(RuntimeError):
    registry.complete(prompt="Hello")


def test_complete_missing_api_key_raises(monkeypatch):
  monkeypatch.setattr(
    "simple_rag_writer.llm.registry.litellm",
    SimpleNamespace(completion=lambda **_: None),
  )
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)

  cfg = AppConfig(
    default_model="openai:gpt-4.1-mini",
    providers={
      "openai": ProviderConfig(type="openai", api_key_env="OPENAI_API_KEY"),
    },
    models=[
      ModelConfig(
        id="openai:gpt-4.1-mini",
        provider="openai",
        model_name="gpt-4.1-mini",
      )
    ],
  )
  registry = ModelRegistry(cfg)
  with pytest.raises(RuntimeError):
    registry.complete(prompt="Hello")


def test_complete_handles_mcp_tool_calls(monkeypatch):
  calls: List[Dict[str, Any]] = []

  def fake_completion(*, messages, tools=None, **kwargs):
    calls.append({"messages": messages, "tools": tools})
    if len(calls) == 1:
      tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(
          name="call_mcp_tool",
          arguments=json.dumps(
            {
              "server": "notes",
              "tool": "search",
              "params": {"query": "topic"},
            }
          ),
        ),
      )
      return SimpleNamespace(
        choices=[
          SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=[tool_call])
          )
        ]
      )
    return SimpleNamespace(
      choices=[
        SimpleNamespace(
          message=SimpleNamespace(content="final answer", tool_calls=[])
        )
      ]
    )

  monkeypatch.setattr(
    "simple_rag_writer.llm.registry.litellm",
    SimpleNamespace(completion=fake_completion),
  )

  cfg = AppConfig(
    default_model="openai:gpt-4.1-mini",
    providers={
      "openai": ProviderConfig(type="openai", api_key="test-key"),
    },
    models=[
      ModelConfig(
        id="openai:gpt-4.1-mini",
        provider="openai",
        model_name="gpt-4.1-mini",
      ),
    ],
    mcp_servers=[McpServerConfig(id="notes", command=["notes"])],
  )
  registry = ModelRegistry(cfg)

  class FakeMcpClient:
    def __init__(self):
      self.calls: List[Tuple[str, str, Dict[str, Any]]] = []

    def call_tool(self, server: str, tool: str, params: Dict[str, Any]) -> McpToolResult:
      self.calls.append((server, tool, params))
      return McpToolResult(
        server_id=server,
        tool_name=tool,
        payload=[{"title": "Fetched", "body": "data"}],
      )

  fake_client = FakeMcpClient()
  result = registry.complete("Hello", mcp_client=fake_client)

  assert result == "final answer"
  assert fake_client.calls == [("notes", "search", {"query": "topic"})]
  assert calls[0]["messages"][0]["role"] == "system"
  assert calls[0]["tools"][0]["name"] == "call_mcp_tool"
  last_message = calls[1]["messages"][-1]
  assert last_message["role"] == "tool"
  assert "Result from notes:search" in last_message["content"]
