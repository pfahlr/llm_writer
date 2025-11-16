from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from simple_rag_writer.config.loader import load_config
from simple_rag_writer.config.models import AppConfig, LlmToolConfig
from simple_rag_writer.llm.registry import ModelRegistry

LATEST_PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "Simple Rag Writer LLM Tool"
SERVER_VERSION = "0.1.0"


class LlmToolHandler:
  def __init__(self, config: AppConfig, registry: Optional[ModelRegistry] = None):
    self._config = config
    self._tool_config = config.llm_tool
    if not self._tool_config:
      raise RuntimeError("llm_tool configuration is required to run the LLM server.")
    if not self._tool_config.skills:
      raise RuntimeError("llm_tool configuration must expose at least one skill.")
    self._registry = registry or ModelRegistry(config)
    self._schema = self._build_input_schema()

  @property
  def tool_config(self) -> LlmToolConfig:
    return self._tool_config

  def list_tools(self) -> List[Dict[str, Any]]:
    skill_options = list(self._tool_config.skills.keys())
    default_skill = self._tool_config.default_skill or skill_options[0]
    schema = dict(self._schema)
    schema["properties"] = dict(schema["properties"])
    schema["properties"]["skill"] = {
      "type": "string",
      "description": "Choose the appropriate model skill.",
      "enum": skill_options,
      "default": default_skill,
    }
    return [
      {
        "name": self._tool_config.tool_name,
        "title": self._tool_config.title or self._tool_config.tool_name,
        "description": self._tool_config.description or "Call a configured LLM skill.",
        "inputSchema": schema,
      }
    ]

  def call_tool(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (arguments.get("prompt") or "").strip()
    if not prompt:
      raise ValueError("Missing required 'prompt' argument.")
    skill = arguments.get("skill") or self._tool_config.default_skill or next(iter(self._tool_config.skills))
    model_id = self._tool_config.skills.get(skill)
    if not model_id:
      raise ValueError(f"Unknown skill '{skill}'. Available: {list(self._tool_config.skills.keys())}")
    task_params = self._build_task_params(arguments)
    response = self._registry.complete(prompt, model_id=model_id, task_params=task_params)
    text = (response or "").strip()
    item = {
      "id": f"{self._tool_config.id}:{skill}",
      "type": "llm",
      "title": skill,
      "body": text,
      "metadata": {
        "model": model_id,
        "skill": skill,
      },
    }
    return {
      "content": [{"type": "text", "text": text}],
      "structuredContent": {"items": [item]},
    }

  def _build_input_schema(self) -> Dict[str, Any]:
    base = {
      "type": "object",
      "properties": {
        "prompt": {
          "type": "string",
          "description": "The text prompt to send to the LLM.",
        },
      },
      "required": ["prompt"],
    }
    for key, spec in {
      "max_tokens": {
        "type": "integer",
        "description": "Maximum tokens to generate.",
        "minimum": 1,
      },
      "temperature": {
        "type": "number",
        "description": "Sampling temperature (0-2).",
        "minimum": 0,
      },
      "top_p": {
        "type": "number",
        "description": "Nucleus sampling probability (0-1).",
        "minimum": 0,
        "maximum": 1,
      },
      "presence_penalty": {
        "type": "number",
        "description": "Presence penalty (>=-2 and <=2).",
        "minimum": -2,
        "maximum": 2,
      },
      "frequency_penalty": {
        "type": "number",
        "description": "Frequency penalty (>=-2 and <=2).",
        "minimum": -2,
        "maximum": 2,
      },
    }.items():
      base["properties"][key] = spec
    return base

  def _build_task_params(self, arguments: Dict[str, Any]) -> Dict[str, Any] | None:
    params: Dict[str, Any] = {}
    for key in ("max_tokens", "temperature", "top_p", "presence_penalty", "frequency_penalty"):
      if key not in arguments:
        continue
      value = arguments[key]
      if value is None:
        continue
      if key == "max_tokens":
        value = int(value)
        limit = self._tool_config.max_tokens_limit
        if limit is not None and value > limit:
          raise ValueError(f"'max_tokens' cannot exceed {limit}.")
      else:
        value = float(value)
      params[key] = value
    return params or None


def _write_response(payload: Dict[str, Any]) -> None:
  sys.stdout.write(json.dumps(payload) + "\n")
  sys.stdout.flush()


def _handle_initialize(message: Dict[str, Any]) -> Dict[str, Any]:
  return {
    "jsonrpc": "2.0",
    "id": message["id"],
    "result": {
      "protocolVersion": LATEST_PROTOCOL_VERSION,
      "serverInfo": {
        "name": SERVER_NAME,
        "version": SERVER_VERSION,
      },
      "capabilities": {
        "tools": {"listChanged": False},
      },
    },
  }


def _handle_list_tools(handler: LlmToolHandler, message: Dict[str, Any]) -> Dict[str, Any]:
  return {
    "jsonrpc": "2.0",
    "id": message["id"],
    "result": {
      "tools": handler.list_tools(),
      "nextCursor": None,
    },
  }


def _handle_call(handler: LlmToolHandler, message: Dict[str, Any]) -> Dict[str, Any]:
  params = message.get("params") or {}
  arguments = params.get("arguments") or {}
  try:
    result = handler.call_tool(arguments)
  except ValueError as exc:
    return {
      "jsonrpc": "2.0",
      "id": message["id"],
      "result": {
        "content": [{"type": "text", "text": str(exc)}],
        "structuredContent": {"items": []},
        "isError": True,
      },
    }
  result["isError"] = False
  return {
    "jsonrpc": "2.0",
    "id": message["id"],
    "result": result,
  }


def _handle_message(handler: LlmToolHandler, raw: str) -> None:
  try:
    message = json.loads(raw)
  except json.JSONDecodeError:
    return
  if "id" not in message:
    return
  method = message.get("method")
  if method == "initialize":
    _write_response(_handle_initialize(message))
  elif method == "tools/list":
    _write_response(_handle_list_tools(handler, message))
  elif method == "tools/call":
    _write_response(_handle_call(handler, message))
  else:
    pass


def main() -> int:
  parser = argparse.ArgumentParser(description="Run the Simple Rag Writer LLM MCP tool server.")
  parser.add_argument("--config", required=True, help="Path to config.yaml")
  args = parser.parse_args()
  config = load_config(Path(args.config))
  handler = LlmToolHandler(config)
  for line in sys.stdin:
    stripped = line.strip()
    if not stripped:
      continue
    _handle_message(handler, stripped)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
