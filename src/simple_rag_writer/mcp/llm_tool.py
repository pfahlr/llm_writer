from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from simple_rag_writer.config import PromptsFile, SkillLibrary, load_prompts_config
from simple_rag_writer.config.loader import load_config
from simple_rag_writer.config.models import AppConfig, LlmToolConfig
from simple_rag_writer.llm.executor import LlmCompletionError, run_completion_with_feedback
from simple_rag_writer.llm.registry import ModelRegistry

LATEST_PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "Simple Rag Writer LLM Tool"
SERVER_VERSION = "0.1.0"


class LlmToolHandler:
  def __init__(
    self,
    config: AppConfig,
    *,
    prompts: Optional[PromptsFile] = None,
    registry: Optional[ModelRegistry] = None,
  ):
    self._config = config
    self._tool_config = config.llm_tool
    if not self._tool_config:
      raise RuntimeError("llm_tool configuration is required to run the LLM server.")
    if not self._tool_config.skills:
      raise RuntimeError("llm_tool configuration must expose at least one skill.")
    self._prompts = prompts or PromptsFile.empty()
    self._skills = SkillLibrary(config, self._prompts)
    self._registry = registry or ModelRegistry(config)
    self._schema = self._build_input_schema()

  @property
  def tool_config(self) -> LlmToolConfig:
    return self._tool_config

  def list_tools(self) -> List[Dict[str, Any]]:
    skill_options = self._skills.list_skill_ids()
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
    skill_id = (
      arguments.get("skill")
      or self._tool_config.default_skill
      or (self._skills.list_skill_ids()[0] if self._skills.list_skill_ids() else None)
    )
    if not skill_id:
      raise ValueError("No skills are configured.")
    try:
      resolved = self._skills.resolve_skill(skill_id)
    except KeyError as exc:
      raise ValueError(str(exc)) from exc
    task_params = self._build_task_params(arguments, base=resolved.default_params)
    try:
      response = run_completion_with_feedback(
        self._registry,
        prompt,
        model_id=resolved.model.id,
        system_prompt=resolved.system_prompt,
        task_params=task_params,
        max_attempts=2,
      )
    except LlmCompletionError as exc:
      return {
        "content": [
          {
            "type": "text",
            "text": f"LLM skill '{skill_id}' failed after retries: {exc.message}",
          }
        ],
        "structuredContent": {"items": []},
        "isError": True,
      }
    text = (response or "").strip()
    item = {
      "id": f"{self._tool_config.id}:{skill_id}",
      "type": "llm",
      "title": resolved.skill.label or skill_id,
      "body": text,
      "metadata": {
        "model": resolved.model.id,
        "skill": skill_id,
        "prompt": resolved.skill.prompt_id,
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

  def _build_task_params(
    self, arguments: Dict[str, Any], base: Optional[Dict[str, Any]] = None
  ) -> Dict[str, Any] | None:
    params: Dict[str, Any] = dict(base or {})
    if "max_tokens" in params:
      params["max_tokens"] = self._validate_max_tokens(int(params["max_tokens"]))
    for key in ("max_tokens", "temperature", "top_p", "presence_penalty", "frequency_penalty"):
      if key not in arguments:
        continue
      value = arguments[key]
      if value is None:
        continue
      if key == "max_tokens":
        value = self._validate_max_tokens(int(value))
      else:
        value = float(value)
      params[key] = value
    return params or None

  def _validate_max_tokens(self, value: int) -> int:
    limit = self._tool_config.max_tokens_limit
    if limit is not None and value > limit:
      raise ValueError(f"'max_tokens' cannot exceed {limit}.")
    return value


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
  parser.add_argument(
    "--prompts",
    help="Path to prompts.yaml (defaults to prompts.yaml next to config file).",
  )
  args = parser.parse_args()
  config_path = Path(args.config)
  config = load_config(config_path)
  prompts_path = Path(args.prompts) if args.prompts else config_path.with_name("prompts.yaml")
  prompts = load_prompts_config(prompts_path)
  handler = LlmToolHandler(config, prompts=prompts)
  for line in sys.stdin:
    stripped = line.strip()
    if not stripped:
      continue
    _handle_message(handler, stripped)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
