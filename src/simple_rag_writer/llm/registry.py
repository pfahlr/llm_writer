from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
  import litellm
except ImportError:  # pragma: no cover - only triggered when dependency missing
  litellm = None

from simple_rag_writer.config.models import AppConfig, ModelConfig, ProviderConfig
from simple_rag_writer.mcp.client import McpClient
from simple_rag_writer.mcp.message_formatting import format_mcp_result_for_llm
from simple_rag_writer.mcp.normalization import normalize_payload
from simple_rag_writer.mcp.types import McpToolResult
from .params import merge_generation_params


# pylint: disable=too-many-instance-attributes
class ModelRegistry:
  def __init__(self, config: AppConfig):
    self._config = config
    self._models: Dict[str, ModelConfig] = {m.id: m for m in config.models}
    if config.default_model not in self._models:
      raise ValueError(f"default_model {config.default_model} not found in models")
    self._current_id = config.default_model
    self._provider_supports_functions: Dict[str, bool] = {}
    self._tool_events: List[str] = []

  @property
  def current_id(self) -> str:
    return self._current_id

  @property
  def current_model(self) -> ModelConfig:
    return self._models[self._current_id]

  def set_current(self, model_id: str) -> None:
    if model_id not in self._models:
      raise KeyError(f"Unknown model id: {model_id}")
    self._current_id = model_id

  def list_models(self) -> List[ModelConfig]:
    return list(self._models.values())

  def _resolve_provider(self, m: ModelConfig) -> ProviderConfig:
    providers = self._config.providers
    if m.provider not in providers:
      raise RuntimeError(f"Unknown provider {m.provider} for model {m.id}")
    return providers[m.provider]

  def _resolve_api_key(self, provider: ProviderConfig) -> str:
    if provider.api_key:
      return provider.api_key
    if provider.api_key_env:
      value = os.environ.get(provider.api_key_env)
      if value:
        return value
    raise RuntimeError("No API key configured for provider")

  def complete(
    self,
    prompt: str,
    model_id: Optional[str] = None,
    task_params: Optional[Dict[str, Any]] = None,
    mcp_client: Optional[McpClient] = None,
  ) -> str:
    if litellm is None:
      raise RuntimeError("litellm is required to perform completions")

    model = self._models[model_id] if model_id else self.current_model
    provider = self._resolve_provider(model)
    api_key = self._resolve_api_key(provider)

    gen_params = merge_generation_params(self._config, model, task_params)

    kwargs: Dict[str, Any] = {
      "model": model.model_name,
      "api_key": api_key,
    }
    if provider.base_url:
      kwargs["api_base"] = provider.base_url

    kwargs.update(gen_params)

    system_message: Optional[Dict[str, str]] = None
    tool_metadata: Dict[str, List[Dict[str, Any]]] = {}
    has_mcp_tools = self._should_enable_mcp_tools(mcp_client)
    if has_mcp_tools and mcp_client is not None:
      tool_metadata = self._collect_mcp_tool_metadata(mcp_client)
    system_message = {
      "role": "system",
      "content": self._build_mcp_tool_instruction(tool_metadata),
    }

    messages: List[Dict[str, object]] = [system_message, {"role": "user", "content": prompt}]

    supports_functions = self._provider_supports_function_calls(provider)
    tools_payload = [self._build_mcp_tool_definition(tool_metadata)] if supports_functions and has_mcp_tools else None
    completion_kwargs = dict(kwargs)
    attempt = 0
    max_tool_iterations = 3
    while True:
      call_kwargs = dict(completion_kwargs)
      call_kwargs["messages"] = [dict(msg) for msg in messages]
      if tools_payload is not None:
        call_kwargs["tools"] = tools_payload
      try:
        response = litellm.completion(**call_kwargs)
      except AttributeError as exc:
        raise RuntimeError("litellm BadRequest: " + str(exc)) from exc
      except Exception as exc:  # noqa: BLE001
        if supports_functions and isinstance(exc, getattr(litellm, "BadRequestError", Exception)):
          self._provider_supports_functions[provider.type] = False
          supports_functions = False
          tools_payload = None
          continue
        raise
      message = response.choices[0].message
      tool_calls = getattr(message, "tool_calls", None)
      text_output = getattr(message, "content", "") or ""
      if tool_calls:
        if mcp_client is None:
          raise RuntimeError("Model tried to invoke MCP tools but no client is configured.")
        if attempt >= max_tool_iterations:
          raise RuntimeError("LLM requested too many MCP tool calls.")
        tool_call = tool_calls[0]
        server_id, tool_name, params = self._parse_mcp_tool_call(tool_call)
        result = mcp_client.call_tool(server_id, tool_name, params)
        messages.append(self._build_assistant_tool_message(message))
        messages.append(self._render_mcp_tool_result(result, tool_call))
        self._log_tool_event(server_id, tool_name, params, result)
        attempt += 1
        continue
      if not supports_functions and has_mcp_tools:
        parsed = self._extract_tool_command_from_text(text_output)
        if parsed:
          if mcp_client is None:
            raise RuntimeError("Model requested MCP tool via text but no client is configured.")
          server_id, tool_name, params = parsed
          result = mcp_client.call_tool(server_id, tool_name, params)
          messages.append({"role": "assistant", "content": text_output})
          messages.append(self._render_textual_tool_message(result, server_id, tool_name, params))
          self._log_tool_event(server_id, tool_name, params, result)
          attempt += 1
          if attempt >= max_tool_iterations:
            raise RuntimeError("LLM requested too many MCP tool calls.")
          continue
      return text_output

  def _should_enable_mcp_tools(self, mcp_client: Optional[McpClient]) -> bool:
    return bool(self._config.mcp_servers) and mcp_client is not None

  def _collect_mcp_tool_metadata(
    self, mcp_client: McpClient
  ) -> Dict[str, List[Dict[str, Any]]]:
    metadata: Dict[str, List[Dict[str, Any]]] = {}
    list_tools = getattr(mcp_client, "list_tools", None)
    for server in self._config.mcp_servers:
      entries: List[Dict[str, Any]] = []
      raw_tools = []
      if list_tools:
        try:
          raw_tools = list_tools(server.id)
        except Exception:
          raw_tools = []
      for tool in raw_tools or []:
        name = tool.get("name") or tool.get("toolName") or tool.get("tool_name")
        if not name:
          continue
        entries.append(
          {
            "name": name,
            "description": tool.get("description") or tool.get("title") or "",
            "inputSchema": tool.get("inputSchema"),
          }
        )
      metadata[server.id] = entries
    return metadata

  def _log_tool_event(
    self,
    server_id: str,
    tool_name: str,
    params: Dict[str, Any],
    result: McpToolResult,
  ) -> None:
    snippet = self._summarize_tool_result(result)
    description = f"Tool {server_id}:{tool_name} params={params} → {snippet}"
    self._tool_events.append(description)

  def _summarize_tool_result(self, result: McpToolResult) -> str:
    items = normalize_payload(result.payload)
    if not items:
      return "no items"
    first = items[0]
    body = (first.body or first.snippet or "").strip()
    return body[:80] + ("…" if len(body or "") > 80 else "")

  def pop_tool_events(self) -> List[str]:
    events = list(self._tool_events)
    self._tool_events.clear()
    return events

  def _build_tool_name_list(self, tool_metadata: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    names = sorted(
      {
        tool["name"]
        for entries in tool_metadata.values()
        for tool in entries
        if tool.get("name")
      }
    )
    return names or ["<tool>"]

  def _render_tool_inventory(self, tool_metadata: Dict[str, List[Dict[str, Any]]]) -> str:
    lines: List[str] = []
    for server, tools in tool_metadata.items():
      if not tools:
        continue
      lines.append(f"{server}:")
      for tool in tools:
        desc = tool.get("description") or "No description provided."
        fields = self._format_schema_fields(tool.get("inputSchema"))
        lines.append(f"  - {tool['name']}: {desc} (params: {fields})")
      if not lines:
        return "No detailed tool metadata is available."
      return "\n".join(lines)

  def _render_params_guidance(self, tool_metadata: Dict[str, List[Dict[str, Any]]]) -> str:
    snippets: List[str] = []
    for server, tools in tool_metadata.items():
      for tool in tools:
        fields = self._format_schema_fields(tool.get("inputSchema"))
        snippets.append(f"{server}/{tool['name']} → {fields}")
    if not snippets:
      return ""
    return "Tool parameter hints: " + "; ".join(snippets)

  @staticmethod
  def _format_schema_fields(schema: Optional[Dict[str, Any]]) -> str:
    if not schema:
      return "any fields"
    props = schema.get("properties") or {}
    if not props:
      return "any fields"
    return ", ".join(sorted(props.keys()))

  def _build_mcp_tool_definition(
    self, tool_metadata: Dict[str, List[Dict[str, Any]]]
  ) -> Dict[str, Any]:
    server_ids = [server.id for server in self._config.mcp_servers] or list(tool_metadata.keys())
    tool_names = self._build_tool_name_list(tool_metadata)
    inventory_description = self._render_tool_inventory(tool_metadata)
    params_guidance = self._render_params_guidance(tool_metadata)
    tool_parameters = {
      "type": "object",
      "properties": {
        "server": {
          "type": "string",
          "description": (
            "MCP server id to contact."
            f" Available servers: {', '.join(server_ids)}."
          ),
          "enum": server_ids,
        },
        "tool": {
          "type": "string",
          "enum": tool_names,
          "description": (
            "Name of the tool exposed by the MCP server."
            f" Tool metadata:\n{inventory_description}"
          ),
        },
        "params": {
          "type": "object",
          "additionalProperties": True,
          "description": (
            "Tool-specific parameters (query, limit, etc.)."
            f" {params_guidance}"
          ),
        },
      },
      "required": ["server", "tool"],
      "additionalProperties": True,
    }

    description = "Use a configured MCP server to fetch structured references."
    return {
      "name": "call_mcp_tool",
      "description": description,
      "parameters": tool_parameters,
    }

  def _provider_supports_function_calls(self, provider: ProviderConfig) -> bool:
    key = provider.type
    cached = self._provider_supports_functions.get(key)
    if cached is not None:
      return cached
    supports = key != "openrouter"
    self._provider_supports_functions[key] = supports
    return supports

  def _extract_tool_command_from_text(self, text: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    marker = "call_mcp_tool"
    idx = text.lower().find(marker)
    if idx == -1:
      return None
    start = text.find("{", idx)
    if start == -1:
      return None
    depth = 0
    end = start
    for pos in range(start, len(text)):
      char = text[pos]
      if char == "{":
        depth += 1
      elif char == "}":
        depth -= 1
        if depth == 0:
          end = pos + 1
          break
    if depth != 0:
      return None
    try:
      payload = json.loads(text[start:end])
    except json.JSONDecodeError:
      return None
    server_id = payload.get("server")
    tool_name = payload.get("tool")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
      return None
    if not server_id or not tool_name:
      return None
    return server_id, tool_name, params

  def _render_textual_tool_message(
    self,
    result: McpToolResult,
    server_id: str,
    tool_name: str,
    params: Dict[str, Any],
  ) -> Dict[str, Any]:
    formatted = format_mcp_result_for_llm(server_id, tool_name, result.payload)
    content = f"TOOL_RESULT call_mcp_tool {server_id}:{tool_name}\n\n{formatted}"
    return {
      "role": "assistant",
      "content": content.strip(),
    }

  def _build_mcp_tool_instruction(self, tool_metadata: Dict[str, List[Dict[str, Any]]]) -> str:
    servers = ", ".join(server.id for server in self._config.mcp_servers) or "no servers configured"
    inventory = self._render_tool_inventory(tool_metadata)
    return (
      "If you need additional context, call `call_mcp_tool` with JSON arguments "
      "containing `server` and `tool`, plus an optional `params` object for tool-specific options. "
      f"Known servers: {servers}.\nAvailable tools:\n{inventory}\n"
      "When function calling is unavailable, respond with `CALL_MCP_TOOL {\"server\":...,\"tool\":...,\"params\":{...}}` "
      "so the client can execute the request."
    )

  def _parse_mcp_tool_call(self, tool_call: Any) -> Tuple[str, str, Dict[str, Any]]:
    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", None)
    if name != "call_mcp_tool":
      raise RuntimeError(f"Unsupported tool called: {name}")
    arguments = getattr(function, "arguments", "") or "{}"
    try:
      parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
      raise RuntimeError("MCP tool arguments must be valid JSON.") from exc
    if not isinstance(parsed, dict):
      raise RuntimeError("MCP tool arguments must be a JSON object.")
    server_id = parsed.pop("server", None)
    tool_name = parsed.pop("tool", None)
    if not server_id or not tool_name:
      raise RuntimeError("MCP tool calls require 'server' and 'tool'.")
    params = parsed.pop("params", None)
    if params is None:
      params = {}
    elif not isinstance(params, dict):
      raise RuntimeError("MCP tool 'params' must be an object.")
    params.update(parsed)
    return server_id, tool_name, params

  @staticmethod
  def _build_assistant_tool_message(message: Any) -> Dict[str, Any]:
    tool_calls = getattr(message, "tool_calls", None)
    payload: Dict[str, Any] = {
      "role": "assistant",
      "content": getattr(message, "content", None),
    }
    if tool_calls:
      payload["tool_calls"] = [
        {
          "id": getattr(call, "id", None),
          "type": getattr(call, "type", "function"),
          "function": {
            "name": getattr(call.function, "name", None),
            "arguments": getattr(call.function, "arguments", ""),
          },
        }
        for call in tool_calls
      ]
    return payload

  def _render_mcp_tool_result(self, result: McpToolResult, tool_call: Any) -> Dict[str, Any]:
    content = format_mcp_result_for_llm(result.server_id, result.tool_name, result.payload)
    if not content.strip():
      content = f"Result from {result.server_id}:{result.tool_name}"
    return {
      "role": "tool",
      "name": getattr(tool_call.function, "name", "call_mcp_tool"),
      "tool_call_id": getattr(tool_call, "id", None),
      "content": content.strip(),
    }
