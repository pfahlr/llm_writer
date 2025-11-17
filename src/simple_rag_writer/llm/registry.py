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
    self._tool_call_history: List[Tuple[str, str, str]] = []  # (server, tool, json_params)

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
    system_prompt: Optional[str] = None,
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

    system_content: Optional[str] = None
    tool_metadata: Dict[str, List[Dict[str, Any]]] = {}
    has_mcp_tools = self._should_enable_mcp_tools(mcp_client)
    if system_prompt:
      system_content = system_prompt.strip()
    elif has_mcp_tools and mcp_client is not None:
      tool_metadata = self._collect_mcp_tool_metadata(mcp_client)
      system_content = self._build_mcp_tool_instruction(tool_metadata)
    elif model.system_prompt:
      system_content = model.system_prompt.strip()
    else:
      system_content = "You are a helpful writing assistant."

    messages: List[Dict[str, object]] = []
    if system_content:
      messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": prompt})

    supports_functions = self._provider_supports_function_calls(provider)
    tools_payload = [self._build_mcp_tool_definition(tool_metadata)] if supports_functions and has_mcp_tools else None
    completion_kwargs = dict(kwargs)
    attempt = 0
    # Get configurable tool iteration settings
    tool_config = self._get_tool_iteration_config()
    max_tool_iterations = tool_config.max_iterations if tool_config else 3
    detect_loops = tool_config.detect_loops if tool_config else True
    # Clear tool call history at start of new completion
    self.clear_tool_call_history()
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
          import sys
          print(
            f"[Warning] Model {model.model_name} does not support function calling. "
            f"Falling back to textual tool mode.",
            file=sys.stderr,
          )
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
          raise RuntimeError(
            f"LLM exceeded maximum tool iterations ({max_tool_iterations}). "
            f"The model may be stuck in a loop trying to call: {server_id}:{tool_name}. "
            f"Try rephrasing your request or switching models."
          )
        tool_call = tool_calls[0]
        server_id, tool_name, params = self._parse_mcp_tool_call(tool_call)

        # Record tool call and check for loops
        call_signature = (server_id, tool_name, json.dumps(params, sort_keys=True))
        if detect_loops and self._is_loop_detected(call_signature):
          raise RuntimeError(
            f"Tool call loop detected: {server_id}:{tool_name} called repeatedly "
            f"with identical parameters. The model may be stuck. "
            f"Try rephrasing your request or reducing complexity."
          )
        self._tool_call_history.append(call_signature)

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

          # Record tool call and check for loops
          call_signature = (server_id, tool_name, json.dumps(params, sort_keys=True))
          if detect_loops and self._is_loop_detected(call_signature):
            raise RuntimeError(
              f"Tool call loop detected: {server_id}:{tool_name} called repeatedly "
              f"with identical parameters. The model may be stuck. "
              f"Try rephrasing your request or reducing complexity."
            )
          self._tool_call_history.append(call_signature)

          result = mcp_client.call_tool(server_id, tool_name, params)
          messages.append({"role": "assistant", "content": text_output})
          messages.append(self._render_textual_tool_message(result, server_id, tool_name, params))
          self._log_tool_event(server_id, tool_name, params, result)
          attempt += 1
          if attempt >= max_tool_iterations:
            raise RuntimeError(
              f"LLM exceeded maximum tool iterations ({max_tool_iterations}) using textual tool calls. "
              f"Last tool requested: {server_id}:{tool_name}. "
              f"The model may not support your current MCP server or query. Try simplifying your request."
            )
          continue
      if not text_output or not text_output.strip():
        raise RuntimeError(
          "LLM returned empty response. This may indicate:\n"
          "  - Model output was filtered by content policy\n"
          "  - Request exceeded context length\n"
          "  - Model encountered an internal error\n"
          "Try rephrasing your request or using a different model."
        )
      return text_output

  def complete_streaming(
    self,
    prompt: str,
    model_id: Optional[str] = None,
    task_params: Optional[Dict[str, Any]] = None,
    mcp_client: Optional[McpClient] = None,
    system_prompt: Optional[str] = None,
    on_chunk: Optional[Any] = None,  # Callable[[str], None]
  ) -> str:
    """
    Complete with streaming output for final response.

    Uses hybrid approach:
    - Non-streaming for tool iterations (better tool call handling)
    - Streaming only for final text response (better UX)

    Args:
      prompt: User prompt
      model_id: Override model
      task_params: Generation parameters
      mcp_client: MCP client for tool calls
      system_prompt: Override system prompt
      on_chunk: Callback for each text chunk (only during final response)

    Returns:
      Complete assembled response text
    """
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

    # Build system content and tool metadata
    system_content, tool_metadata, has_mcp_tools = self._build_system_content(
      model, mcp_client, system_prompt
    )

    messages: List[Dict[str, object]] = []
    if system_content:
      messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": prompt})

    supports_functions = self._provider_supports_function_calls(provider)
    tools_payload = [self._build_mcp_tool_definition(tool_metadata)] if supports_functions and has_mcp_tools else None
    completion_kwargs = dict(kwargs)
    attempt = 0

    # Get configurable tool iteration settings
    tool_config = self._get_tool_iteration_config()
    max_tool_iterations = tool_config.max_iterations if tool_config else 3
    detect_loops = tool_config.detect_loops if tool_config else True
    self.clear_tool_call_history()

    while True:
      call_kwargs = dict(completion_kwargs)
      call_kwargs["messages"] = [dict(msg) for msg in messages]
      if tools_payload is not None:
        call_kwargs["tools"] = tools_payload

      # Check if this might be the final response (no more tool calls expected)
      # We'll try streaming first, but if tool calls arrive, we'll handle them
      is_final_likely = attempt > 0  # After first tool call, subsequent responses often final

      try:
        if is_final_likely and on_chunk:
          # Try streaming for potentially final response
          call_kwargs["stream"] = True
          response_stream = litellm.completion(**call_kwargs)

          accumulated_text = []
          tool_calls_found = False

          for chunk in response_stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
              continue

            # Check for tool calls
            chunk_tool_calls = getattr(delta, "tool_calls", None)
            if chunk_tool_calls:
              tool_calls_found = True
              break

            # Stream text content
            content = getattr(delta, "content", None)
            if content:
              accumulated_text.append(content)
              on_chunk(content)

          if tool_calls_found:
            # Tool call detected mid-stream; fall back to non-streaming
            # Re-call without streaming to properly handle tool call
            call_kwargs["stream"] = False
            response = litellm.completion(**call_kwargs)
          else:
            # Streaming complete, assemble final text
            text_output = "".join(accumulated_text)
            if not text_output or not text_output.strip():
              raise RuntimeError(
                "LLM returned empty response. This may indicate:\n"
                "  - Model output was filtered by content policy\n"
                "  - Request exceeded context length\n"
                "  - Model encountered an internal error\n"
                "Try rephrasing your request or using a different model."
              )
            return text_output
        else:
          # Non-streaming call (tool iterations)
          response = litellm.completion(**call_kwargs)

      except AttributeError as exc:
        raise RuntimeError("litellm BadRequest: " + str(exc)) from exc
      except Exception as exc:  # noqa: BLE001
        if supports_functions and isinstance(exc, getattr(litellm, "BadRequestError", Exception)):
          import sys
          print(
            f"[Warning] Model {model.model_name} does not support function calling. "
            f"Falling back to textual tool mode.",
            file=sys.stderr,
          )
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
          raise RuntimeError(
            f"LLM exceeded maximum tool iterations ({max_tool_iterations}). "
            f"The model may be stuck in a loop. Try rephrasing your request."
          )
        tool_call = tool_calls[0]
        server_id, tool_name, params = self._parse_mcp_tool_call(tool_call)

        # Record tool call and check for loops
        call_signature = (server_id, tool_name, json.dumps(params, sort_keys=True))
        if detect_loops and self._is_loop_detected(call_signature):
          raise RuntimeError(
            f"Tool call loop detected: {server_id}:{tool_name} called repeatedly "
            f"with identical parameters. The model may be stuck."
          )
        self._tool_call_history.append(call_signature)

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

          # Record tool call and check for loops
          call_signature = (server_id, tool_name, json.dumps(params, sort_keys=True))
          if detect_loops and self._is_loop_detected(call_signature):
            raise RuntimeError(
              f"Tool call loop detected: {server_id}:{tool_name} called repeatedly "
              f"with identical parameters. The model may be stuck."
            )
          self._tool_call_history.append(call_signature)

          result = mcp_client.call_tool(server_id, tool_name, params)
          messages.append({"role": "assistant", "content": text_output})
          messages.append(self._render_textual_tool_message(result, server_id, tool_name, params))
          self._log_tool_event(server_id, tool_name, params, result)
          attempt += 1
          if attempt >= max_tool_iterations:
            raise RuntimeError(
              f"LLM exceeded maximum tool iterations ({max_tool_iterations}) using textual tool calls."
            )
          continue

      if not text_output or not text_output.strip():
        raise RuntimeError(
          "LLM returned empty response. This may indicate:\n"
          "  - Model output was filtered by content policy\n"
          "  - Request exceeded context length\n"
          "  - Model encountered an internal error\n"
          "Try rephrasing your request or using a different model."
        )
      return text_output

  def _build_system_content(
    self,
    model: Any,  # ModelConfig
    mcp_client: Optional[McpClient],
    system_prompt: Optional[str],
  ) -> tuple[Optional[str], Dict[str, List[Dict[str, Any]]], bool]:
    """
    Build system content and collect tool metadata.

    Returns:
      (system_content, tool_metadata, has_mcp_tools)
    """
    tool_metadata: Dict[str, List[Dict[str, Any]]] = {}
    has_mcp_tools = self._should_enable_mcp_tools(mcp_client)

    if system_prompt:
      system_content = system_prompt.strip()
    elif has_mcp_tools and mcp_client is not None:
      tool_metadata = self._collect_mcp_tool_metadata(mcp_client)
      system_content = self._build_mcp_tool_instruction(tool_metadata)
    elif model.system_prompt:
      system_content = model.system_prompt.strip()
    else:
      system_content = "You are a helpful writing assistant."

    return system_content, tool_metadata, has_mcp_tools

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

  def get_tool_call_history(self) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return recent tool call history for this completion."""
    return [
      (server, tool, json.loads(params_json))
      for server, tool, params_json in self._tool_call_history
    ]

  def clear_tool_call_history(self) -> None:
    """Clear tool call history (called at start of new completion)."""
    self._tool_call_history.clear()

  def _get_tool_iteration_config(self) -> Optional[Any]:
    """Get effective tool iteration config for current model."""
    from simple_rag_writer.config.models import ToolIterationConfig

    model = self.current_model
    # Priority: model override > app defaults > built-in defaults
    return (
      model.tool_iteration_override
      or self._config.tool_iteration_defaults
      or ToolIterationConfig()  # Use default values
    )

  def _is_loop_detected(self, call_signature: Tuple[str, str, str]) -> bool:
    """
    Check if current call matches recent history (loop detection).

    A loop is detected if the same (server, tool, params) appears
    multiple times within the loop_window.
    """
    config = self._get_tool_iteration_config()
    if not config or not config.detect_loops:
      return False

    window = config.loop_window

    # Check last N calls
    recent = self._tool_call_history[-window:] if len(self._tool_call_history) >= window else self._tool_call_history

    # Count how many times this signature appears in recent history
    matches = [call for call in recent if call == call_signature]

    # If we've seen this exact call 2+ times in window, it's a loop
    return len(matches) >= 2

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
