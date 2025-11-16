from __future__ import annotations

from contextlib import asynccontextmanager
import json
from typing import Any, Dict, List

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import mcp.types as mcp_types

from simple_rag_writer.config.models import AppConfig, McpServerConfig

from .types import McpToolResult


class McpClient:
  """
  Placeholder for MCP client. Responsibilities:

  - Start/stop MCP servers (from AppConfig.mcp_servers).
  - Call tools on servers and return McpToolResult.
  """

  def __init__(self, config: AppConfig):
    self._servers: Dict[str, McpServerConfig] = {server.id: server for server in config.mcp_servers}

  def call_tool(self, server_id: str, tool_name: str, params: Dict[str, Any]) -> McpToolResult:
    return anyio.run(self._call_tool_async, server_id, tool_name, dict(params))

  def list_tools(self, server_id: str) -> List[Dict[str, Any]]:
    return anyio.run(self._list_tools_async, server_id)

  async def _list_tools_async(self, server_id: str) -> List[Dict[str, Any]]:
    server = self._get_server(server_id)
    async with self._connect(server) as session:
      result = await session.list_tools()
      return [self._normalize_tool(tool) for tool in result.tools]

  async def _call_tool_async(self, server_id: str, tool_name: str, params: Dict[str, Any]) -> McpToolResult:
    server = self._get_server(server_id)
    async with self._connect(server) as session:
      result = await session.call_tool(tool_name, params or {})
    payload = self._extract_payload(result)
    return McpToolResult(server_id=server_id, tool_name=tool_name, payload=payload)

  def _get_server(self, server_id: str) -> McpServerConfig:
    if server_id not in self._servers:
      raise KeyError(server_id)
    return self._servers[server_id]

  @asynccontextmanager
  async def _connect(self, server: McpServerConfig):
    if not server.command:
      raise RuntimeError(f"MCP server '{server.id}' has no command configured")
    params = StdioServerParameters(command=server.command[0], args=list(server.command[1:]))
    try:
      async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
          await session.initialize()
          yield session
    except OSError as exc:
      raise RuntimeError(f"Failed to start MCP server '{server.id}': {exc}") from exc

  def _normalize_tool(self, tool: mcp_types.Tool) -> Dict[str, Any]:
    return tool.model_dump(mode="json", by_alias=True, exclude_none=True)

  def _extract_payload(self, result: mcp_types.CallToolResult) -> Any:
    if result.isError:
      message = self._first_text_block(result) or "MCP tool returned an error"
      raise RuntimeError(message)
    if result.structuredContent is not None:
      return self._unwrap_structured(result.structuredContent)
    if not result.content:
      return []
    structured_blocks: List[Dict[str, Any]] = []
    text_blocks: List[str] = []
    for block in result.content:
      if isinstance(block, mcp_types.TextContent):
        text_blocks.append(block.text)
      else:
        structured_blocks.append(block.model_dump(mode="json", by_alias=True))
    if structured_blocks:
      return structured_blocks
    if len(text_blocks) == 1:
      return self._maybe_parse_json(text_blocks[0])
    return text_blocks

  def _unwrap_structured(self, data: Any) -> Any:
    if isinstance(data, dict):
      for key in ("items", "references", "results", "data"):
        if key in data:
          return data[key]
    return data

  def _maybe_parse_json(self, text: str) -> Any:
    try:
      return json.loads(text)
    except json.JSONDecodeError:
      return text

  def _first_text_block(self, result: mcp_types.CallToolResult) -> str | None:
    for block in result.content or []:
      if isinstance(block, mcp_types.TextContent):
        return block.text
    return None
