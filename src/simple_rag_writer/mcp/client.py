from __future__ import annotations

from typing import Any, Dict

from .types import McpToolResult


class McpClient:
  """
  Placeholder for MCP client. Responsibilities:

  - Start/stop MCP servers (from AppConfig.mcp_servers).
  - Call tools on servers and return McpToolResult.
  """

  def __init__(self, config):
    self._config = config
    # TODO: process spawning / connection to MCP servers

  def call_tool(self, server_id: str, tool_name: str, params: Dict[str, Any]) -> McpToolResult:
    # TODO: real MCP implementation
    raise NotImplementedError("McpClient.call_tool is not implemented yet")
