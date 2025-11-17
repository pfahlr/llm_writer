from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from simple_rag_writer.config.models import AppConfig
from simple_rag_writer.mcp.client import McpClient


@dataclass
class ServerHealthStatus:
  """Health check result for an MCP server."""

  server_id: str
  available: bool
  error: Optional[str] = None
  tool_count: int = 0
  response_time_ms: Optional[float] = None


def check_server_health(
  mcp_client: McpClient, server_id: str, timeout: float = 5.0
) -> ServerHealthStatus:
  """
  Check if an MCP server is healthy.

  Returns:
    ServerHealthStatus with availability and metrics
  """
  start = time.time()

  try:
    # Try listing tools as health check
    tools = mcp_client.list_tools(server_id)
    elapsed_ms = (time.time() - start) * 1000

    return ServerHealthStatus(
      server_id=server_id,
      available=True,
      tool_count=len(tools) if tools else 0,
      response_time_ms=elapsed_ms,
    )
  except Exception as exc:  # noqa: BLE001
    return ServerHealthStatus(
      server_id=server_id,
      available=False,
      error=str(exc),
    )


def check_required_servers(
  config: AppConfig, mcp_client: McpClient
) -> tuple[bool, List[ServerHealthStatus]]:
  """
  Check health of all required MCP servers.

  Returns:
    (all_required_ok, list of health statuses)
  """
  statuses = []
  all_ok = True

  for server in config.mcp_servers or []:
    status = check_server_health(mcp_client, server.id)
    statuses.append(status)

    if server.criticality == "required" and not status.available:
      all_ok = False

  return all_ok, statuses
