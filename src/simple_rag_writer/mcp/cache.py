from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from simple_rag_writer.mcp.types import McpToolResult


class McpResultCache:
  """
  Simple file-based cache for MCP tool results.

  Enables fallback to previous results when servers are unavailable.
  """

  def __init__(self, cache_dir: Path, ttl_seconds: int = 3600):
    self.cache_dir = cache_dir
    self.ttl_seconds = ttl_seconds
    self.cache_dir.mkdir(parents=True, exist_ok=True)

  def _cache_key(self, server: str, tool: str, params: Dict[str, Any]) -> str:
    """Generate cache key from call signature."""
    signature = json.dumps(
      {"server": server, "tool": tool, "params": params}, sort_keys=True
    )
    return hashlib.sha256(signature.encode()).hexdigest()[:16]

  def get(
    self, server: str, tool: str, params: Dict[str, Any]
  ) -> Optional[McpToolResult]:
    """
    Retrieve cached result if available and fresh.

    Returns:
      Cached McpToolResult or None if not found/expired
    """
    key = self._cache_key(server, tool, params)
    cache_file = self.cache_dir / f"{key}.json"

    if not cache_file.exists():
      return None

    try:
      with cache_file.open("r") as f:
        data = json.load(f)

      # Check TTL
      cached_at = data.get("cached_at", 0)
      age = time.time() - cached_at
      if age > self.ttl_seconds:
        # Expired
        cache_file.unlink()
        return None

      return McpToolResult(
        server_id=data["server_id"],
        tool_name=data["tool_name"],
        payload=data["payload"],
      )
    except Exception:  # noqa: BLE001
      return None

  def put(
    self, server: str, tool: str, params: Dict[str, Any], result: McpToolResult
  ) -> None:
    """Cache a successful MCP result."""
    key = self._cache_key(server, tool, params)
    cache_file = self.cache_dir / f"{key}.json"

    data = {
      "server_id": result.server_id,
      "tool_name": result.tool_name,
      "payload": result.payload,
      "cached_at": time.time(),
    }

    try:
      with cache_file.open("w") as f:
        json.dump(data, f, indent=2)
    except Exception:  # noqa: BLE001
      pass  # Cache write failure is non-critical
