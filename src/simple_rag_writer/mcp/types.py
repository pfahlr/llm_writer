from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class McpToolResult(BaseModel):
  server_id: str
  tool_name: str
  payload: Any


class NormalizedItem(BaseModel):
  id: Optional[str] = None
  type: Optional[str] = None
  title: Optional[str] = None
  snippet: Optional[str] = None
  body: Optional[str] = None
  url: Optional[str] = None
  metadata: Dict[str, Any] = Field(default_factory=dict)
