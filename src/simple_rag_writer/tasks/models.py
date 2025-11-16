from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContextSpec(BaseModel):
  outline_path: Optional[str] = None
  outline_id: Optional[str] = None


class ReferenceCommon(BaseModel):
  type: str
  label: Optional[str] = None
  item_type: Optional[str] = None
  prompt_mode: Optional[str] = None
  max_items: Optional[int] = None
  max_chars: Optional[int] = None
  summary_max_tokens: Optional[int] = None


class UrlReference(ReferenceCommon):
  type: str = "url"
  url: str


class McpReference(ReferenceCommon):
  type: str = "mcp"
  server: str
  tool: str
  params: Dict[str, Any] = Field(default_factory=dict)


ReferenceSpec = UrlReference | McpReference


class TaskSpec(BaseModel):
  title: str
  id: str
  description: str
  context: Optional[ContextSpec] = None
  references: List[ReferenceSpec] = Field(default_factory=list)
  output: str
  model: Optional[str] = None
  model_params: Dict[str, Any] = Field(default_factory=dict)
  mcp_error_mode: str = "skip_with_warning"
  style: Optional[str] = None
