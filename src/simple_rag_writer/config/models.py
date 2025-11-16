from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
  type: str
  api_key_env: Optional[str] = None
  api_key: Optional[str] = None
  base_url: Optional[str] = None
  model_prefix: Optional[str] = None


class ModelConfig(BaseModel):
  id: str
  provider: str
  model_name: str
  params: Dict[str, Any] = Field(default_factory=dict)


class RawCappedPolicy(BaseModel):
  max_items_per_reference: int = 3
  max_chars_per_item: int = 800
  max_total_chars: int = 8000


class SummaryPolicy(BaseModel):
  summarizer_model: str = "openai:gpt-4.1-mini"
  max_items_per_reference: int = 10
  summary_max_tokens: int = 512
  per_type_prompts: Dict[str, str] = Field(default_factory=dict)
  default_prompt: Optional[str] = None


class McpPromptPolicy(BaseModel):
  default_mode: Literal["raw_capped", "summary"] = "raw_capped"
  raw_capped: RawCappedPolicy = Field(default_factory=RawCappedPolicy)
  summary: SummaryPolicy = Field(default_factory=SummaryPolicy)


class McpServerConfig(BaseModel):
  id: str
  command: List[str]
  auto_start: bool = True


class PlanningLoggingConfig(BaseModel):
  enabled: bool = True
  dir: str = "logs"
  include_mcp_events: bool = True
  mcp_inline: bool = True


class LoggingConfig(BaseModel):
  planning: PlanningLoggingConfig = Field(default_factory=PlanningLoggingConfig)


class AppConfig(BaseModel):
  default_model: str
  providers: Dict[str, ProviderConfig]
  model_defaults: Dict[str, Any] = Field(default_factory=dict)
  models: List[ModelConfig]
  mcp_servers: List[McpServerConfig] = Field(default_factory=list)
  mcp_prompt_policy: McpPromptPolicy = Field(default_factory=McpPromptPolicy)
  logging: LoggingConfig = Field(default_factory=LoggingConfig)

  @property
  def config_path(self) -> Optional[Path]:
    return getattr(self, "_config_path", None)

  def with_path(self, path: Path) -> "AppConfig":
    object.__setattr__(self, "_config_path", path)
    return self
