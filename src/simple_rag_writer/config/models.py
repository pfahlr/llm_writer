from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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
  label: Optional[str] = None
  system_prompt: Optional[str] = None
  tags: List[str] = Field(default_factory=list)
  max_context_tokens: Optional[int] = None

  @model_validator(mode="before")
  @classmethod
  def _apply_aliases(cls, data: Any) -> Any:
    if not isinstance(data, dict):
      return data
    normalized = dict(data)
    if "model_name" not in normalized:
      alias_value = normalized.get("litellm_model") or normalized.get("model")
      if alias_value:
        normalized["model_name"] = alias_value
    if "params" not in normalized and "default_params" in normalized:
      defaults = normalized.pop("default_params") or {}
      normalized["params"] = defaults
    return normalized


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


class SkillConfig(BaseModel):
  id: str
  label: str
  description: Optional[str] = None
  model_id: str
  prompt_id: Optional[str] = None
  max_output_tokens: Optional[int] = None
  temperature: Optional[float] = None


class LlmToolConfig(BaseModel):
  id: str = "llm"
  tool_name: str = "llm"
  title: Optional[str] = None
  description: Optional[str] = None
  default_skill: Optional[str] = None
  skills: List[SkillConfig] = Field(default_factory=list)
  max_tokens_limit: Optional[int] = None


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
  llm_tool: Optional[LlmToolConfig] = None
  logging: LoggingConfig = Field(default_factory=LoggingConfig)

  @property
  def config_path(self) -> Optional[Path]:
    return getattr(self, "_config_path", None)

  def with_path(self, path: Path) -> "AppConfig":
    object.__setattr__(self, "_config_path", path)
    return self
