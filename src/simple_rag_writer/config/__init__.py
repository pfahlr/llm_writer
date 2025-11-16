from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .loader import load_config
from .models import AppConfig, ModelConfig, SkillConfig

DEFAULT_SYSTEM_PROMPT = (
  "You are a helpful writing assistant. Follow the provided system prompt if available."
)

__all__ = [
  "load_config",
  "load_prompts_config",
  "PromptsFile",
  "PromptDefinition",
  "ResolvedSkill",
  "SkillLibrary",
]


@dataclass(frozen=True)
class PromptDefinition:
  id: str
  label: str
  description: Optional[str] = None
  tags: tuple[str, ...] = ()
  category: Optional[str] = None
  model_hint: Optional[str] = None
  system_prompt: str = ""
  template_vars: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptsFile:
  spec_version: str
  validate_prompts: bool
  prompt_guidelines_url: Optional[str]
  prompts: Dict[str, PromptDefinition]

  @classmethod
  def empty(cls) -> "PromptsFile":
    return cls(
      spec_version="1.0.0",
      validate_prompts=False,
      prompt_guidelines_url=None,
      prompts={},
    )


@dataclass(frozen=True)
class ResolvedSkill:
  skill: SkillConfig
  model: ModelConfig
  prompt: Optional[PromptDefinition]
  system_prompt: str
  default_params: Dict[str, float | int]


def load_prompts_config(path: Path | str = "prompts.yaml") -> PromptsFile:
  prompt_path = Path(path)
  if not prompt_path.exists():
    raise FileNotFoundError(f"prompts file not found: {prompt_path}")
  with prompt_path.open("r", encoding="utf-8") as fh:
    payload = yaml.safe_load(fh) or {}
  try:
    raw_prompts = payload.get("prompts") or []
  except AttributeError as exc:  # pragma: no cover - defensive
    raise ValueError("prompts.yaml must be a mapping with a 'prompts' key.") from exc

  prompts: Dict[str, PromptDefinition] = {}
  for entry in raw_prompts:
    if not entry or not isinstance(entry, dict):
      continue
    identifier = entry.get("id")
    label = entry.get("label") or identifier
    system_prompt = entry.get("system_prompt") or ""
    if not identifier or not label:
      continue
    prompts[identifier] = PromptDefinition(
      id=identifier,
      label=label,
      description=entry.get("description"),
      tags=tuple(entry.get("tags") or ()),
      category=entry.get("category"),
      model_hint=entry.get("model_hint"),
      system_prompt=system_prompt.strip(),
      template_vars=tuple(entry.get("template_vars") or ()),
    )
  return PromptsFile(
    spec_version=str(payload.get("spec_version", "1.0.0")),
    validate_prompts=bool(payload.get("validate_prompts", True)),
    prompt_guidelines_url=payload.get("prompt_guidelines_url"),
    prompts=prompts,
  )


class SkillLibrary:
  def __init__(self, config: AppConfig, prompts: Optional[PromptsFile] = None):
    self._config = config
    self._prompts = prompts or PromptsFile.empty()
    self._models: Dict[str, ModelConfig] = {model.id: model for model in config.models}
    tool_config = config.llm_tool
    skills = tool_config.skills if tool_config else []
    self._skills: Dict[str, SkillConfig] = {skill.id: skill for skill in skills}

  def list_skill_ids(self) -> List[str]:
    return list(self._skills.keys())

  def resolve_model(self, model_id: str) -> ModelConfig:
    if model_id not in self._models:
      raise KeyError(f"Model '{model_id}' is not configured.")
    return self._models[model_id]

  def resolve_prompt(self, prompt_id: str) -> PromptDefinition:
    if prompt_id not in self._prompts.prompts:
      raise KeyError(f"Prompt '{prompt_id}' is not defined.")
    return self._prompts.prompts[prompt_id]

  def resolve_skill(self, skill_id: str) -> ResolvedSkill:
    if skill_id not in self._skills:
      raise KeyError(f"Skill '{skill_id}' is not defined.")
    skill = self._skills[skill_id]
    model = self.resolve_model(skill.model_id)
    prompt = self._resolve_prompt_optional(skill.prompt_id)
    system_prompt = self._derive_system_prompt(model, prompt)
    params = self._derive_default_params(skill)
    return ResolvedSkill(
      skill=skill,
      model=model,
      prompt=prompt,
      system_prompt=system_prompt,
      default_params=params,
    )

  def _resolve_prompt_optional(self, prompt_id: Optional[str]) -> Optional[PromptDefinition]:
    if not prompt_id:
      return None
    try:
      return self.resolve_prompt(prompt_id)
    except KeyError:
      raise KeyError(f"Prompt '{prompt_id}' is not defined.") from None

  def _derive_system_prompt(
    self,
    model: ModelConfig,
    prompt: Optional[PromptDefinition],
  ) -> str:
    if prompt and prompt.system_prompt:
      return prompt.system_prompt
    if model.system_prompt:
      return model.system_prompt.strip()
    return DEFAULT_SYSTEM_PROMPT

  @staticmethod
  def _derive_default_params(skill: SkillConfig) -> Dict[str, float | int]:
    params: Dict[str, float | int] = {}
    if skill.max_output_tokens:
      params["max_tokens"] = skill.max_output_tokens
    if skill.temperature is not None:
      params["temperature"] = skill.temperature
    return params
