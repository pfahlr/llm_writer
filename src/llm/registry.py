from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import litellm

from simple_rag_writer.config.models import AppConfig, ModelConfig, ProviderConfig
from .params import merge_generation_params


class ModelRegistry:
  def __init__(self, config: AppConfig):
    self._config = config
    self._models: Dict[str, ModelConfig] = {m.id: m for m in config.models}
    if config.default_model not in self._models:
      raise ValueError(f"default_model {config.default_model} not found in models")
    self._current_id = config.default_model

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
      raise ValueError(f"Unknown provider {m.provider} for model {m.id}")
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
  ) -> str:
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

    response = litellm.completion(
      messages=[{"role": "user", "content": prompt}],
      **kwargs,
    )
    message = response.choices[0].message
    return getattr(message, "content", "") or ""
