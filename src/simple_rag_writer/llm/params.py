from __future__ import annotations

from typing import Any, Dict, Optional

from simple_rag_writer.config.models import AppConfig, ModelConfig


def merge_generation_params(
  app_config: AppConfig,
  model_config: ModelConfig,
  task_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  params: Dict[str, Any] = {}
  params.update(app_config.model_defaults or {})
  params.update(model_config.params or {})
  if task_params:
    params.update(task_params)
  return params
