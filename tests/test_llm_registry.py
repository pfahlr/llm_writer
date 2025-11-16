import pytest

from simple_rag_writer.config.models import AppConfig, ProviderConfig, ModelConfig
from simple_rag_writer.llm.registry import ModelRegistry


def test_model_registry_sets_default_model():
  pytest.skip("Needs litellm mocking before enabling")

  cfg = AppConfig(
    default_model="openai:gpt-4.1-mini",
    providers={
      "openai": ProviderConfig(type="openai", api_key_env="OPENAI_API_KEY"),
    },
    models=[
      ModelConfig(
        id="openai:gpt-4.1-mini",
        provider="openai",
        model_name="gpt-4.1-mini",
      )
    ],
  )
  registry = ModelRegistry(cfg)
  assert registry.current_id == "openai:gpt-4.1-mini"
