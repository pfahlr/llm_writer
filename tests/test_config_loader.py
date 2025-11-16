from pathlib import Path

import pytest

from simple_rag_writer.config.loader import load_config


def test_load_config_round_trip(tmp_path: Path):
  pytest.skip("Implement config loader test once schemas settle")

  cfg_text = '''
  default_model: "openai:gpt-4.1-mini"
  providers:
    openai:
      type: "openai"
      api_key_env: "OPENAI_API_KEY"
  models:
    - id: "openai:gpt-4.1-mini"
      provider: "openai"
      model_name: "gpt-4.1-mini"
  '''
  path = tmp_path / "config.yaml"
  path.write_text(cfg_text, encoding="utf-8")

  cfg = load_config(path)
  assert cfg.default_model == "openai:gpt-4.1-mini"
  assert "openai" in cfg.providers
  assert cfg.models[0].id == "openai:gpt-4.1-mini"
