from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from simple_rag_writer.config.loader import load_config


def test_load_config_round_trip(tmp_path: Path) -> None:
  cfg_text = dedent(
    """
    default_model: "openai:gpt-4.1-mini"
    providers:
      openai:
        type: "openai"
        api_key_env: "OPENAI_API_KEY"
      openrouter:
        type: "openrouter"
        api_key_env: "OPENROUTER_API_KEY"
        base_url: "https://openrouter.ai/api/v1"
    model_defaults:
      temperature: 0.3
    models:
      - id: "openai:gpt-4.1-mini"
        provider: "openai"
        model_name: "gpt-4.1-mini"
        params:
          response_format: {type: "text"}
      - id: "openrouter:meta-llama/3.1"
        provider: "openrouter"
        model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
        params:
          temperature: 0.2
    mcp_prompt_policy:
      default_mode: "summary"
      raw_capped:
        max_items_per_reference: 2
      summary:
        summarizer_model: "openai:gpt-4.1-mini"
        summary_max_tokens: 256
        per_type_prompts:
          file: "Summarize file content"
    logging:
      planning:
        enabled: true
        dir: "plan-logs"
        include_mcp_events: false
        mcp_inline: false
    """
  )

  path = tmp_path / "config.yaml"
  path.write_text(cfg_text, encoding="utf-8")

  cfg = load_config(path)
  assert cfg.config_path == path
  assert cfg.default_model == "openai:gpt-4.1-mini"
  assert "openai" in cfg.providers and cfg.providers["openai"].type == "openai"
  assert cfg.providers["openrouter"].base_url == "https://openrouter.ai/api/v1"
  assert cfg.model_defaults == {"temperature": 0.3}
  assert cfg.models[0].id == "openai:gpt-4.1-mini"
  assert cfg.models[1].params["temperature"] == 0.2
  assert cfg.mcp_prompt_policy.default_mode == "summary"
  assert cfg.logging.planning.dir == "plan-logs"


def test_load_config_missing_required_field(tmp_path: Path) -> None:
  cfg_text = dedent(
    """
    providers:
      openai:
        type: "openai"
    models:
      - id: "missing-default"
        provider: "openai"
        model_name: "gpt-4o-mini"
    """
  )
  path = tmp_path / "config.yaml"
  path.write_text(cfg_text, encoding="utf-8")

  with pytest.raises(ValidationError) as excinfo:
    load_config(path)

  assert "default_model" in str(excinfo.value)


def test_load_config_rejects_invalid_prompt_policy_mode(tmp_path: Path) -> None:
  cfg_text = dedent(
    """
    default_model: "openai:gpt-4.1-mini"
    providers:
      openai:
        type: "openai"
    models:
      - id: "openai:gpt-4.1-mini"
        provider: "openai"
        model_name: "gpt-4.1-mini"
    mcp_prompt_policy:
      default_mode: "invalid-mode"
    """
  )
  path = tmp_path / "config.yaml"
  path.write_text(cfg_text, encoding="utf-8")

  with pytest.raises(ValidationError) as excinfo:
    load_config(path)

  assert "default_mode" in str(excinfo.value)


def test_load_config_accepts_litellm_model_and_defaults(tmp_path: Path) -> None:
  cfg_text = dedent(
    """
    default_model: "openrouter:mistral-small"
    providers:
      openrouter:
        type: "openrouter"
        api_key_env: "OPENROUTER_API_KEY"
    models:
      - id: "openrouter:mistral-small"
        provider: "openrouter"
        litellm_model: "openrouter/mistralai/mistral-small"
        label: "Mistral Small"
        tags: ["mistral", "planning"]
        max_context_tokens: 131072
        default_params:
          temperature: 0.15
          top_p: 0.9
    """
  )
  path = tmp_path / "config.yaml"
  path.write_text(cfg_text, encoding="utf-8")

  cfg = load_config(path)

  model = cfg.models[0]
  assert model.id == "openrouter:mistral-small"
  assert model.model_name == "openrouter/mistralai/mistral-small"
  assert model.label == "Mistral Small"
  assert model.tags == ["mistral", "planning"]
  assert model.max_context_tokens == 131072
  assert model.params == {"temperature": 0.15, "top_p": 0.9}
