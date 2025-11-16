from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from simple_rag_writer.config import (
  PromptsFile,
  PromptDefinition,
  SkillLibrary,
  load_prompts_config,
)
from simple_rag_writer.config.models import (
  AppConfig,
  LlmToolConfig,
  ModelConfig,
  ProviderConfig,
  SkillConfig,
)


def test_load_prompts_config_parses_entries(tmp_path: Path) -> None:
  text = dedent(
    """
    spec_version: 1.0.0
    validate_prompts: true
    prompt_guidelines_url: https://example.com/prompts
    prompts:
      - id: essay_drafter
        label: Essay Drafting Assistant
        description: Helps draft essays.
        tags: [essay, drafting]
        category: drafting
        system_prompt: |
          You are an essay assistant.
      - id: creative_story
        label: Creative Storyteller
        tags: [story]
        system_prompt: |
          Tell imaginative tales.
        template_vars: [character_name]
    """
  )
  path = tmp_path / "prompts.yaml"
  path.write_text(text, encoding="utf-8")

  prompts = load_prompts_config(path)

  assert prompts.spec_version == "1.0.0"
  assert prompts.validate_prompts is True
  assert prompts.prompt_guidelines_url == "https://example.com/prompts"
  assert "essay_drafter" in prompts.prompts
  assert prompts.prompts["essay_drafter"].tags == ("essay", "drafting")
  assert prompts.prompts["creative_story"].template_vars == ("character_name",)


def test_skill_library_resolves_skills_with_prompt_and_overrides(tmp_path: Path) -> None:
  prompts = PromptsFile(
    spec_version="1.0.0",
    validate_prompts=True,
    prompt_guidelines_url=None,
    prompts={
      "essay_drafter": PromptDefinition(
        id="essay_drafter",
        label="Essay Draft Assistant",
        description="Draft structured essays.",
        tags=("essay",),
        category="drafting",
        model_hint=None,
        system_prompt="You are Essay Pro.",
        template_vars=(),
      )
    },
  )
  config = AppConfig(
    default_model="model-a",
    providers={"local": ProviderConfig(type="openai", api_key="dummy")},
    models=[
      ModelConfig(
        id="model-a",
        provider="local",
        model_name="model-a",
        system_prompt="Model A default prompt.",
      ),
      ModelConfig(
        id="model-b",
        provider="local",
        model_name="model-b",
        system_prompt="Fallback prompt.",
      ),
    ],
    llm_tool=LlmToolConfig(
      skills=[
        SkillConfig(
          id="essay",
          label="Essay Drafter",
          description="Writes essays",
          model_id="model-a",
          prompt_id="essay_drafter",
          max_output_tokens=800,
          temperature=0.4,
        ),
        SkillConfig(
          id="summary",
          label="Summarizer",
          description="Summarize text",
          model_id="model-b",
          prompt_id=None,
        ),
      ],
      default_skill="essay",
    ),
  )

  library = SkillLibrary(config, prompts)

  resolved = library.resolve_skill("essay")
  assert resolved.model.id == "model-a"
  assert resolved.prompt and resolved.prompt.id == "essay_drafter"
  assert resolved.system_prompt == "You are Essay Pro."
  assert resolved.default_params == {"max_tokens": 800, "temperature": 0.4}

  fallback = library.resolve_skill("summary")
  assert fallback.prompt is None
  assert fallback.system_prompt == "Fallback prompt."
  assert fallback.default_params == {}


def test_skill_library_errors_on_unknown_skill(tmp_path: Path) -> None:
  prompts = PromptsFile(
    spec_version="1.0.0",
    validate_prompts=True,
    prompt_guidelines_url=None,
    prompts={},
  )
  config = AppConfig(
    default_model="model-a",
    providers={"local": ProviderConfig(type="openai", api_key="dummy")},
    models=[
      ModelConfig(id="model-a", provider="local", model_name="model-a", system_prompt=None),
    ],
    llm_tool=LlmToolConfig(
      skills=[
        SkillConfig(
          id="essay",
          label="Essay Drafter",
          description=None,
          model_id="model-a",
          prompt_id=None,
        )
      ],
      default_skill="essay",
    ),
  )

  library = SkillLibrary(config, prompts)

  with pytest.raises(KeyError):
    library.resolve_skill("missing")
