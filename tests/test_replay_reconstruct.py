from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from simple_rag_writer.replay.reconstruct import reconstruct_prompt_for_turn


def _write_log(tmp_path: Path, text: str) -> Path:
  path = tmp_path / "plan-log.md"
  path.write_text(dedent(text).lstrip("\n"), encoding="utf-8")
  return path


def test_reconstruct_prompt_includes_previous_turn_context(tmp_path: Path) -> None:
  log_path = _write_log(
    tmp_path,
    """
    # Planning Session — 2024-05-01
    - Config: config.yaml
    - Default model: writer
    - Models used: writer

    ## Turn 1

    **User:**

    Need help outlining the chapter.

    **Assistant:**

    Start by listing sections and transitions.

    ## Turn 2

    **User:**

    Can you expand the outline with more details?

    **Assistant:**

    Let's add bullet points per section.
    """,
  )

  prompt, meta = reconstruct_prompt_for_turn(log_path, 2)

  assert "You are a planning assistant" in prompt
  assert "User: Need help outlining the chapter." in prompt
  assert "Assistant: Start by listing sections and transitions." in prompt
  assert prompt.strip().endswith("Can you expand the outline with more details?")
  assert meta.turn_index == 2
  assert meta.user_text == "Can you expand the outline with more details?"
  assert meta.history_turns == (1,)


def test_reconstruct_prompt_includes_mcp_yaml_context(tmp_path: Path) -> None:
  log_path = _write_log(
    tmp_path,
    """
    # Planning Session — 2024-05-01
    - Config: config.yaml
    - Default model: writer
    - Models used: writer

    ## Turn 1

    **User:**

    Summarize new notes.

    **Assistant:**

    Sure thing.

    ## Turn 2

    **User:**

    Use the MCP references from earlier.

    ### MCP References Injected

    | idx | server | tool | label | normalized_id | title |
    | --- | ------ | ---- | ----- | ------------- | ----- |
    | 1 | notes | search | Spec | notes#1 | Outline draft |

    ```mcp-yaml
    references:
      - idx: 1
        server: notes
        tool: search
        label: Spec
        normalized_id: notes#1
        title: Outline draft
        snippet: First few lines
        body: Full body text
      - idx: 2
        server: notes
        tool: search
        label: Plan
        normalized_id: notes#2
        title: Planning memo
        snippet: Another snippet
        body: Another body text
    ```

    **Assistant:**

    Injected the MCP references.
    """,
  )

  prompt, meta = reconstruct_prompt_for_turn(log_path, 2)

  assert "## Context from external sources" in prompt
  assert "references:" in prompt
  assert "title: Outline draft" in prompt
  assert meta.mcp_reference_count == 2


def test_reconstruct_prompt_errors_when_turn_missing(tmp_path: Path) -> None:
  log_path = _write_log(
    tmp_path,
    """
    # Planning Session — 2024-05-01
    - Config: config.yaml
    - Default model: writer
    - Models used: writer

    ## Turn 1

    **User:**

    Hello

    **Assistant:**

    Hi
    """,
  )

  with pytest.raises(ValueError):
    reconstruct_prompt_for_turn(log_path, 3)
