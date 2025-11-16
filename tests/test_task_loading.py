from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from simple_rag_writer.tasks.loader import expand_task_paths, load_task


def _write_task(tmp_path: Path, name: str, body: str) -> Path:
  path = tmp_path / name
  path.write_text(dedent(body), encoding="utf-8")
  return path


def test_load_minimal_task_spec(tmp_path: Path) -> None:
  task_path = _write_task(
    tmp_path,
    "task_minimal.yaml",
    """
    title: "Chapter Prelude"
    id: "book1-prelude"
    description: "Draft the prelude."
    output: "drafts/prelude.md"
    """,
  )

  task = load_task(task_path)

  assert task.title == "Chapter Prelude"
  assert task.id == "book1-prelude"
  assert task.references == []
  assert task.context is None
  assert task.model is None
  assert task.model_params == {}
  assert task.mcp_error_mode == "skip_with_warning"
  assert task.output == "drafts/prelude.md"


def test_load_task_with_url_and_mcp_references(tmp_path: Path) -> None:
  task_path = _write_task(
    tmp_path,
    "task_with_refs.yaml",
    """
    title: "Case Study"
    id: "case-01"
    description: "Summarize a case."
    output: "drafts/case.md"
    references:
      - type: "url"
        label: "Spec"
        url: "https://example.com/spec"
        prompt_mode: "summary"
        max_items: 3
      - type: "mcp"
        label: "Case notes"
        server: "notes"
        tool: "search_notes"
        item_type: "note"
        params:
          query: "case alpha"
    """,
  )

  task = load_task(task_path)

  assert len(task.references) == 2
  url_ref = task.references[0]
  assert url_ref.type == "url"
  assert url_ref.url == "https://example.com/spec"
  assert url_ref.label == "Spec"
  assert url_ref.prompt_mode == "summary"
  assert url_ref.max_items == 3

  mcp_ref = task.references[1]
  assert mcp_ref.type == "mcp"
  assert mcp_ref.server == "notes"
  assert mcp_ref.tool == "search_notes"
  assert mcp_ref.item_type == "note"
  assert mcp_ref.params == {"query": "case alpha"}


def test_expand_task_paths_with_globs_and_dedup(tmp_path: Path) -> None:
  second = _write_task(
    tmp_path,
    "b_task.yaml",
    """
    title: "Second"
    id: "b"
    description: "Second task"
    output: "drafts/b.md"
    """,
  ).resolve()
  first = _write_task(
    tmp_path,
    "a_task.yaml",
    """
    title: "First"
    id: "a"
    description: "First task"
    output: "drafts/a.md"
    """,
  ).resolve()

  expanded = expand_task_paths(
    [
      second,
      tmp_path / "*.yaml",
    ]
  )

  assert expanded == [second, first]
