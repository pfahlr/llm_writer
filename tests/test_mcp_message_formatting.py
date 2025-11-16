from __future__ import annotations

from simple_rag_writer.mcp.message_formatting import format_mcp_result_for_llm


def test_format_dict_payload_produces_json_block():
  payload = {"total_results": 12, "papers": [{"title": "LLM education"}]}

  content = format_mcp_result_for_llm("arxiv", "search_papers", payload)

  assert "Result from arxiv:search_papers" in content
  assert "```json" in content
  assert '"total_results": 12' in content
  assert "{'" not in content


def test_format_items_includes_metadata_as_json():
  payload = [
    {
      "title": "Doc",
      "body": "Body text",
      "url": "https://example.com/doc",
      "extra": {"key": "value"},
    }
  ]

  content = format_mcp_result_for_llm("notes", "search", payload)

  assert "Doc" in content
  assert "Body text" in content
  assert "URL: https://example.com/doc" in content
  assert '"key": "value"' in content


def test_format_handles_empty_payload_gracefully():
  content = format_mcp_result_for_llm("notes", "missing", None)

  assert "Result from notes:missing" in content
