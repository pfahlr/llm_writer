from __future__ import annotations

from simple_rag_writer.mcp.normalization import normalize_payload


def test_string_payload_normalizes_to_single_item() -> None:
  result = normalize_payload("plain text blob", item_type_hint="note")

  assert len(result) == 1
  item = result[0]
  assert item.type == "note"
  assert item.body == "plain text blob"
  assert item.metadata == {}


def test_list_of_strings_gets_index_ids() -> None:
  result = normalize_payload(["alpha", "beta"])

  assert [item.id for item in result] == ["0", "1"]
  assert [item.body for item in result] == ["alpha", "beta"]


def test_list_of_dicts_maps_known_fields_and_metadata() -> None:
  payload = [
    {
      "id": "custom",
      "type": "document",
      "title": "Doc Title",
      "snippet": "Short snippet",
      "body": "Full body text",
      "url": "https://example.test/doc",
      "extra": 123,
    }
  ]

  result = normalize_payload(payload, item_type_hint="fallback")

  assert len(result) == 1
  item = result[0]
  assert item.id == "custom"
  assert item.type == "document"
  assert item.title == "Doc Title"
  assert item.snippet == "Short snippet"
  assert item.body == "Full body text"
  assert item.url == "https://example.test/doc"
  assert item.metadata == {"extra": 123}


def test_list_entries_with_unknown_type_are_stringified() -> None:
  result = normalize_payload(["alpha", 7])

  assert len(result) == 2
  assert result[0].body == "alpha"
  assert result[1].body == "7"


def test_unknown_payload_is_stringified() -> None:
  class Dummy:
    def __str__(self) -> str:  # pragma: no cover - trivial
      return "Dummy payload"

  [item] = normalize_payload(Dummy())
  assert item.body == "Dummy payload"
  assert item.metadata == {}
