from __future__ import annotations

from pathlib import Path

from simple_rag_writer.planning.memory import ManualMemoryStore


def test_store_adds_and_persists_entries(tmp_path: Path) -> None:
  path = tmp_path / "memory.json"
  store = ManualMemoryStore(path=path)

  entry = store.add("Important detail about the project", label="Project notes")

  assert entry.entry_id
  entries = store.list_entries()
  assert len(entries) == 1
  assert entries[0].label == "Project notes"
  assert "project" in entries[0].text.lower()

  # Reload from disk and ensure the entry persists.
  reloaded = ManualMemoryStore(path=path)
  persisted = reloaded.list_entries()
  assert len(persisted) == 1
  assert persisted[0].entry_id == entry.entry_id


def test_store_delete_and_clear(tmp_path: Path) -> None:
  path = tmp_path / "memory.json"
  store = ManualMemoryStore(path=path)
  entry_a = store.add("First chunk")
  entry_b = store.add("Second chunk", label="Second")

  assert store.delete(entry_a.entry_id) is True
  assert store.delete("missing") is False
  assert [e.entry_id for e in store.list_entries()] == [entry_b.entry_id]

  store.clear()
  assert store.list_entries() == []
  reloaded = ManualMemoryStore(path=path)
  assert reloaded.list_entries() == []
