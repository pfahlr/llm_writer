from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


@dataclass
class ManualMemoryEntry:
  entry_id: str
  label: Optional[str]
  text: str
  created_at: datetime

  def to_payload(self) -> dict:
    return {
      "entry_id": self.entry_id,
      "label": self.label,
      "text": self.text,
      "created_at": self.created_at.isoformat(),
    }

  @classmethod
  def from_payload(cls, payload: dict) -> "ManualMemoryEntry":
    created_text = payload.get("created_at")
    try:
      created_at = datetime.fromisoformat(created_text) if created_text else datetime.now(timezone.utc)
    except ValueError:
      created_at = datetime.now(timezone.utc)
    return cls(
      entry_id=str(payload.get("entry_id") or uuid.uuid4().hex[:8]),
      label=payload.get("label"),
      text=str(payload.get("text") or "").strip(),
      created_at=created_at,
    )


class ManualMemoryStore:
  """Stores manual context snippets for reuse in planning sessions."""

  def __init__(self, path: Optional[Path] = None):
    self._path = path
    self._entries: List[ManualMemoryEntry] = []
    if path:
      self._load()

  def list_entries(self) -> List[ManualMemoryEntry]:
    return sorted(self._entries, key=lambda entry: entry.created_at)

  def get(self, entry_id: str) -> Optional[ManualMemoryEntry]:
    for entry in self._entries:
      if entry.entry_id == entry_id:
        return entry
    return None

  def add(self, text: str, label: Optional[str] = None) -> ManualMemoryEntry:
    cleaned = (text or "").strip()
    if not cleaned:
      raise ValueError("Memory text cannot be empty.")
    entry = ManualMemoryEntry(
      entry_id=uuid.uuid4().hex[:8],
      label=(label or "").strip() or None,
      text=cleaned,
      created_at=datetime.now(timezone.utc),
    )
    self._entries.append(entry)
    self._persist()
    return entry

  def delete(self, entry_id: str) -> bool:
    for idx, entry in enumerate(self._entries):
      if entry.entry_id == entry_id:
        del self._entries[idx]
        self._persist()
        return True
    return False

  def clear(self) -> None:
    self._entries.clear()
    self._persist()

  def _load(self) -> None:
    if not self._path or not self._path.exists():
      return
    try:
      raw = json.loads(self._path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - corrupted files fall back to empty
      self._entries = []
      return
    if not isinstance(raw, list):
      self._entries = []
      return
    entries = []
    for payload in raw:
      if isinstance(payload, dict):
        entry = ManualMemoryEntry.from_payload(payload)
        if entry.text:
          entries.append(entry)
    self._entries = entries

  def _persist(self) -> None:
    if not self._path:
      return
    self._path.parent.mkdir(parents=True, exist_ok=True)
    payload = [entry.to_payload() for entry in self._entries]
    self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
