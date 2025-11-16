from __future__ import annotations

from typing import Any, Dict, List, Optional

from .types import NormalizedItem


def normalize_payload(
  payload: Any,
  item_type_hint: Optional[str] = None,
) -> List[NormalizedItem]:
  """Convert raw MCP payload into a list of NormalizedItem."""
  items: List[NormalizedItem] = []

  if isinstance(payload, str):
    items.append(NormalizedItem(body=payload, type=item_type_hint))
  elif isinstance(payload, list):
    for idx, entry in enumerate(payload):
      if isinstance(entry, str):
        items.append(NormalizedItem(id=str(idx), body=entry, type=item_type_hint))
      elif isinstance(entry, dict):
        items.append(
          NormalizedItem(
            id=str(entry.get("id") or idx),
            type=entry.get("type", item_type_hint),
            title=entry.get("title"),
            snippet=entry.get("snippet"),
            body=entry.get("body"),
            url=entry.get("url"),
            metadata={
              k: v
              for k, v in entry.items()
              if k not in {"id", "type", "title", "snippet", "body", "url"}
            },
          )
        )
      else:
        items.append(
          NormalizedItem(
            id=str(idx),
            body=str(entry),
            type=item_type_hint,
          )
        )
  else:
    items.append(NormalizedItem(body=str(payload), type=item_type_hint))

  return items
