from __future__ import annotations

from typing import Any, List

import json

from .normalization import normalize_payload
from .types import NormalizedItem


def format_mcp_result_for_llm(server_id: str, tool_name: str, payload: Any) -> str:
  """Render MCP payloads into provider-friendly text for LLM messages."""

  def render_items(items: List[NormalizedItem]) -> List[str]:
    sections: List[str] = []
    for idx, item in enumerate(items, start=1):
      heading = item.title or item.id or f"Item {idx}"
      lines = [heading]
      body_text = (item.body or item.snippet or "").strip()
      if body_text:
        lines.append(body_text)
      if item.url:
        lines.append(f"URL: {item.url}")
      if item.metadata:
        metadata_json = json.dumps(item.metadata, ensure_ascii=False, default=str)
        lines.append(f"Metadata: {metadata_json}")
      block = "\n".join(line for line in lines if line).strip()
      if block:
        sections.append(block)
    return sections

  header = f"Result from {server_id}:{tool_name}"
  if isinstance(payload, dict):
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    return f"{header}\n\n```json\n{payload_json}\n```"

  sections = render_items(normalize_payload(payload))
  if sections:
    return "\n\n".join([header, *sections]).strip()

  text = str(payload or "").strip()
  if text:
    return f"{header}\n\n{text}"
  return header
