from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

LATEST_PROTOCOL_VERSION = "2025-06-18"

TOOLS: List[Dict[str, Any]] = [
  {
    "name": "search",
    "title": "Search Notes",
    "description": "Search synthetic notes",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
      },
      "required": ["query"],
    },
  },
  {
    "name": "recent",
    "title": "Recent Notes",
    "description": "List recent synthetic updates",
    "inputSchema": {
      "type": "object",
      "properties": {
        "limit": {"type": "integer"},
      },
    },
  },
]

NOTE_ITEMS = [
  {
    "id": "notes#1",
    "title": "Notebook entry",
    "body": "Alpha body",
    "url": "https://example.com/spec",
  },
  {
    "id": "notes#2",
    "title": "Second entry",
    "body": "Beta body",
    "url": "https://example.com/second",
  },
]


def _write_response(payload: Dict[str, Any]) -> None:
  sys.stdout.write(json.dumps(payload) + "\n")
  sys.stdout.flush()


def _handle_initialize(message: Dict[str, Any]) -> Dict[str, Any]:
  return {
    "jsonrpc": "2.0",
    "id": message["id"],
    "result": {
      "protocolVersion": LATEST_PROTOCOL_VERSION,
      "serverInfo": {
        "name": "Notes Fixture Server",
        "version": "0.0.0",
      },
      "capabilities": {
        "tools": {"listChanged": False},
      },
    },
  }


def _handle_list_tools(message: Dict[str, Any]) -> Dict[str, Any]:
  return {
    "jsonrpc": "2.0",
    "id": message["id"],
    "result": {
      "tools": TOOLS,
      "nextCursor": None,
    },
  }


def _build_items(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
  name = arguments.get("name")
  limit = int(arguments.get("limit", 5) or 5)
  if name == "recent":
    items: List[Dict[str, Any]] = []
    for idx in range(1, max(1, limit) + 1):
      items.append(
        {
          "id": f"recent#{idx}",
          "title": f"Recent {idx}",
          "body": f"Update {idx}",
        }
      )
    return items
  query = arguments.get("query", "query")
  items = []
  for entry in NOTE_ITEMS:
    clone = dict(entry)
    clone["snippet"] = f"{query}: {entry['title']}"
    items.append(clone)
  return items[: max(1, limit)]


def _handle_call(message: Dict[str, Any]) -> Dict[str, Any]:
  params = message.get("params") or {}
  name = params.get("name")
  arguments = params.get("arguments") or {}
  arguments["name"] = name
  items = _build_items(arguments)
  return {
    "jsonrpc": "2.0",
    "id": message["id"],
    "result": {
      "content": [
        {
          "type": "text",
          "text": f"{len(items)} item(s)",
        }
      ],
      "structuredContent": {"items": items},
      "isError": False,
    },
  }


def _handle_message(raw: str) -> None:
  try:
    message = json.loads(raw)
  except json.JSONDecodeError:
    return
  if "id" not in message:
    return
  method = message.get("method")
  if method == "initialize":
    _write_response(_handle_initialize(message))
    return
  if method == "tools/list":
    _write_response(_handle_list_tools(message))
    return
  if method == "tools/call":
    _write_response(_handle_call(message))


def main() -> None:
  for line in sys.stdin:
    stripped = line.strip()
    if not stripped:
      continue
    _handle_message(stripped)


if __name__ == "__main__":
  main()
