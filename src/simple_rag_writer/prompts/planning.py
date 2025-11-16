from __future__ import annotations

from typing import List, Optional, Tuple

DEFAULT_HISTORY_WINDOW = 5


def build_planning_prompt(
  history: List[Tuple[str, str]],
  user_message: str,
  mcp_context: Optional[str] = None,
  history_window: int = DEFAULT_HISTORY_WINDOW,
) -> str:
  """Build planning prompt from optional MCP context, recent history, and new user text."""
  window = max(history_window, 0)
  recent_history = history[-window:] if window else []

  parts: List[str] = [
    (
      "You are a planning assistant helping the user design and refine writing tasks.\n"
      "Respond concisely but clearly, and propose concrete steps or task specs when useful."
    )
  ]

  if mcp_context:
    parts.extend(
      [
        "",
        "## Context from external sources",
        mcp_context.strip(),
      ]
    )

  history_heading = "## Conversation so far"
  if window:
    history_heading += f" (last {window} turns)"
  parts.append("")
  parts.append(history_heading)
  if recent_history:
    for user_text, assistant_text in recent_history:
      parts.append("")
      parts.append(f"User: {user_text}")
      parts.append(f"Assistant: {assistant_text}")
  else:
    parts.append("")
    parts.append("No previous conversation yet.")

  parts.extend(
    [
      "",
      "## New message",
      user_message.strip(),
    ]
  )

  return "\n".join(parts).strip()
