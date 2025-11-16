from __future__ import annotations

from typing import List, Optional, Tuple


def build_planning_prompt(
  history: List[Tuple[str, str]],
  user_message: str,
  mcp_context: Optional[str] = None,
) -> str:
  """Build planning-mode prompt from history, optional MCP context, and the new message."""
  parts: List[str] = []
  parts.append(
    "You are a planning assistant helping the user design and refine writing tasks.\n"
    "Respond concisely but clearly, and propose concrete steps or task specs when useful."
  )
  if mcp_context:
    parts.append("\n## Context from external sources\n")
    parts.append(mcp_context)
  parts.append("\n## Conversation so far\n")
  for u, a in history:
    parts.append(f"User: {u}\nAssistant: {a}\n")
  parts.append("\n## New message\n")
  parts.append(user_message)
  return "\n".join(parts)
