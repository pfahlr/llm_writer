from __future__ import annotations

from typing import Any, Dict, List, Optional

from simple_rag_writer.tasks.models import TaskSpec


def build_task_prompt(
  task: TaskSpec,
  outline_context: Optional[Dict[str, Any]],
  reference_blobs: List[str],
) -> str:
  parts: List[str] = []
  parts.append(f"# Task: {task.title}\n")
  parts.append("## Description\n")
  parts.append(task.description)
  if task.style:
    parts.append(f"\n\nStyle: {task.style}")
  if outline_context:
    parts.append("\n\n## Outline Context\n")
    parts.append(_format_outline_context(outline_context))
  if reference_blobs:
    parts.append("\n\n## Reference material\n")
    for idx, blob in enumerate(reference_blobs, start=1):
      parts.append(f"\n### Reference {idx}\n")
      parts.append(blob)
  parts.append("\n\n## Instructions\n")
  parts.append(
    "Write a complete draft in Markdown for the section described above. "
    "Use the outline and references as guidance, but do not mention them explicitly "
    "unless the instructions say otherwise."
  )
  return "\n".join(parts)


def _format_outline_context(context: Dict[str, Any]) -> str:
  """Format outline context dictionary into readable text for LLM."""
  context_parts = []

  # Part info
  if context.get("part"):
    part = context["part"]
    part_text = f"**Part:** {part['title']}"
    if part.get("summary"):
      part_text += f" — {part['summary']}"
    context_parts.append(part_text)

  # Section info
  if context.get("section"):
    sec = context["section"]
    context_parts.append(f"**Section Summary:** {sec['summary']}")

  # Navigation context
  if context.get("previous_section"):
    prev = context["previous_section"]
    context_parts.append(f"**Previous:** {prev['title']}")

  if context.get("next_section"):
    nxt = context["next_section"]
    context_parts.append(f"**Next:** {nxt['title']}")

  # Sibling context (brief list)
  if context.get("sibling_sections"):
    siblings = context["sibling_sections"]
    sibling_list = "\n".join(f"  - {s['title']}: {s['summary']}" for s in siblings)
    context_parts.append(f"**Chapter Structure:**\n{sibling_list}")

  return "\n\n".join(context_parts)
