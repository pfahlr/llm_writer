from __future__ import annotations

from typing import List, Optional

from simple_rag_writer.tasks.models import TaskSpec


def build_task_prompt(
  task: TaskSpec,
  outline_context: Optional[str],
  reference_blobs: List[str],
) -> str:
  parts: List[str] = []
  parts.append(f"# Task: {task.title}\n")
  parts.append("## Description\n")
  parts.append(task.description)
  if task.style:
    parts.append(f"\n\nStyle: {task.style}")
  if outline_context:
    parts.append("\n\n## Outline context\n")
    parts.append(outline_context)
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
