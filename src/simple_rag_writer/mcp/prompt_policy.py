from __future__ import annotations

from typing import List, Optional

from simple_rag_writer.config.models import AppConfig, RawCappedPolicy, SummaryPolicy
from simple_rag_writer.llm.registry import ModelRegistry
from simple_rag_writer.mcp.types import NormalizedItem
from simple_rag_writer.tasks.models import ReferenceCommon

DEFAULT_SUMMARY_PROMPT = (
  "Summarize the following reference material into key bullet points, highlighting"
  " actionable facts and important context."
)


def _format_item(item: NormalizedItem) -> str:
  parts: List[str] = []
  if item.title:
    parts.append(f"Title: {item.title}")
  if item.url:
    parts.append(f"URL: {item.url}")
  body = (item.body or item.snippet or "").strip()
  if body:
    parts.append(body)
  elif item.metadata:
    parts.append(str(item.metadata))
  return "\n".join(parts).strip()


def _render_raw_blob(reference: ReferenceCommon, items: List[NormalizedItem], policy: RawCappedPolicy) -> Optional[str]:
  if not items:
    return None
  per_item_limit = reference.max_chars or policy.max_chars_per_item
  total_limit = policy.max_total_chars
  max_items = reference.max_items or policy.max_items_per_reference
  chunks: List[str] = []
  used = 0
  for item in items[:max_items]:
    formatted = _format_item(item)
    if not formatted:
      continue
    snippet = formatted[:per_item_limit].strip()
    if not snippet:
      continue
    if used + len(snippet) > total_limit:
      remaining = total_limit - used
      if remaining <= 0:
        break
      snippet = snippet[:remaining].rstrip()
    chunks.append(snippet)
    used += len(snippet)
    if used >= total_limit:
      break
  if not chunks:
    return None
  return "\n\n".join(chunks)


def _build_summary_prompt(reference: ReferenceCommon, items: List[NormalizedItem], policy: SummaryPolicy) -> str:
  ref_type = reference.item_type or next((itm.type for itm in items if itm.type), None)
  template = None
  if ref_type:
    template = policy.per_type_prompts.get(ref_type)
  if not template:
    template = policy.default_prompt or DEFAULT_SUMMARY_PROMPT
  sections: List[str] = []
  for idx, item in enumerate(items, start=1):
    heading = item.title or f"Item {idx}"
    body = (item.body or item.snippet or "").strip()
    lines = [f"## {heading}"]
    if body:
      lines.append(body)
    if item.url:
      lines.append(f"Source: {item.url}")
    sections.append("\n".join(lines).strip())
  body_text = "\n\n".join(sections)
  return f"{template.strip()}\n\n{body_text}".strip()


def _render_summary_blob(
  reference: ReferenceCommon,
  items: List[NormalizedItem],
  policy: SummaryPolicy,
  registry: ModelRegistry,
) -> Optional[str]:
  if not items:
    return None
  max_items = reference.max_items or policy.max_items_per_reference
  limited = items[:max_items]
  prompt = _build_summary_prompt(reference, limited, policy)
  max_tokens = reference.summary_max_tokens or policy.summary_max_tokens
  summary = registry.complete(
    prompt,
    model_id=policy.summarizer_model,
    task_params={"max_tokens": max_tokens},
  )
  return summary.strip() or None


def apply_prompt_policy(
  config: AppConfig,
  items: List[NormalizedItem],
  reference: ReferenceCommon,
  registry: ModelRegistry,
) -> Optional[str]:
  if not items:
    return None
  policy = config.mcp_prompt_policy
  mode = reference.prompt_mode or policy.default_mode
  if mode == "summary":
    return _render_summary_blob(reference, items, policy.summary, registry)
  return _render_raw_blob(reference, items, policy.raw_capped)
