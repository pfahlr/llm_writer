from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from simple_rag_writer.config.models import AppConfig


@dataclass
class McpLogItem:
  idx: int
  server: str
  tool: str
  label: Optional[str]
  normalized_id: Optional[str]
  title: Optional[str]
  type: Optional[str]
  snippet: Optional[str]
  body: Optional[str]
  url: Optional[str]
  metadata: Dict[str, Any] = field(default_factory=dict)


MODELS_PLACEHOLDER = "- Models used: __PENDING__"


class PlanningLogWriter:
  def __init__(
    self,
    path: Optional[Path],
    enabled: bool,
    *,
    include_mcp_events: bool = True,
    mcp_inline: bool = True,
  ):
    self._enabled = enabled and path is not None
    self._path = path
    self._include_mcp_events = include_mcp_events
    self._mcp_inline = mcp_inline
    self._fh = path.open("w", encoding="utf-8") if self._enabled else None
    self._models_used: Set[str] = set()
    self._header_written = False

  @classmethod
  def from_config(
    cls,
    cfg: AppConfig,
    config_path: Optional[Path],
    default_model_id: str,
  ) -> "PlanningLogWriter":
    planning_cfg = cfg.logging.planning
    if not planning_cfg.enabled:
      return cls(path=None, enabled=False)

    log_dir = Path(planning_cfg.dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = log_dir / f"plan-{ts}.md"
    writer = cls(
      path=path,
      enabled=True,
      include_mcp_events=planning_cfg.include_mcp_events,
      mcp_inline=planning_cfg.mcp_inline,
    )
    writer._write_header(config_path, default_model_id)
    return writer

  def _write_header(self, config_path: Optional[Path], default_model_id: str) -> None:
    if not self._enabled or not self._fh:
      return
    ts_iso = datetime.now().isoformat()
    self._fh.write(f"# Planning Session — {ts_iso}\n\n")
    self._fh.write(f"- Config: {config_path or '<unknown>'}\n")
    self._fh.write(f"- Default model: {default_model_id}\n")
    self._fh.write(f"{MODELS_PLACEHOLDER}\n\n")
    self._header_written = True

  def log_model_used(self, model_id: str) -> None:
    if not self._enabled:
      return
    self._models_used.add(model_id)

  def start_turn(self, turn_index: int, user_text: str) -> None:
    if not self._enabled or not self._fh:
      return
    self._fh.write(f"## Turn {turn_index}\n\n")
    self._fh.write("**User:**\n\n")
    self._fh.write(user_text + "\n\n")

  def log_mcp_injection(self, turn_index: int, items: List[McpLogItem]) -> None:
    if (
      not self._enabled
      or not self._fh
      or not self._include_mcp_events
      or not self._mcp_inline
      or not items
    ):
      return
    self._fh.write("### MCP References Injected\n\n")
    self._fh.write("| idx | server | tool | label | normalized_id | title |\n")
    self._fh.write("| --- | ------ | ---- | ----- | ------------- | ----- |\n")
    for item in items:
      self._fh.write(
        f"| {item.idx} | {item.server} | {item.tool} | {item.label or ''} | "
        f"{item.normalized_id or ''} | {item.title or ''} |\n"
      )
    payload = {"references": [self._serialize_mcp_item(item) for item in items]}
    yaml_text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    self._fh.write("\n```mcp-yaml\n")
    self._fh.write(yaml_text)
    if not yaml_text.endswith("\n"):
      self._fh.write("\n")
    self._fh.write("```\n\n")

  def end_turn(self, turn_index: int, assistant_text: str) -> None:
    if not self._enabled or not self._fh:
      return
    self._fh.write("**Assistant:**\n\n")
    self._fh.write(assistant_text + "\n\n")

  def close(self) -> None:
    if self._fh:
      self._fh.close()
      self._fh = None
    if self._enabled and self._path and self._path.exists():
      self._finalize_models_used()

  def _serialize_mcp_item(self, item: McpLogItem) -> Dict[str, Any]:
    data: Dict[str, Any] = {
      "idx": item.idx,
      "server": item.server,
      "tool": item.tool,
    }
    if item.label is not None:
      data["label"] = item.label
    if item.normalized_id is not None:
      data["normalized_id"] = item.normalized_id
    if item.title is not None:
      data["title"] = item.title
    if item.type is not None:
      data["type"] = item.type
    if item.snippet is not None:
      data["snippet"] = item.snippet
    if item.body is not None:
      data["body"] = item.body
    if item.url is not None:
      data["url"] = item.url
    if item.metadata:
      data["metadata"] = item.metadata
    return data

  def _finalize_models_used(self) -> None:
    if not self._path:
      return
    text = self._path.read_text(encoding="utf-8")
    if MODELS_PLACEHOLDER not in text:
      return
    if self._models_used:
      models_text = ", ".join(sorted(self._models_used))
    else:
      models_text = "<none>"
    updated = text.replace(MODELS_PLACEHOLDER, f"- Models used: {models_text}", 1)
    self._path.write_text(updated, encoding="utf-8")
