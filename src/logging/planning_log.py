from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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


class PlanningLogWriter:
  def __init__(self, path: Optional[Path], enabled: bool):
    self._enabled = enabled and path is not None
    self._path = path
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
    if not cfg.logging.planning.enabled:
      return cls(path=None, enabled=False)

    log_dir = Path(cfg.logging.planning.dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = log_dir / f"plan-{ts}.md"
    writer = cls(path=path, enabled=True)
    writer._write_header(config_path, default_model_id)
    return writer

  def _write_header(self, config_path: Optional[Path], default_model_id: str) -> None:
    if not self._enabled or not self._fh:
      return
    ts_iso = datetime.now().isoformat()
    self._fh.write(f"# Planning Session — {ts_iso}\n\n")
    self._fh.write(f"- Config: {config_path or '<unknown>'}\n")
    self._fh.write(f"- Default model: {default_model_id}\n")
    self._fh.write("- Models used: (populated during session)\n\n")
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
    if not self._enabled or not self._fh:
      return
    self._fh.write("### MCP References Injected\n\n")
    self._fh.write("| idx | server | tool | label | normalized_id | title |\n")
    self._fh.write("| --- | ------ | ---- | ----- | ------------- | ----- |\n")
    for item in items:
      self._fh.write(
        f"| {item.idx} | {item.server} | {item.tool} | {item.label or ''} | "
        f"{item.normalized_id or ''} | {item.title or ''} |\n"
      )
    self._fh.write("\n```mcp-yaml\n")
    self._fh.write("references:\n")
    for item in items:
      self._fh.write(f"  - idx: {item.idx}\n")
      self._fh.write(f"    server: {item.server}\n")
      self._fh.write(f"    tool: {item.tool}\n")
      if item.label:
        self._fh.write(f"    label: {item.label}\n")
      if item.normalized_id:
        self._fh.write(f"    normalized_id: {item.normalized_id}\n")
      if item.title:
        self._fh.write(f"    title: {item.title}\n")
      if item.type:
        self._fh.write(f"    type: {item.type}\n")
      if item.snippet:
        self._fh.write("    snippet: |\n      " + item.snippet + "\n")
      if item.body:
        self._fh.write("    body: |\n      " + item.body + "\n")
      if item.url:
        self._fh.write(f"    url: {item.url}\n")
      if item.metadata:
        self._fh.write(f"    metadata: {item.metadata}\n")
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
