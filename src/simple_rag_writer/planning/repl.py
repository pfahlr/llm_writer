from __future__ import annotations

from typing import List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel

from simple_rag_writer.config.models import AppConfig
from simple_rag_writer.llm.registry import ModelRegistry
from simple_rag_writer.logging.planning_log import PlanningLogWriter
from simple_rag_writer.prompts.planning import (
  DEFAULT_HISTORY_WINDOW,
  build_planning_prompt,
)

console = Console()
HISTORY_WINDOW = DEFAULT_HISTORY_WINDOW


class PlanningRepl:
  def __init__(
    self,
    config: AppConfig,
    model_registry: ModelRegistry,
    log_writer: PlanningLogWriter,
  ) -> None:
    self._config = config
    self._registry = model_registry
    self._log = log_writer
    self._history: List[Tuple[str, str]] = []
    self._mcp_context: Optional[str] = None
    self._turn_index = 0

  def run(self) -> None:
    console.print(
      Panel(
        "Planning mode. Type to chat.\n"
        "/models to list models, /model <id> to switch, /quit to exit.",
        title="Simple Rag Writer",
      )
    )
    while True:
      try:
        line = console.input("[bold cyan]> [/bold cyan]").strip()
      except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Exiting planning mode.[/dim]")
        break

      if not line:
        continue

      if line.startswith("/"):
        if self._handle_command(line):
          break
        continue

      self._turn_index += 1
      self._log.start_turn(self._turn_index, line)

      window = max(HISTORY_WINDOW, 0)
      history_slice = self._history[-window:] if window else []
      prompt = build_planning_prompt(history_slice, line, self._mcp_context)
      output = self._registry.complete(prompt)
      self._log.log_model_used(self._registry.current_id)
      self._log.end_turn(self._turn_index, output)
      console.print(f"[bold green]{output}[/bold green]")
      self._history.append((line, output))

    self._log.close()

  def _handle_command(self, line: str) -> bool:
    if line in ("/quit", "/q", "/exit"):
      return True
    if line == "/models":
      models = self._registry.list_models()
      for m in models:
        mark = "*" if m.id == self._registry.current_id else " "
        console.print(f"{mark} {m.id} ({m.model_name})")
      return False
    if line.startswith("/model "):
      _, _, mid = line.partition(" ")
      mid = mid.strip()
      try:
        self._registry.set_current(mid)
        console.print(f"Switched model to [bold]{mid}[/bold]")
      except KeyError:
        console.print(f"[red]Unknown model id:[/red] {mid}")
      return False
    console.print(f"[yellow]Unknown command:[/yellow] {line}")
    return False
