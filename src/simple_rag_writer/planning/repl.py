from __future__ import annotations

from dataclasses import dataclass
import shlex
from typing import List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from simple_rag_writer.config.models import AppConfig
from simple_rag_writer.llm.registry import ModelRegistry
from simple_rag_writer.logging.planning_log import McpLogItem, PlanningLogWriter
from simple_rag_writer.mcp.client import McpClient
from simple_rag_writer.mcp.normalization import normalize_payload
from simple_rag_writer.mcp.types import NormalizedItem
from simple_rag_writer.prompts.planning import (
  DEFAULT_HISTORY_WINDOW,
  build_planning_prompt,
)
from simple_rag_writer.runner.url_fetcher import build_url_items, fetch_url_text
from simple_rag_writer.tasks.models import UrlReference

console = Console()
HISTORY_WINDOW = DEFAULT_HISTORY_WINDOW


@dataclass
class _ResultBatch:
  source: str
  items: List[NormalizedItem]
  label: Optional[str] = None
  server: Optional[str] = None
  tool: Optional[str] = None
  url: Optional[str] = None


class PlanningRepl:
  def __init__(
    self,
    config: AppConfig,
    model_registry: ModelRegistry,
    log_writer: PlanningLogWriter,
    mcp_client: Optional[McpClient] = None,
  ) -> None:
    self._config = config
    self._registry = model_registry
    self._log = log_writer
    self._mcp_client = mcp_client or McpClient(config)
    self._history: List[Tuple[str, str]] = []
    self._mcp_context: Optional[str] = None
    self._context_chunks: List[str] = []
    self._pending_log_items: List[McpLogItem] = []
    self._last_batch: Optional[_ResultBatch] = None
    self._turn_index = 0

  def run(self) -> None:
    console.print(
      Panel(
        "Planning mode. Type to chat.\n"
        "/models list models, /model <id> switches.\n"
        "/sources shows MCP servers, /use and /url fetch references.\n"
        "/inject adds selected references to context, /context inspects it, /quit exits.",
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
      self._flush_pending_injections()

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
    try:
      parts = shlex.split(line)
    except ValueError as exc:
      console.print(f"[red]Failed to parse command:[/red] {exc}")
      return False
    if not parts:
      return False
    cmd, args = parts[0], parts[1:]
    if cmd in ("/quit", "/q", "/exit"):
      return True
    if cmd == "/models":
      self._show_models()
      return False
    if cmd == "/model":
      self._switch_model(args)
      return False
    if cmd == "/sources":
      self._list_sources()
      return False
    if cmd == "/use":
      self._run_mcp_tool(args)
      return False
    if cmd == "/url":
      self._fetch_url_reference(args)
      return False
    if cmd == "/inject":
      self._inject_results(args)
      return False
    if cmd == "/context":
      self._show_context()
      return False
    console.print(f"[yellow]Unknown command:[/yellow] {line}")
    return False

  def _show_models(self) -> None:
    models = self._registry.list_models()
    for m in models:
      mark = "*" if m.id == self._registry.current_id else " "
      console.print(f"{mark} {m.id} ({m.model_name})")

  def _switch_model(self, args: List[str]) -> None:
    if not args:
      console.print("[yellow]Usage: /model <id>[/yellow]")
      return
    mid = args[0]
    try:
      self._registry.set_current(mid)
      console.print(f"Switched model to [bold]{mid}[/bold]")
    except KeyError:
      console.print(f"[red]Unknown model id:[/red] {mid}")

  def _list_sources(self) -> None:
    servers = self._config.mcp_servers
    if not servers:
      console.print("[yellow]No MCP servers configured.[/yellow]")
      return
    table = Table(title="MCP Servers")
    table.add_column("Server", style="cyan", no_wrap=True)
    table.add_column("Tools")
    table.add_column("Command")
    for server in servers:
      tools = self._format_tool_names(server.id) or "—"
      command_text = " ".join(server.command)
      table.add_row(server.id, tools, command_text)
    console.print(table)

  def _format_tool_names(self, server_id: str) -> str:
    list_tools = getattr(self._mcp_client, "list_tools", None)
    if not list_tools:
      return ""
    try:
      entries = list_tools(server_id)
    except Exception:  # pragma: no cover - depends on MCP implementation
      return ""
    if not entries:
      return ""
    lines: List[str] = []
    for entry in entries:
      if isinstance(entry, dict):
        name = entry.get("name")
        description = entry.get("description") or entry.get("title")
      else:
        name = getattr(entry, "name", None)
        description = getattr(entry, "description", None) or getattr(entry, "title", None)
        if name is None:
          name = str(entry)
      if not name:
        continue
      text = f"[bold]{name}[/bold]"
      if description:
        text += f" — {description}"
      lines.append(text)
    return "\n".join(lines)

  def _run_mcp_tool(self, args: List[str]) -> None:
    if len(args) < 3:
      console.print("[yellow]Usage: /use <server> <tool> \"query\" [limit][/yellow]")
      return
    server, tool, query = args[:3]
    limit: Optional[int] = None
    if len(args) >= 4:
      try:
        limit = int(args[3])
      except ValueError:
        console.print("[yellow]Limit must be an integer.[/yellow]")
        return
    params: dict[str, object] = {"query": query}
    if limit is not None:
      params["limit"] = limit
    try:
      result = self._mcp_client.call_tool(server, tool, params)
    except Exception as exc:  # noqa: BLE001
      console.print(f"[red]MCP error:[/red] {exc}")
      return
    items = normalize_payload(result.payload)
    if not items:
      console.print("[yellow]No results returned.[/yellow]")
      self._last_batch = None
      return
    self._last_batch = _ResultBatch(
      source="mcp",
      items=items,
      label=query,
      server=server,
      tool=tool,
    )
    self._show_items_table(items, title=f"{server}:{tool}")

  def _fetch_url_reference(self, args: List[str]) -> None:
    if not args:
      console.print("[yellow]Usage: /url <url> [label][/yellow]")
      return
    url = args[0]
    label = " ".join(args[1:]).strip() or None
    try:
      text = fetch_url_text(url)
    except Exception as exc:  # noqa: BLE001
      console.print(f"[red]Failed to fetch URL:[/red] {exc}")
      return
    if not text.strip():
      console.print("[yellow]URL returned no readable text.[/yellow]")
      return
    reference = UrlReference(url=url, label=label)
    items = build_url_items(reference, text)
    self._last_batch = _ResultBatch(
      source="url",
      items=items,
      label=label or url,
      server="url",
      tool="fetch",
      url=url,
    )
    preview = ""
    if items and items[0].body:
      body_sample = items[0].body.strip()
      preview = body_sample[:160] + ("…" if len(body_sample) > 160 else "")
    message = f"Fetched {label or url}"
    if preview:
      message += f": {preview}"
    console.print(f"[green]{message}[/green]")

  def _inject_results(self, args: List[str]) -> None:
    if not self._last_batch or not self._last_batch.items:
      console.print("[yellow]No reference data to inject. Run /use or /url first.[/yellow]")
      return
    if not args:
      console.print("[yellow]Usage: /inject <indices>[/yellow]")
      return
    indices = self._parse_indices(args)
    if not indices:
      console.print("[yellow]No valid indices provided.[/yellow]")
      return
    selected: List[NormalizedItem] = []
    for idx in indices:
      if 0 <= idx < len(self._last_batch.items):
        selected.append(self._last_batch.items[idx])
    if not selected:
      console.print("[yellow]No matching items for the given indices.[/yellow]")
      return
    chunk = self._format_context_chunk(self._last_batch, selected)
    if not chunk:
      console.print("[yellow]No text extracted from the selected items.[/yellow]")
      return
    self._context_chunks.append(chunk)
    self._mcp_context = "\n\n".join(self._context_chunks).strip()
    log_items = self._build_log_items(self._last_batch, selected)
    if log_items:
      self._pending_log_items.extend(log_items)
    console.print(f"[green]Injected {len(selected)} item(s) into context.[/green]")

  def _parse_indices(self, args: List[str]) -> List[int]:
    text = " ".join(args).replace(",", " ")
    indices: List[int] = []
    for token in text.split():
      if "-" in token:
        start_text, _, end_text = token.partition("-")
        try:
          start = int(start_text)
          end = int(end_text)
        except ValueError:
          continue
        rng = range(start, end + 1) if start <= end else range(start, end - 1, -1)
        for num in rng:
          indices.append(num - 1)
      else:
        try:
          indices.append(int(token) - 1)
        except ValueError:
          continue
    deduped: List[int] = []
    for idx in indices:
      if idx not in deduped:
        deduped.append(idx)
    return deduped

  def _format_context_chunk(self, batch: _ResultBatch, items: List[NormalizedItem]) -> str:
    label = batch.label or batch.url or (f"{batch.server}:{batch.tool}".strip(":"))
    header = f"### {label}" if label else None
    blocks: List[str] = []
    if header:
      blocks.append(header)
    for idx, item in enumerate(items, start=1):
      heading = item.title or f"Item {idx}"
      body = (item.body or item.snippet or "").strip()
      lines = [heading]
      if body:
        lines.append(body)
      if item.url and item.url != batch.url:
        lines.append(f"Source: {item.url}")
      blocks.append("\n".join(lines).strip())
    return "\n\n".join(block for block in blocks if block).strip()

  def _build_log_items(self, batch: _ResultBatch, items: List[NormalizedItem]) -> List[McpLogItem]:
    log_items: List[McpLogItem] = []
    for idx, item in enumerate(items, start=1):
      log_items.append(
        McpLogItem(
          idx=idx,
          server=batch.server or batch.source,
          tool=batch.tool or batch.source,
          label=batch.label,
          normalized_id=item.id,
          title=item.title,
          type=item.type,
          snippet=item.snippet,
          body=item.body,
          url=item.url or batch.url,
          metadata=dict(item.metadata),
        )
      )
    return log_items

  def _show_items_table(self, items: List[NormalizedItem], *, title: str) -> None:
    table = Table(title=f"Results — {title}")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Title")
    table.add_column("Snippet")
    table.add_column("URL")
    for idx, item in enumerate(items, start=1):
      table.add_row(
        str(idx),
        item.title or f"Item {idx}",
        self._format_snippet(item),
        item.url or "",
      )
    console.print(table)

  def _format_snippet(self, item: NormalizedItem) -> str:
    text = (item.snippet or item.body or "").strip()
    if not text:
      return ""
    return text[:96] + ("…" if len(text) > 96 else "")

  def _flush_pending_injections(self) -> None:
    if not self._pending_log_items:
      return
    for idx, item in enumerate(self._pending_log_items, start=1):
      item.idx = idx
    self._log.log_mcp_injection(self._turn_index, list(self._pending_log_items))
    self._pending_log_items.clear()

  def _show_context(self) -> None:
    if not self._context_chunks:
      console.print("[yellow]Context buffer is empty. Use /inject to add references.[/yellow]")
      return
    title = f"MCP Context ({len(self._context_chunks)} chunk(s))"
    console.print(Panel(self._mcp_context or "", title=title))
