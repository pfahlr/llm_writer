from __future__ import annotations

import shlex
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from simple_rag_writer.config import PromptsFile, PromptDefinition
from simple_rag_writer.config.models import AppConfig
from simple_rag_writer.llm.executor import LlmCompletionError, run_completion_with_feedback
from simple_rag_writer.llm.registry import ModelRegistry
from simple_rag_writer.logging.planning_log import McpLogItem, PlanningLogWriter
from simple_rag_writer.mcp.client import McpClient
from simple_rag_writer.mcp.normalization import normalize_payload
from simple_rag_writer.mcp.types import NormalizedItem
from simple_rag_writer.planning.memory import ManualMemoryEntry, ManualMemoryStore
from simple_rag_writer.prompts.planning import (
  DEFAULT_HISTORY_WINDOW,
  build_planning_prompt,
)
from simple_rag_writer.runner.url_fetcher import build_url_items, fetch_url_text
from simple_rag_writer.tasks.models import UrlReference

console = Console()
HISTORY_WINDOW = DEFAULT_HISTORY_WINDOW
MAX_LLM_COMPLETION_ATTEMPTS = 2
MCP_QUERY_HISTORY_LIMIT = 5
MAX_MEMORY_SNAPSHOT_CHARS = 4000

# Display formatting constants
MEMORY_SNIPPET_PREVIEW_LENGTH = 80
TABLE_TEXT_PREVIEW_LENGTH = 96


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
    prompts: Optional[PromptsFile] = None,
    memory_store: Optional[ManualMemoryStore] = None,
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
    self._mcp_query_history: List[str] = []
    self._turn_index = 0
    prompt_library = prompts.prompts if prompts else {}
    self._prompts: Dict[str, PromptDefinition] = dict(prompt_library)
    self._selected_prompt_id: Optional[str] = None
    self._selected_system_prompt: Optional[str] = None
    self._memory_store = memory_store or ManualMemoryStore()

  def run(self) -> None:
    console.print(
      Panel(
        "Planning mode. Type to chat.\n"
        "/models list models, /model <id> switches.\n"
        "/sources shows MCP servers, /mcp-status for diagnostics.\n"
        "/use and /url fetch references, /paste [label] for manual context.\n"
        "/prompts lists system prompts, /prompt <id|default> selects one.\n"
        "/remember label:: text saves manual memory. /memory list|inject manage it.\n"
        "/inject adds selected references to context, /context inspects it.\n"
        "/stream [on|off] toggles streaming output, /quit exits.",
        title="Simple Rag Writer",
      )
    )

    # Check health of required MCP servers before starting
    self._check_required_servers_health()

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
      prompt = build_planning_prompt(
        history_slice,
        line,
        self._mcp_context,
        mcp_query_history=self._mcp_query_history[-MCP_QUERY_HISTORY_LIMIT :]
        if self._mcp_query_history
        else None,
      )
      try:
        # Check if streaming is enabled
        streaming_config = self._get_streaming_config()
        if streaming_config and streaming_config.enabled:
          output = self._run_with_streaming(prompt)
        else:
          output = run_completion_with_feedback(
            self._registry,
            prompt,
            mcp_client=self._mcp_client,
            system_prompt=self._selected_system_prompt,
            max_attempts=MAX_LLM_COMPLETION_ATTEMPTS,
            on_attempt_failure=partial(
              self._report_retry_attempt, MAX_LLM_COMPLETION_ATTEMPTS
            ),
          )
      except LlmCompletionError as exc:
        message = exc.message
        context_info = [
          f"[red bold]LLM Call Failed[/red bold]",
          f"Model: {self._registry.current_id}",
          f"Turn: {self._turn_index}",
          f"Error: {message}",
        ]
        if self._mcp_context:
          context_info.append(f"Context active: {len(self._context_chunks)} chunk(s)")
        if self._mcp_query_history:
          context_info.append(f"Recent MCP queries: {len(self._mcp_query_history)}")
        console.print(Panel("\n".join(context_info), border_style="red"))
        error_text = f"LLM call failed: {message}"
        self._log.end_turn(self._turn_index, error_text)
        continue
      tool_events = self._registry.pop_tool_events()
      if tool_events:
        console.print(Panel("\n\n".join(tool_events), title="LLM tool trace"))
      self._log.log_model_used(self._registry.current_id)
      self._log.end_turn(self._turn_index, output)
      console.print(f"[bold green]{output}[/bold green]")
      self._history.append((line, output))
      self._record_turn_snapshot(line, output)

    self._log.close()

  def _report_retry_attempt(
    self, max_attempts: int, attempt: int, message: str, will_retry: bool
  ) -> None:
    if not will_retry:
      return
    text = f"LLM call failed (attempt {attempt}/{max_attempts}): {message}"
    console.print(f"[red]{text}[/red]")

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
    if cmd == "/prompts":
      self._show_prompts()
      return False
    if cmd == "/prompt":
      self._configure_prompt(line[len(cmd) :].strip())
      return False
    if cmd == "/remember":
      self._remember_text(line[len(cmd) :].strip())
      return False
    if cmd == "/memory":
      self._handle_memory_command(args)
      return False
    if cmd == "/sources":
      self._list_sources()
      return False
    if cmd == "/mcp-status":
      self._show_mcp_diagnostics()
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
    if cmd == "/stream":
      self._toggle_streaming(args)
      return False
    if cmd == "/paste":
      self._paste_manual_context(args)
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

  def _show_prompts(self) -> None:
    if not self._prompts:
      console.print("[yellow]No prompts configured. Add prompts.yaml to enable this feature.[/yellow]")
      return
    table = Table(title="System Prompts")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Label")
    table.add_column("Tags")
    table.add_column("Category")
    for pid, prompt in sorted(self._prompts.items()):
      tags = ", ".join(prompt.tags) if prompt.tags else "—"
      category = prompt.category or "—"
      table.add_row(pid, prompt.label or pid, tags or "—", category)
    console.print(table)
    status = (
      f"Current: [bold]{self._selected_prompt_id}[/bold]"
      if self._selected_prompt_id
      else "Current: model default"
    )
    console.print(status)

  def _configure_prompt(self, raw: str) -> None:
    if not self._prompts:
      console.print("[yellow]No prompts available.[/yellow]")
      return
    choice = raw.strip()
    if not choice:
      if self._selected_prompt_id:
        console.print(f"Using prompt [bold]{self._selected_prompt_id}[/bold].")
      else:
        console.print("Using model default system prompt.")
      return
    if choice.lower() in {"default", "clear", "none"}:
      self._selected_prompt_id = None
      self._selected_system_prompt = None
      console.print("[green]Cleared custom prompt. Using model default.[/green]")
      return
    prompt = self._prompts.get(choice)
    if not prompt:
      console.print(f"[red]Unknown prompt id:[/red] {choice}")
      return
    if not prompt.system_prompt:
      console.print(f"[yellow]Prompt '{choice}' has no system prompt text.[/yellow]")
      return
    self._selected_prompt_id = choice
    self._selected_system_prompt = prompt.system_prompt.strip()
    console.print(f"[green]Selected prompt[/green] [bold]{choice}[/bold]: {prompt.label}")

  def _remember_text(self, raw: str) -> None:
    text = (raw or "").strip()
    if not text:
      console.print("[yellow]Usage: /remember label:: important note[/yellow]")
      return
    label: Optional[str] = None
    body = text
    if "::" in text:
      label_text, body_text = text.split("::", 1)
      if body_text.strip():
        label = label_text.strip() or None
        body = body_text
    try:
      entry = self._memory_store.add(body.strip(), label=label)
    except ValueError as exc:
      console.print(f"[yellow]{exc}[/yellow]")
      return
    label_desc = f" ({entry.label})" if entry.label else ""
    console.print(f"[green]Saved memory {entry.entry_id}{label_desc}.[/green]")

  def _handle_memory_command(self, args: List[str]) -> None:
    if not args:
      console.print(
        "[yellow]Usage: /memory list | show <id> | inject <id> | delete <id> | clear[/yellow]"
      )
      return
    action = args[0].lower()
    if action == "list":
      self._list_memory_entries()
      return
    if action == "show":
      if len(args) < 2:
        console.print("[yellow]Usage: /memory show <id>[/yellow]")
        return
      self._show_memory_entry(args[1])
      return
    if action == "inject":
      if len(args) < 2:
        console.print("[yellow]Usage: /memory inject <id>[/yellow]")
        return
      self._inject_memory_entry(args[1])
      return
    if action == "delete":
      if len(args) < 2:
        console.print("[yellow]Usage: /memory delete <id>[/yellow]")
        return
      removed = self._memory_store.delete(args[1])
      if removed:
        console.print(f"[green]Deleted memory entry {args[1]}.[/green]")
      else:
        console.print(f"[yellow]No memory entry found for id {args[1]}.[/yellow]")
      return
    if action == "clear":
      self._memory_store.clear()
      console.print("[green]Cleared all manual memory entries.[/green]")
      return
    console.print(f"[yellow]Unknown /memory action:[/yellow] {action}")

  def _list_memory_entries(self) -> None:
    entries = self._memory_store.list_entries()
    if not entries:
      console.print("[yellow]No saved memory entries.[/yellow]")
      return
    table = Table(title="Manual Memory")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Label")
    table.add_column("Snippet")
    for entry in entries:
      snippet = entry.text.replace("\n", " ").strip()
      snippet = snippet[:MEMORY_SNIPPET_PREVIEW_LENGTH] + ("…" if len(snippet) > MEMORY_SNIPPET_PREVIEW_LENGTH else "")
      table.add_row(entry.entry_id, entry.label or "—", snippet or "—")
    console.print(table)

  def _show_memory_entry(self, entry_id: str) -> None:
    entry = self._memory_store.get(entry_id)
    if not entry:
      console.print(f"[yellow]No memory entry found with id {entry_id}.[/yellow]")
      return
    title = entry.label or entry_id
    console.print(Panel(entry.text, title=f"Memory {entry_id}: {title}"))

  def _inject_memory_entry(self, entry_id: str) -> None:
    entry = self._memory_store.get(entry_id)
    if not entry:
      console.print(f"[yellow]No memory entry found with id {entry_id}.[/yellow]")
      return
    chunk = self._build_memory_chunk(entry)
    if not chunk:
      console.print("[yellow]Memory entry has no text to inject.[/yellow]")
      return
    self._context_chunks.append(chunk)
    self._mcp_context = "\n\n".join(self._context_chunks).strip()
    console.print(f"[green]Injected memory entry {entry.entry_id} into context.[/green]")

  def _build_memory_chunk(self, entry: ManualMemoryEntry) -> str:
    label = entry.label or f"memory-{entry.entry_id}"
    content = entry.text.strip()
    if not content:
      return ""
    return f"### {label}\n{content}".strip()

  def _save_memory_chunk(self, chunk: str, label: Optional[str]) -> None:
    text = (chunk or "").strip()
    if not text:
      return
    prefix = label or "reference"
    try:
      self._memory_store.add(text, label=f"reference:{prefix}")
    except ValueError:
      pass

  def _record_turn_snapshot(self, user_text: str, assistant_text: str) -> None:
    text = (
      f"User turn {self._turn_index}:\n{user_text.strip()}\n\nAssistant:\n{assistant_text.strip()}"
    ).strip()
    if not text:
      return
    if len(text) > MAX_MEMORY_SNAPSHOT_CHARS:
      text = text[: MAX_MEMORY_SNAPSHOT_CHARS - 1] + "…"
    label = f"turn-{self._turn_index}"
    try:
      self._memory_store.add(text, label=label)
    except ValueError:
      pass

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

  def _show_mcp_diagnostics(self) -> None:
    """Show detailed MCP connection diagnostics."""
    if not self._config.mcp_servers:
      console.print("[yellow]No MCP servers configured.[/yellow]")
      return

    console.print("[bold]MCP Connection Diagnostics[/bold]\n")

    for server in self._config.mcp_servers:
      status_parts = [f"[cyan]Server:[/cyan] {server.id}"]

      # Connection status - try to list tools to check if server is responsive
      try:
        tools = self._mcp_client.list_tools(server.id)
        status_parts.append(f"  Status: [green]✓ Available[/green]")
        status_parts.append(f"  Tools: {len(tools)} available")
      except Exception as exc:  # noqa: BLE001
        status_parts.append(f"  Status: [red]✗ Unavailable[/red]")
        error_msg = str(exc)
        if len(error_msg) > 80:
          error_msg = error_msg[:77] + "..."
        status_parts.append(f"  Error: {error_msg}")

      # Configuration
      status_parts.append(f"  Timeout: {server.timeout}s")
      status_parts.append(f"  Retry attempts: {server.retry_attempts}")
      status_parts.append(f"  Retry delay: {server.retry_delay_seconds}s")
      status_parts.append(f"  Command: {' '.join(server.command)}")

      console.print("\n".join(status_parts))
      console.print()  # Blank line between servers

  def _run_mcp_tool(self, args: List[str]) -> None:
    if len(args) < 3:
      console.print(
        "[yellow]Usage: /use <server> <tool> \"query\" [limit] or key:value args[/yellow]"
      )
      return
    server, tool, *param_tokens = args
    if not param_tokens:
      console.print(
        "[yellow]Usage: /use <server> <tool> \"query\" [limit] or key:value args[/yellow]"
      )
      return
    try:
      params, label = self._build_tool_params(param_tokens)
    except ValueError as exc:
      console.print(f"[yellow]{exc}[/yellow]")
      return
    if not params:
      console.print(
        "[yellow]Usage: /use <server> <tool> \"query\" [limit] or key:value args[/yellow]"
      )
      return
    label = label or " ".join(param_tokens)
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
    self._record_mcp_query(server, tool, params)
    self._last_batch = _ResultBatch(
      source="mcp",
      items=items,
      label=label,
      server=server,
      tool=tool,
    )
    self._show_items_table(items, title=f"{server}:{tool}")

  def _build_tool_params(
    self, tokens: List[str]
  ) -> Tuple[Optional[dict[str, object]], Optional[str]]:
    if not tokens:
      return None, None
    named: dict[str, object] = {}
    positional: List[str] = []
    for token in tokens:
      key, value = self._split_param_token(token)
      if key is None:
        positional.append(token)
      else:
        named[key] = self._coerce_param_value(key, value)
    if named:
      return named, self._derive_param_label(named)
    if not positional:
      return None, None
    query = positional[0]
    params: dict[str, object] = {"query": query}
    if len(positional) >= 2:
      params["limit"] = self._parse_limit(positional[1])
    return params, query

  def _split_param_token(self, token: str) -> Tuple[Optional[str], str]:
    for delimiter in (":", "="):
      if delimiter in token:
        key, value = token.split(delimiter, 1)
        key = key.strip()
        if not key:
          break
        return key, value.strip()
    return None, token

  def _coerce_param_value(self, key: str, value: str) -> object:
    if key == "limit":
      return self._parse_limit(value)
    return value

  def _parse_limit(self, text: str) -> int:
    try:
      return int(text)
    except ValueError as exc:
      raise ValueError("Limit must be an integer.") from exc

  def _derive_param_label(self, params: dict[str, object]) -> Optional[str]:
    query_value = params.get("query")
    if isinstance(query_value, str) and query_value.strip():
      return query_value
    for value in params.values():
      if isinstance(value, str) and value.strip():
        return value
    return None

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
    chunk_label = self._context_chunk_label(self._last_batch)
    chunk = self._format_context_chunk(
      self._last_batch, selected, label_override=chunk_label
    )
    if not chunk:
      console.print("[yellow]No text extracted from the selected items.[/yellow]")
      return
    self._context_chunks.append(chunk)
    self._mcp_context = "\n\n".join(self._context_chunks).strip()
    log_items = self._build_log_items(self._last_batch, selected)
    if log_items:
      self._pending_log_items.extend(log_items)
    console.print(f"[green]Injected {len(selected)} item(s) into context.[/green]")
    self._save_memory_chunk(chunk, chunk_label)

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

  def _format_context_chunk(
    self,
    batch: _ResultBatch,
    items: List[NormalizedItem],
    *,
    label_override: Optional[str] = None,
  ) -> str:
    label = (
      label_override
      or batch.label
      or batch.url
      or (f"{batch.server}:{batch.tool}".strip(":"))
    )
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

  def _context_chunk_label(self, batch: Optional[_ResultBatch]) -> Optional[str]:
    if not batch:
      return None
    if batch.server and batch.tool:
      return f"{batch.server}:{batch.tool}"
    return batch.label

  def _record_mcp_query(self, server: str, tool: str, params: Dict[str, object]) -> None:
    summary = self._format_query_summary(server, tool, params)
    self._mcp_query_history.append(summary)
    if len(self._mcp_query_history) > MCP_QUERY_HISTORY_LIMIT:
      self._mcp_query_history.pop(0)

  @staticmethod
  def _format_query_summary(server: str, tool: str, params: Dict[str, object]) -> str:
    entries = []
    for key in sorted(params.keys()):
      value = params[key]
      entries.append(f"{key}={value}")
    param_text = f" ({', '.join(entries)})" if entries else ""
    return f"{server}/{tool}{param_text}"

  def _format_snippet(self, item: NormalizedItem) -> str:
    text = (item.snippet or item.body or "").strip()
    if not text:
      return ""
    return text[:TABLE_TEXT_PREVIEW_LENGTH] + ("…" if len(text) > TABLE_TEXT_PREVIEW_LENGTH else "")

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

  def _get_streaming_config(self) -> Optional[Any]:  # Optional[StreamingConfig]
    """Get effective streaming configuration."""
    from simple_rag_writer.config.models import StreamingConfig
    model = self._registry.current_model
    return (
      model.streaming_override
      or self._config.streaming_defaults
      or None
    )

  def _toggle_streaming(self, args: List[str]) -> None:
    """Toggle streaming mode on or off."""
    from simple_rag_writer.config.models import StreamingConfig

    config = self._get_streaming_config()
    if not config:
      # Create default config if none exists
      if not self._config.streaming_defaults:
        self._config.streaming_defaults = StreamingConfig()
      config = self._config.streaming_defaults

    if args and args[0].lower() in ("on", "off"):
      enable = args[0].lower() == "on"
      config.enabled = enable
      status = "enabled" if enable else "disabled"
      console.print(f"[green]Streaming {status}.[/green]")
    else:
      # Toggle
      config.enabled = not config.enabled
      status = "enabled" if config.enabled else "disabled"
      console.print(f"[green]Streaming {status}.[/green]")

  def _paste_manual_context(self, args: List[str]) -> None:
    """
    Allow user to paste multi-line context manually.

    Useful when MCP servers are unavailable or for ad-hoc context injection.
    """
    label = " ".join(args).strip() or "Manual context"

    console.print(
      f"[yellow]Paste or type context for '{label}'.[/yellow]\n"
      "[dim]End with a line containing only '###' or press Ctrl+D[/dim]\n"
    )

    lines = []
    while True:
      try:
        line = input()
        if line.strip() == "###":
          break
        lines.append(line)
      except (EOFError, KeyboardInterrupt):
        break

    if not lines:
      console.print("[yellow]No context provided.[/yellow]")
      return

    context_text = "\n".join(lines).strip()
    chunk = f"### {label}\n{context_text}"

    self._context_chunks.append(chunk)
    self._mcp_context = "\n\n".join(self._context_chunks).strip()

    console.print(
      f"[green]✓ Added {len(lines)} lines of manual context as '{label}'.[/green]"
    )

  def _check_required_servers_health(self) -> None:
    """Check health of required MCP servers before starting session."""
    from simple_rag_writer.mcp.health import check_required_servers

    if not self._config.mcp_servers:
      # No MCP servers configured, nothing to check
      return

    all_ok, statuses = check_required_servers(self._config, self._mcp_client)

    if not all_ok:
      console.print("[yellow]⚠ Warning: Some required MCP servers are unavailable:[/yellow]\n")

      table = Table()
      table.add_column("Server", style="cyan")
      table.add_column("Criticality")
      table.add_column("Status")
      table.add_column("Error")

      for status in statuses:
        server_cfg = next(
          (s for s in self._config.mcp_servers if s.id == status.server_id), None
        )
        if not server_cfg:
          continue

        criticality_style = {
          "required": "[red]Required[/red]",
          "optional": "[yellow]Optional[/yellow]",
          "best_effort": "[dim]Best effort[/dim]",
        }.get(server_cfg.criticality, server_cfg.criticality)

        if server_cfg.criticality == "required" and not status.available:
          table.add_row(
            status.server_id,
            criticality_style,
            "[red]✗ Unavailable[/red]",
            status.error or "Unknown error",
          )

      if table.row_count > 0:
        console.print(table)
        console.print("\n[yellow]Continue anyway? (y/N):[/yellow] ", end="")
        try:
          response = input().strip().lower()
          if response not in ("y", "yes"):
            console.print("[red]Aborting due to unavailable required servers.[/red]")
            import sys
            sys.exit(1)
        except (EOFError, KeyboardInterrupt):
          console.print("\n[red]Aborting due to unavailable required servers.[/red]")
          import sys
          sys.exit(1)
        console.print("[green]Continuing with degraded MCP functionality...[/green]\n")

  def _run_with_streaming(self, prompt: str) -> str:
    """
    Run LLM completion with streaming output.

    Uses hybrid approach: non-streaming for tool iterations,
    streaming for final response.

    Returns:
      Complete response text
    """
    accumulated = []
    def on_chunk(text: str) -> None:
      """Handle each streamed chunk."""
      accumulated.append(text)
      console.print(text, end="", style="bold green")

    try:
      output = self._registry.complete_streaming(
        prompt,
        mcp_client=self._mcp_client,
        system_prompt=self._selected_system_prompt,
        on_chunk=on_chunk,
      )
      console.print()  # Newline after streaming completes
      return output
    except KeyboardInterrupt:
      # User interrupted; return partial
      console.print("\n[yellow][Interrupted][/yellow]")
      partial = "".join(accumulated)
      if partial:
        return partial
      raise
