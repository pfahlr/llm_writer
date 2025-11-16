from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rich.panel import Panel
from rich.table import Table

from simple_rag_writer.config import PromptDefinition, PromptsFile
from simple_rag_writer.config.models import (
  AppConfig,
  McpServerConfig,
  ModelConfig,
  ProviderConfig,
)
from simple_rag_writer.mcp.types import McpToolResult
from simple_rag_writer.planning import repl as repl_module
from simple_rag_writer.planning.repl import MCP_QUERY_HISTORY_LIMIT, PlanningRepl


def _make_config(
  *,
  mcp_servers: Optional[Sequence[McpServerConfig]] = None,
) -> AppConfig:
  return AppConfig(
    default_model="writer",
    providers={"local": ProviderConfig(type="openai", api_key="dummy")},
    models=[
      ModelConfig(id="writer", provider="local", model_name="Writer Model"),
      ModelConfig(id="editor", provider="local", model_name="Editor Model"),
    ],
    mcp_servers=list(mcp_servers or []),
  )


@dataclass
class _FakeModel:
  id: str
  model_name: str


class _FakeModelRegistry:
  def __init__(self) -> None:
    self.models = [
      _FakeModel(id="writer", model_name="Writer Model"),
      _FakeModel(id="editor", model_name="Editor Model"),
    ]
    self.current_id = "writer"
    self.completions: List[str] = []
    self.last_system_prompt: Optional[str] = None

  def list_models(self) -> List[_FakeModel]:
    return list(self.models)

  def set_current(self, model_id: str) -> None:
    if model_id not in {m.id for m in self.models}:
      raise KeyError(model_id)
    self.current_id = model_id

  def complete(
    self,
    prompt: str,
    model_id: Optional[str] = None,
    task_params=None,
    mcp_client=None,
    system_prompt: Optional[str] = None,
  ) -> str:  # pragma: no cover - exercised via tests
    self.completions.append(prompt)
    self.last_system_prompt = system_prompt
    return f"assistant-{len(self.completions)}"
  def pop_tool_events(self) -> List[str]:
    return []


class _FakeMcpClient:
  def __init__(
    self,
    payload: Optional[List[Dict[str, str]]] = None,
    tools: Optional[Dict[str, List[Dict[str, str]]]] = None,
  ) -> None:
    self.payload = payload or [
      {"title": "Notebook entry", "body": "Alpha body", "url": "https://example.com/spec"},
      {"title": "Second entry", "body": "Beta body"},
    ]
    self.tools = tools or {
      "notes": [
        {"name": "search", "description": "Search notes"},
        {"name": "recent", "description": "Recent updates"},
      ]
    }
    self.calls: List[Tuple[str, str, Dict[str, object]]] = []

  def call_tool(self, server: str, tool: str, params: Dict[str, object]) -> McpToolResult:
    self.calls.append((server, tool, params))
    return McpToolResult(server_id=server, tool_name=tool, payload=list(self.payload))

  def list_tools(self, server: str) -> List[Dict[str, str]]:
    return list(self.tools.get(server, []))


class _FakePlanningLogWriter:
  def __init__(self) -> None:
    self.started: List[Tuple[int, str]] = []
    self.ended: List[Tuple[int, str]] = []
    self.models_used: List[str] = []
    self.injections: List[Tuple[int, List[object]]] = []
    self.closed = False

  def start_turn(self, turn_index: int, user_text: str) -> None:
    self.started.append((turn_index, user_text))

  def log_model_used(self, model_id: str) -> None:
    self.models_used.append(model_id)

  def log_mcp_injection(self, turn_index: int, items) -> None:  # noqa: ANN001 - helper for tests
    self.injections.append((turn_index, list(items)))

  def end_turn(self, turn_index: int, assistant_text: str) -> None:
    self.ended.append((turn_index, assistant_text))

  def close(self) -> None:
    self.closed = True


class _FakeConsole:
  def __init__(self, inputs: Optional[Sequence[str]] = None) -> None:
    self._inputs = list(inputs or [])
    self.printed: List[object] = []
    self.prompts: List[str] = []

  def print(self, *args, **kwargs) -> None:  # pragma: no cover - basic forwarding
    if args:
      self.printed.append(args[0])
    else:
      self.printed.append(None)

  def input(self, prompt: str) -> str:
    self.prompts.append(prompt)
    if not self._inputs:
      raise EOFError("No more input")
    return self._inputs.pop(0)


def _make_repl(
  monkeypatch,
  *,
  console_inputs: Optional[Sequence[str]] = None,
  config: Optional[AppConfig] = None,
  mcp_payload: Optional[List[Dict[str, str]]] = None,
  prompts: Optional["PromptsFile"] = None,
) -> Tuple[PlanningRepl, _FakeModelRegistry, _FakePlanningLogWriter, _FakeConsole]:
  registry = _FakeModelRegistry()
  log_writer = _FakePlanningLogWriter()
  fake_client = _FakeMcpClient(payload=mcp_payload)
  repl = PlanningRepl(
    config or _make_config(),
    registry,
    log_writer,
    mcp_client=fake_client,
    prompts=prompts,
  )
  fake_console = _FakeConsole(console_inputs)
  monkeypatch.setattr(repl_module, "console", fake_console)
  return repl, registry, log_writer, fake_console


def test_handle_command_lists_models_with_current_marker(monkeypatch) -> None:
  repl, registry, _, fake_console = _make_repl(monkeypatch)

  assert repl._handle_command("/models") is False
  assert fake_console.printed == [
    "* writer (Writer Model)",
    "  editor (Editor Model)",
  ]


def _make_sample_prompts() -> PromptsFile:
  return PromptsFile(
    spec_version="1.0.0",
    validate_prompts=True,
    prompt_guidelines_url=None,
    prompts={
      "outline": PromptDefinition(
        id="outline",
        label="Outline Architect",
        description="Planner prompt",
        tags=("planning",),
        category="drafting",
        model_hint=None,
        system_prompt="Outline system prompt",
        template_vars=(),
      ),
      "summary": PromptDefinition(
        id="summary",
        label="Summary Prompt",
        description="Summaries",
        tags=("summary",),
        category="editing",
        model_hint=None,
        system_prompt="Summary system prompt",
        template_vars=(),
      ),
    },
  )


def test_prompt_command_sets_system_prompt(monkeypatch) -> None:
  prompts = _make_sample_prompts()
  repl, registry, _, fake_console = _make_repl(
    monkeypatch,
    console_inputs=["/prompt outline", "Need plan", "/quit"],
    prompts=prompts,
  )

  repl.run()

  assert registry.last_system_prompt == "Outline system prompt"
  assert any(
    isinstance(entry, str) and "Selected prompt" in entry for entry in fake_console.printed
  )


def test_show_prompts_lists_available_entries(monkeypatch) -> None:
  prompts = _make_sample_prompts()
  repl, _, _, fake_console = _make_repl(monkeypatch, prompts=prompts)

  repl._handle_command("/prompts")

  assert any(isinstance(entry, Table) for entry in fake_console.printed)


def test_handle_command_lists_sources(monkeypatch) -> None:
  config = _make_config(
    mcp_servers=[
      McpServerConfig(id="notes", command=["notes-cmd"]),
      McpServerConfig(id="papers", command=["papers", "--flag"]),
    ]
  )
  repl, _, _, fake_console = _make_repl(monkeypatch, config=config)

  assert repl._handle_command("/sources") is False
  table = fake_console.printed[-1]
  assert isinstance(table, Table)
  assert table.row_count == 2
  assert table.columns[0]._cells == ["notes", "papers"]
  assert "search" in table.columns[1]._cells[0]
  assert "Search notes" in table.columns[1]._cells[0]
  # ensure tools column shows placeholder when unavailable
  assert table.columns[1]._cells[1] == "—"
  assert table.columns[1].header.lower() == "tools"


def test_handle_command_switches_models_and_reports_errors(monkeypatch) -> None:
  repl, registry, _, fake_console = _make_repl(monkeypatch)

  assert repl._handle_command("/model editor") is False
  assert registry.current_id == "editor"
  assert fake_console.printed[-1] == "Switched model to [bold]editor[/bold]"

  assert repl._handle_command("/model missing") is False
  assert fake_console.printed[-1] == "[red]Unknown model id:[/red] missing"


def test_handle_command_allows_quit(monkeypatch) -> None:
  repl, _, _, _ = _make_repl(monkeypatch)
  assert repl._handle_command("/quit") is True
  assert repl._handle_command("/q") is True


def test_handle_command_use_runs_mcp_tool_and_stores_results(monkeypatch) -> None:
  repl, _, _, fake_console = _make_repl(monkeypatch)

  assert repl._handle_command('/use notes search "outline ideas" 2') is False
  batch = repl._last_batch
  assert batch is not None
  assert batch.source == "mcp"
  assert batch.server == "notes"
  assert batch.tool == "search"
  assert len(batch.items) == len(repl._mcp_client.payload)
  assert repl._mcp_client.calls == [("notes", "search", {"query": "outline ideas", "limit": 2})]
  table = fake_console.printed[-1]
  assert isinstance(table, Table)
  assert table.row_count == len(batch.items)


def test_handle_command_use_accepts_key_value_arguments(monkeypatch) -> None:
  repl, _, _, fake_console = _make_repl(monkeypatch)

  command = (
    '/use notes search query:"outline ideas" '
    'paper_id:"2511.01043" categories:"cs.AI" limit:3'
  )
  assert repl._handle_command(command) is False
  batch = repl._last_batch
  assert batch is not None
  assert batch.server == "notes"
  assert batch.tool == "search"
  assert batch.label == "outline ideas"
  assert repl._mcp_client.calls == [
    (
      "notes",
      "search",
      {
        "query": "outline ideas",
        "paper_id": "2511.01043",
        "categories": "cs.AI",
        "limit": 3,
      },
    )
  ]
  table = fake_console.printed[-1]
  assert isinstance(table, Table)
  assert table.row_count == len(batch.items)


def test_handle_command_url_fetches_and_sets_batch(monkeypatch) -> None:
  repl, _, _, fake_console = _make_repl(monkeypatch)
  monkeypatch.setattr(repl_module, "fetch_url_text", lambda url: "URL body text for " + url)

  assert repl._handle_command('/url https://example.com/spec "Spec Doc"') is False
  batch = repl._last_batch
  assert batch is not None
  assert batch.source == "url"
  assert batch.url == "https://example.com/spec"
  assert batch.label == "Spec Doc"
  assert batch.items and "URL body text" in batch.items[0].body
  assert "Spec Doc" in fake_console.printed[-1]


def test_run_loop_logs_turns_and_uses_history_window(monkeypatch) -> None:
  repl, registry, log_writer, fake_console = _make_repl(
    monkeypatch,
    console_inputs=[
      "one",
      "two",
      "three",
      "four",
      "five",
      "six",
      "seven",
      "/quit",
    ],
  )
  monkeypatch.setattr(repl_module, "HISTORY_WINDOW", 2, raising=False)

  prompt_calls: List[Tuple[List[Tuple[str, str]], str, Optional[str]]] = []

  def fake_prompt_builder(
    history: List[Tuple[str, str]],
    user_message: str,
    mcp_context: Optional[str],
    *,
    mcp_query_history: Optional[List[str]] = None,
    history_window: int = 5,
  ) -> str:
    prompt_calls.append((list(history), user_message, mcp_context))
    return f"prompt-{len(prompt_calls)}"

  monkeypatch.setattr(repl_module, "build_planning_prompt", fake_prompt_builder)

  repl.run()

  assert isinstance(fake_console.printed[0], Panel)
  assert fake_console.printed[1:] == [
    "[bold green]assistant-1[/bold green]",
    "[bold green]assistant-2[/bold green]",
    "[bold green]assistant-3[/bold green]",
    "[bold green]assistant-4[/bold green]",
    "[bold green]assistant-5[/bold green]",
    "[bold green]assistant-6[/bold green]",
    "[bold green]assistant-7[/bold green]",
  ]

  assert log_writer.started == [
    (1, "one"),
    (2, "two"),
    (3, "three"),
    (4, "four"),
    (5, "five"),
    (6, "six"),
    (7, "seven"),
  ]
  assert log_writer.ended == [
    (1, "assistant-1"),
    (2, "assistant-2"),
    (3, "assistant-3"),
    (4, "assistant-4"),
    (5, "assistant-5"),
    (6, "assistant-6"),
    (7, "assistant-7"),
  ]
  assert log_writer.models_used == ["writer"] * 7
  assert log_writer.closed is True
  assert len(prompt_calls) == 7

  last_history, last_user, last_ctx = prompt_calls[-1]
  assert last_user == "seven"
  assert last_ctx is None
  assert last_history == [
    ("five", "assistant-5"),
    ("six", "assistant-6"),
  ]


def test_run_loop_injects_context_and_logs(monkeypatch) -> None:
  repl, registry, log_writer, fake_console = _make_repl(
    monkeypatch,
    console_inputs=[
      '/use notes search "outline ideas"',
      "/inject 1",
      "Draft plan please",
      "/quit",
    ],
  )

  prompt_calls: List[Tuple[List[Tuple[str, str]], str, Optional[str]]] = []

  def fake_prompt_builder(
    history: List[Tuple[str, str]],
    user_message: str,
    mcp_context: Optional[str],
    *,
    mcp_query_history: Optional[List[str]] = None,
    history_window: int = 5,
  ) -> str:
    prompt_calls.append((list(history), user_message, mcp_context))
    return "prompt-with-context"

  monkeypatch.setattr(repl_module, "build_planning_prompt", fake_prompt_builder)

  repl.run()

  assert log_writer.started == [(1, "Draft plan please")]
  assert log_writer.injections and log_writer.injections[0][0] == 1
  assert "Alpha body" in log_writer.injections[0][1][0].body
  assert prompt_calls[-1][2] is not None
  assert "Alpha body" in prompt_calls[-1][2]


def test_injected_context_chunk_uses_server_tool_label(monkeypatch) -> None:
  repl, _, _, _ = _make_repl(monkeypatch)

  repl._run_mcp_tool(['notes', 'search', '"outline ideas"', '2'])
  repl._inject_results(["1"])

  chunk = repl._context_chunks[-1]
  assert chunk.startswith("### notes:search")
  assert "outline ideas" not in chunk


def test_mcp_query_history_passed_to_prompt(monkeypatch) -> None:
  repl, _, _, _ = _make_repl(monkeypatch, console_inputs=["Plan it", "/quit"])
  repl._run_mcp_tool(['notes', 'search', '"outline ideas"', '2'])

  captured: List[Optional[List[str]]] = []

  def fake_prompt_builder(
    history: List[Tuple[str, str]],
    user_message: str,
    mcp_context: Optional[str],
    *,
    mcp_query_history: Optional[List[str]] = None,
    history_window: int = 5,
  ) -> str:
    captured.append(mcp_query_history)
    return "prompted"

  monkeypatch.setattr(repl_module, "build_planning_prompt", fake_prompt_builder)

  repl.run()

  assert captured
  assert captured[-1] == repl._mcp_query_history[-MCP_QUERY_HISTORY_LIMIT :]


def test_run_loop_retries_after_llm_error(monkeypatch) -> None:
  repl, registry, log_writer, fake_console = _make_repl(
    monkeypatch,
    console_inputs=[
      "Need outline",
      "/quit",
    ],
  )

  base_prompt = "Prompt base"
  monkeypatch.setattr(
    repl_module,
    "build_planning_prompt",
    lambda history, user, ctx, *, mcp_query_history=None, history_window=5: base_prompt,
  )

  original_complete = registry.complete
  prompts: List[str] = []

  def flaky_complete(prompt, model_id=None, task_params=None, mcp_client=None, system_prompt=None):
    prompts.append(prompt)
    if len(prompts) == 1:
      raise RuntimeError("bad format around MCP tool block")
    return original_complete(
      prompt,
      model_id=model_id,
      task_params=task_params,
      mcp_client=mcp_client,
      system_prompt=system_prompt,
    )

  registry.complete = flaky_complete  # type: ignore[assignment]

  repl.run()

  assert log_writer.started == [(1, "Need outline")]
  assert log_writer.ended == [(1, "assistant-1")]
  assert log_writer.models_used == ["writer"]
  assert repl._history == [("Need outline", "assistant-1")]
  assert len(prompts) == 2
  assert prompts[0] == base_prompt
  assert "SYSTEM FEEDBACK" in prompts[1]
  assert "bad format" in prompts[1]
  assert any("attempt 1" in str(msg) for msg in fake_console.printed if isinstance(msg, str))


def test_run_loop_recovers_from_llm_errors(monkeypatch) -> None:
  repl, registry, log_writer, fake_console = _make_repl(
    monkeypatch,
    console_inputs=[
      "Need outline",
      "/quit",
    ],
  )

  def failing_complete(*_args, **_kwargs):
    raise RuntimeError("model exploded")

  registry.complete = failing_complete  # type: ignore[assignment]
  monkeypatch.setattr(
    repl_module,
    "build_planning_prompt",
    lambda history, user, ctx, *, mcp_query_history=None, history_window=5: "prompt-error",
  )

  repl.run()

  assert log_writer.started == [(1, "Need outline")]
  assert log_writer.ended == [(1, "LLM call failed: model exploded")]
  assert log_writer.models_used == []
  assert repl._history == []
  assert any(
    isinstance(msg, str) and "LLM call failed" in msg for msg in fake_console.printed
  )


def test_context_command_prints_current_buffer(monkeypatch) -> None:
  repl, _, _, fake_console = _make_repl(monkeypatch)

  assert repl._handle_command('/use notes search "outline ideas"') is False
  assert repl._handle_command("/inject 1") is False
  assert repl._handle_command("/context") is False

  panel = fake_console.printed[-1]
  assert isinstance(panel, Panel)
  assert "Alpha body" in panel.renderable
