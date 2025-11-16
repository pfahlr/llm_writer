from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from rich.panel import Panel

from simple_rag_writer.config.models import AppConfig, ModelConfig, ProviderConfig
from simple_rag_writer.planning import repl as repl_module
from simple_rag_writer.planning.repl import PlanningRepl


def _make_config() -> AppConfig:
  return AppConfig(
    default_model="writer",
    providers={"local": ProviderConfig(type="openai", api_key="dummy")},
    models=[
      ModelConfig(id="writer", provider="local", model_name="Writer Model"),
      ModelConfig(id="editor", provider="local", model_name="Editor Model"),
    ],
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

  def list_models(self) -> List[_FakeModel]:
    return list(self.models)

  def set_current(self, model_id: str) -> None:
    if model_id not in {m.id for m in self.models}:
      raise KeyError(model_id)
    self.current_id = model_id

  def complete(self, prompt: str, model_id: Optional[str] = None, task_params=None) -> str:  # pragma: no cover - exercised via tests
    self.completions.append(prompt)
    return f"assistant-{len(self.completions)}"


class _FakePlanningLogWriter:
  def __init__(self) -> None:
    self.started: List[Tuple[int, str]] = []
    self.ended: List[Tuple[int, str]] = []
    self.models_used: List[str] = []
    self.closed = False

  def start_turn(self, turn_index: int, user_text: str) -> None:
    self.started.append((turn_index, user_text))

  def log_model_used(self, model_id: str) -> None:
    self.models_used.append(model_id)

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


def _make_repl(monkeypatch, *, console_inputs: Optional[Sequence[str]] = None) -> Tuple[PlanningRepl, _FakeModelRegistry, _FakePlanningLogWriter, _FakeConsole]:
  registry = _FakeModelRegistry()
  log_writer = _FakePlanningLogWriter()
  repl = PlanningRepl(_make_config(), registry, log_writer)
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
