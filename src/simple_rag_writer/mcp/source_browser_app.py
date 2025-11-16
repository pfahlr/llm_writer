from __future__ import annotations

from typing import Iterable, List, Sequence

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header

from simple_rag_writer.mcp.types import NormalizedItem


def build_demo_items() -> List[NormalizedItem]:
  """Return a predictable set of stubbed items for the browser."""
  return [
    NormalizedItem(
      id="alpha",
      type="note",
      title="Welcome Packet",
      snippet="Outline of the onboarding packet for the SRW demo.",
      url="https://example.com/welcome",
    ),
    NormalizedItem(
      id="beta",
      type="doc",
      title="Integration Notes",
      snippet="Collected integration steps for MCP and planning mode.",
      url="https://example.com/notes",
    ),
    NormalizedItem(
      id="gamma",
      type="link",
      title="Reference API",
      snippet="Summaries pulled via MCP reference API bridge.",
      url="https://example.com/reference",
    ),
  ]


def _format_snippet(item: NormalizedItem) -> str:
  text = item.snippet or item.body or ""
  return text[:96] + ("…" if len(text) > 96 else "")


class SourceBrowserApp(App[None]):
  """Minimal Textual application for browsing normalized MCP items."""

  CSS = """
  DataTable {
    height: 1fr;
  }
  """
  BINDINGS = [("q", "quit", "Quit")]

  def __init__(
    self,
    items: Sequence[NormalizedItem] | None = None,
    title: str = "MCP Source Browser",
  ) -> None:
    super().__init__()
    self.items = list(items) if items is not None else build_demo_items()
    self.app_title = title

  def compose(self) -> ComposeResult:
    yield Header(show_clock=False)
    table = DataTable(id="mcp-items")
    table.zebra_stripes = True
    table.add_columns("Title", "Snippet", "URL")
    for idx, row in enumerate(self._item_rows(self.items), start=1):
      display_title, snippet, url = row
      table.add_row(display_title or f"Item {idx}", snippet, url or "")
    yield table
    yield Footer()

  def on_mount(self) -> None:
    self.title = self.app_title

  @staticmethod
  def _item_rows(items: Iterable[NormalizedItem]) -> Iterable[tuple[str, str, str]]:
    for item in items:
      yield (
        item.title,
        _format_snippet(item),
        item.url or "",
      )


def launch_demo(items: Sequence[NormalizedItem] | None = None) -> None:
  """Run the stub browser with optional custom items."""
  SourceBrowserApp(items=items).run()


if __name__ == "__main__":
  launch_demo()
