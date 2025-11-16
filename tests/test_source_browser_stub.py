from __future__ import annotations

import pytest

pytest.importorskip("textual")

from simple_rag_writer.mcp.source_browser_app import (  # noqa: E402
  SourceBrowserApp,
  build_demo_items,
)
from simple_rag_writer.mcp.types import NormalizedItem  # noqa: E402


def test_source_browser_app_is_textual_app():
  from textual.app import App

  assert issubclass(SourceBrowserApp, App)


def test_source_browser_app_defaults_to_demo_items():
  app = SourceBrowserApp()

  assert app.items == build_demo_items()


def test_source_browser_app_uses_passed_items():
  items = [
    NormalizedItem(title="Doc 1", snippet="Snippet", url="https://example.com/1"),
  ]

  app = SourceBrowserApp(items=items, title="Custom")

  assert app.items == items
  assert app.app_title == "Custom"


def test_build_demo_items_returns_stable_items():
  demo = build_demo_items()

  assert len(demo) >= 1
  assert all(isinstance(item, NormalizedItem) for item in demo)
