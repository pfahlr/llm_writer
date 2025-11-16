from __future__ import annotations

from simple_rag_writer.runner import url_fetcher


def test_fetch_url_text_uses_pdf_conversion(monkeypatch):
  calls = {}

  class DummyResponse:
    def __init__(self):
      self.headers = {"Content-Type": "application/pdf"}

    def read(self) -> bytes:
      calls["read"] = True
      return b"%PDF-sample"

    def __enter__(self):
      return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
      return False

  def fake_urlopen(request, timeout):  # noqa: ANN001
    calls["url"] = request.full_url
    calls["timeout"] = timeout
    return DummyResponse()

  monkeypatch.setattr(url_fetcher, "urlopen", fake_urlopen)
  monkeypatch.setattr(url_fetcher, "_pdf_bytes_to_text", lambda data: "PDF TEXT")

  text = url_fetcher.fetch_url_text("https://example.com/doc.pdf")

  assert text == "PDF TEXT"
  assert calls["url"].endswith("doc.pdf")
  assert calls["read"] is True


def test_pdf_bytes_to_text_extracts_from_reader(monkeypatch):
  captured = {}

  class FakePage:
    def __init__(self, text: str) -> None:
      self._text = text

    def extract_text(self) -> str:
      return self._text

  class FakeReader:
    def __init__(self, stream) -> None:
      captured["data"] = stream.read()
      self.pages = [FakePage("First"), FakePage("Second")]

  monkeypatch.setattr(url_fetcher, "PdfReader", lambda stream: FakeReader(stream))

  text = url_fetcher._pdf_bytes_to_text(b"data-pdf")

  assert text == "First\n\nSecond"
  assert captured["data"] == b"data-pdf"
