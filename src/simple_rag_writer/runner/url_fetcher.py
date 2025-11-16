from __future__ import annotations

from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote, urlparse
from urllib.request import Request, url2pathname, urlopen

try:  # pragma: no cover - dependency is optional at import time
  from pypdf import PdfReader
except ImportError:  # pragma: no cover
  PdfReader = None

from simple_rag_writer.mcp.types import NormalizedItem
from simple_rag_writer.tasks.models import UrlReference

DEFAULT_USER_AGENT = "simple-rag-writer/0.1"


class _HTMLToTextParser(HTMLParser):
  def __init__(self) -> None:
    super().__init__()
    self._parts: list[str] = []
    self._skip_depth = 0

  def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
    lowered = tag.lower()
    if lowered in {"br", "p", "div", "section", "article", "li", "tr"}:
      self._parts.append("\n")
    if lowered in {"script", "style"}:
      self._skip_depth += 1

  def handle_endtag(self, tag: str):  # type: ignore[override]
    lowered = tag.lower()
    if lowered in {"script", "style"} and self._skip_depth:
      self._skip_depth -= 1
    elif lowered in {"p", "div", "section", "article", "li"}:
      self._parts.append("\n")

  def handle_data(self, data: str):  # type: ignore[override]
    if self._skip_depth:
      return
    if data.strip():
      self._parts.append(data)

  def get_text(self) -> str:
    raw = "".join(self._parts)
    lines = [line.strip() for line in raw.splitlines()]
    filtered = "\n".join(line for line in lines if line)
    return filtered.strip()


def _extract_file_path(url: str) -> Path:
  parsed = urlparse(url)
  path = unquote(parsed.path or "")
  if parsed.netloc:
    path = f"//{parsed.netloc}{path}"
  fs_path = Path(url2pathname(path))
  return fs_path


def _html_to_text(html: str) -> str:
  parser = _HTMLToTextParser()
  parser.feed(html)
  parser.close()
  text = parser.get_text()
  return text or html


def _pdf_bytes_to_text(data: bytes) -> str:
  if PdfReader is None:  # pragma: no cover - exercised when dependency missing
    raise RuntimeError("pypdf is required to extract PDF text")
  reader = PdfReader(BytesIO(data))
  chunks: List[str] = []
  for page in reader.pages:
    try:
      page_text = page.extract_text() or ""
    except Exception:  # pragma: no cover - defensive against parser quirks
      page_text = ""
    if page_text:
      chunks.append(page_text.strip())
  return "\n\n".join(chunk for chunk in chunks if chunk).strip()


def fetch_url_text(url: str, *, timeout: float = 15.0) -> str:
  parsed = urlparse(url)
  scheme = (parsed.scheme or "").lower()

  if scheme in {"", "file"}:
    path = _extract_file_path(url)
    if path.suffix.lower() == ".pdf":
      return _pdf_bytes_to_text(path.read_bytes())
    return path.read_text(encoding="utf-8")

  if scheme not in {"http", "https"}:
    raise ValueError(f"Unsupported URL scheme: {parsed.scheme or 'unknown'}")

  request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
  with urlopen(request, timeout=timeout) as response:  # nosec: B310 in controlled use
    content_type = response.headers.get("Content-Type", "")
    charset: Optional[str] = None
    try:
      charset = response.headers.get_content_charset()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - very old Python versions only
      charset = None
    data = response.read()
  lower_content_type = content_type.lower()
  if "pdf" in lower_content_type or parsed.path.lower().endswith(".pdf"):
    return _pdf_bytes_to_text(data)
  encoding = charset or "utf-8"
  text = data.decode(encoding, errors="replace")
  if "html" in lower_content_type:
    return _html_to_text(text)
  return text


def build_url_items(reference: UrlReference, text: str) -> List[NormalizedItem]:
  body = text.strip()
  return [
    NormalizedItem(
      id=reference.url,
      type=reference.item_type or "url",
      title=reference.label or reference.url,
      body=body,
      url=reference.url,
    )
  ]


__all__ = ["fetch_url_text", "build_url_items", "_pdf_bytes_to_text"]
