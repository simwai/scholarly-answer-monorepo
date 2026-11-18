"""PDF text extraction, bounded and failure-tolerant."""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

# pypdf logs parse warnings at WARNING for every malformed or scanned PDF it
# touches. On an MCP stdio server that noise pollutes the channel, and the
# failures are handled deliberately here anyway, so quiet it down.
logging.getLogger("pypdf").setLevel(logging.CRITICAL)
logging.getLogger("pypdf._reader").setLevel(logging.CRITICAL)

# pypdf is a hard dependency. The import is guarded only so that a broken
# environment produces a legible message instead of a confusing AttributeError;
# it must never silently disable extraction, so the failure is logged loudly.
try:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError as _PdfError
except ImportError as exc:  # pragma: no cover
    logger.error("pypdf is not importable - full-text extraction is unavailable: %s", exc)
    PdfReader = None  # type: ignore[assignment]
    _PdfError = Exception  # type: ignore[assignment,misc]


def extract_pdf_text(body: bytes, max_chars: int) -> str | None:
    """Extract plain text from a PDF byte string.

    Scanned PDFs with no text layer yield nothing, which is a normal outcome -
    the caller falls back to the abstract. Never raises.
    """
    if PdfReader is None or not body:
        return None
    try:
        reader = PdfReader(io.BytesIO(body), strict=False)
    except _PdfError as exc:
        logger.debug("pdf parse failed: %s", exc)
        return None

    parts: list[str] = []
    total = 0
    try:
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except _PdfError as exc:
                # A single bad page must not abort the rest of the document.
                logger.debug("page extraction failed: %s", exc)
                continue
            if not text:
                continue
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break
    except Exception as exc:  # noqa: BLE001 - malformed PDFs can raise arbitrary errors
        logger.debug("pdf iteration failed: %s", exc)
        return None

    flat = " ".join(" ".join(parts).split())
    return flat[:max_chars] if flat else None
