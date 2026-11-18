"""Tests for PDF text extraction, using a PDF generated in-process."""

from __future__ import annotations

import io

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from scholarly_answer_engine.fulltext.pdftext import extract_pdf_text


def make_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 18 Tf 20 150 Td ({text}) Tj ET".encode())
    content_ref = writer._add_object(stream)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    page[NameObject("/Contents")] = content_ref
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_extracts_text_from_valid_pdf():
    raw = make_pdf("Hello scholarly world")
    assert extract_pdf_text(raw, 500) == "Hello scholarly world"


def test_truncates_to_max_chars():
    raw = make_pdf("A" * 5000)
    text = extract_pdf_text(raw, 100)
    assert text is not None
    assert len(text) <= 100


def test_garbage_returns_none_not_an_exception():
    assert extract_pdf_text(b"this is not a pdf", 100) is None


def test_empty_returns_none():
    assert extract_pdf_text(b"", 100) is None


def test_truncated_pdf_returns_none_not_an_exception():
    raw = make_pdf("Hello scholarly world")
    assert extract_pdf_text(raw[: len(raw) // 2], 100) is None


def test_header_only_pdf_is_handled():
    assert extract_pdf_text(b"%PDF-1.4\n%%EOF", 100) is None
