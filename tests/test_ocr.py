"""Unit tests for OCR post-processing (no model required)."""

import re

import pytest

from src.ocr.reader import _BIB_PATTERN, _extract_bib


@pytest.mark.parametrize(
    "text, expected",
    [
        ("123", "123"),
        ("BIB 42", "BIB42"),  # cleaned → 5 chars, valid
        ("hello", None),  # too long after clean
        ("!@#123", "123"),
        ("12345", "12345"),
        ("123456", None),  # 6 chars → invalid
        ("", None),
        ("  45  ", "45"),
    ],
)
def test_extract_bib(text: str, expected: str | None) -> None:
    assert _extract_bib(text) == expected
