"""Text encoding repair helpers for report rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "â€œ", "â€", "æ", "ç", "è", "é")


def repair_mojibake_text(value: str) -> str:
    """Repair common UTF-8-as-Latin-1 mojibake while leaving normal text alone."""

    if not value or not any(marker in value for marker in _MOJIBAKE_MARKERS):
        return value
    try:
        repaired = value.encode("latin-1").decode("utf-8")
    except UnicodeError:
        try:
            repaired = value.encode("cp1252").decode("utf-8")
        except UnicodeError:
            return value
    return repaired if repaired else value


def repair_mojibake(value: Any) -> Any:
    """Recursively repair mojibake in JSON-like values."""

    if isinstance(value, str):
        return repair_mojibake_text(value)
    if isinstance(value, Mapping):
        return {key: repair_mojibake(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [repair_mojibake(item) for item in value]
    return value
