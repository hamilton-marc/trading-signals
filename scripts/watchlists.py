"""Shared watchlist parsing helpers."""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_WATCHLIST_PATH = "watchlists/watchlist.md"

_SYMBOL_RE = re.compile(r"^\^?[A-Za-z][A-Za-z0-9./_-]*$")
_ORDERED_LIST_RE = re.compile(r"^\d+\.\s+")
_UNORDERED_LIST_RE = re.compile(r"^[-*+]\s+")
_TASK_LIST_RE = re.compile(r"^\[[ xX]\]\s+")


def _normalize_watchlist_line(raw_line: str) -> str | None:
    line = raw_line.strip()
    if not line:
        return None
    if line.startswith("#"):
        return None
    if set(line) <= {"-", "*", "_"}:
        return None

    line = _ORDERED_LIST_RE.sub("", line, count=1)
    line = _UNORDERED_LIST_RE.sub("", line, count=1)
    line = _TASK_LIST_RE.sub("", line, count=1)
    line = line.strip().strip("`").strip()
    if not line:
        return None
    if not _SYMBOL_RE.fullmatch(line):
        return None
    return line.upper()


def read_watchlist(path: Path) -> list[str]:
    """Read a plain-text or Markdown watchlist and return normalized symbols."""
    if not path.exists():
        raise FileNotFoundError(f"Watchlist not found: {path}")

    symbols: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        symbol = _normalize_watchlist_line(raw_line)
        if symbol is None or symbol in seen:
            continue
        symbols.append(symbol)
        seen.add(symbol)
    return symbols
