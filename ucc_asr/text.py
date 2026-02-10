from __future__ import annotations

import re


_SPACE_RE = re.compile(r"\\s+")


def normalize_text(s: str) -> str:
    # Keep it simple and stable: lowercase + collapse whitespace.
    s = s.strip().lower()
    s = _SPACE_RE.sub(" ", s)
    return s


def words(s: str) -> list[str]:
    s = normalize_text(s)
    if not s:
        return []
    return s.split(" ")

