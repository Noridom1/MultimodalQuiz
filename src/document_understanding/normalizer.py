from __future__ import annotations

import re


def normalize_concepts(concepts: list[str]) -> list[str]:
    """Merge duplicate or near-duplicate concepts."""
    seen: set[str] = set()
    normalized: list[str] = []

    for concept in concepts:
        cleaned = _normalize_text(concept)
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)

    return normalized


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" .,:;\t\n\r")
    return text.lower() if text else ""
