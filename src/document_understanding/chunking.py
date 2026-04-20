from __future__ import annotations

import re


def semantic_chunk(sentences: list[str], *, max_chunk_size: int = 5) -> list[list[str]]:
    """Group sentences into semantically coherent chunks."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero")

    cleaned = [sentence.strip() for sentence in sentences if sentence and sentence.strip()]
    if not cleaned:
        return []

    chunks: list[list[str]] = []
    current_chunk: list[str] = []
    current_token_count = 0

    for sentence in cleaned:
        sentence_token_count = len(re.findall(r"\w+", sentence))
        if current_chunk and (len(current_chunk) >= max_chunk_size or current_token_count + sentence_token_count > 120):
            chunks.append(current_chunk)
            current_chunk = []
            current_token_count = 0

        current_chunk.append(sentence)
        current_token_count += sentence_token_count

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
