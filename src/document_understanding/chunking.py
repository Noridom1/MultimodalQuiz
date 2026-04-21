from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class MarkdownBlockKind(str, Enum):
    heading = "heading"
    paragraph = "paragraph"
    table = "table"
    code = "code"
    image = "image"
    details = "details"
    list_item = "list_item"
    quote = "quote"
    raw_html = "raw_html"


class MarkdownBlock(BaseModel):
    id: str
    kind: MarkdownBlockKind
    text: str
    section_path: list[str] = Field(default_factory=list)
    level: int | None = None
    image_path: str | None = None
    caption: str | None = None
    page_idx: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticChunk(BaseModel):
    id: str
    section_path: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    block_kinds: list[MarkdownBlockKind] = Field(default_factory=list)
    text: str
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkInspectionReport(BaseModel):
    source_file: str
    block_count: int
    chunk_count: int
    blocks: list[MarkdownBlock]
    chunks: list[SemanticChunk]
    summary: dict[str, Any] = Field(default_factory=dict)


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*$")
IMAGE_PATTERN = re.compile(r"!\[(?P<alt>.*?)\]\((?P<path>[^)]+)\)")
TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
LIST_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(?P<item>.+?)\s*$")
FENCE_PATTERN = re.compile(r"^\s*(```+|~~~+)\s*(?P<lang>[\w+-]*)\s*$")
DETAILS_OPEN_PATTERN = re.compile(r"^\s*<details\b.*?>\s*$", re.IGNORECASE)
DETAILS_CLOSE_PATTERN = re.compile(r"^\s*</details>\s*$", re.IGNORECASE)
RAW_HTML_PATTERN = re.compile(r"^\s*<[^>]+>\s*$")

ATOMIC_BLOCK_KINDS = {
    MarkdownBlockKind.table,
    MarkdownBlockKind.code,
    MarkdownBlockKind.image,
    MarkdownBlockKind.details,
    MarkdownBlockKind.raw_html,
}


def parse_markdown_blocks(markdown_text: str, *, source_file: str | Path | None = None) -> list[MarkdownBlock]:
    """Parse markdown into typed blocks while preserving tables, code, images, and details blocks."""
    source_name = Path(source_file).name if source_file is not None else "document"
    lines = markdown_text.splitlines()
    blocks: list[MarkdownBlock] = []
    section_stack: list[str] = []
    current_paragraph: list[str] = []
    current_block_index = 0
    line_index = 0

    def current_section_path() -> list[str]:
        return list(section_stack)

    def make_id(kind: str, text: str) -> str:
        nonlocal current_block_index
        slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48] or "item"
        block_id = f"{Path(source_name).stem}::{kind}::{current_block_index}::{slug}"
        current_block_index += 1
        return block_id

    def flush_paragraph() -> None:
        nonlocal current_paragraph
        text = _compact_text(current_paragraph)
        if not text:
            current_paragraph = []
            return
        block_text, image_path = _extract_inline_image(text)
        block = MarkdownBlock(
            id=make_id("paragraph", block_text),
            kind=MarkdownBlockKind.paragraph,
            text=block_text,
            section_path=current_section_path(),
            image_path=image_path,
            metadata={"has_inline_image": image_path is not None},
        )
        blocks.append(block)
        current_paragraph = []

    while line_index < len(lines):
        line = lines[line_index].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            line_index += 1
            continue

        heading_match = HEADING_PATTERN.match(stripped)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            title = heading_match.group("title").strip()
            section_stack[:] = section_stack[: level - 1]
            block = MarkdownBlock(
                id=make_id("heading", title),
                kind=MarkdownBlockKind.heading,
                text=title,
                section_path=list(section_stack),
                level=level,
                metadata={"heading_level": level},
            )
            blocks.append(block)
            section_stack.append(title)
            line_index += 1
            continue

        fence_match = FENCE_PATTERN.match(stripped)
        if fence_match:
            flush_paragraph()
            fence = fence_match.group(1)
            lang = fence_match.group("lang") or None
            code_lines = [line]
            line_index += 1
            while line_index < len(lines):
                code_line = lines[line_index]
                code_lines.append(code_line)
                if code_line.strip().startswith(fence):
                    line_index += 1
                    break
                line_index += 1
            block_text = "\n".join(code_lines)
            blocks.append(
                MarkdownBlock(
                    id=make_id("code", block_text),
                    kind=MarkdownBlockKind.code,
                    text=block_text,
                    section_path=current_section_path(),
                    metadata={"language": lang},
                )
            )
            continue

        if DETAILS_OPEN_PATTERN.match(stripped):
            flush_paragraph()
            details_lines = [line]
            line_index += 1
            while line_index < len(lines):
                details_line = lines[line_index]
                details_lines.append(details_line)
                if DETAILS_CLOSE_PATTERN.match(details_line.strip()):
                    line_index += 1
                    break
                line_index += 1
            block_text = "\n".join(details_lines)
            blocks.append(
                MarkdownBlock(
                    id=make_id("details", block_text),
                    kind=MarkdownBlockKind.details,
                    text=block_text,
                    section_path=current_section_path(),
                    metadata={"format": "html_details"},
                )
            )
            continue

        if _looks_like_table_row(stripped, lines, line_index):
            flush_paragraph()
            table_lines = [line]
            line_index += 1
            while line_index < len(lines):
                table_line = lines[line_index].rstrip()
                if not table_line.strip():
                    break
                if not _looks_like_table_row(table_line, lines, line_index):
                    break
                table_lines.append(table_line)
                line_index += 1
            block_text = "\n".join(table_lines)
            blocks.append(
                MarkdownBlock(
                    id=make_id("table", block_text),
                    kind=MarkdownBlockKind.table,
                    text=block_text,
                    section_path=current_section_path(),
                    metadata={"rows": len(table_lines)},
                )
            )
            continue

        image_match = IMAGE_PATTERN.search(stripped)
        if image_match and len(stripped) == len(image_match.group(0)):
            flush_paragraph()
            blocks.append(
                MarkdownBlock(
                    id=make_id("image", stripped),
                    kind=MarkdownBlockKind.image,
                    text=stripped,
                    section_path=current_section_path(),
                    image_path=image_match.group("path").strip(),
                    caption=image_match.group("alt").strip() or None,
                    metadata={"alt_text": image_match.group("alt").strip()},
                )
            )
            line_index += 1
            continue

        list_match = LIST_PATTERN.match(stripped)
        if list_match:
            flush_paragraph()
            blocks.append(
                MarkdownBlock(
                    id=make_id("list_item", stripped),
                    kind=MarkdownBlockKind.list_item,
                    text=list_match.group("item").strip(),
                    section_path=current_section_path(),
                    metadata={"marker": stripped[:2].strip()},
                )
            )
            line_index += 1
            continue

        if RAW_HTML_PATTERN.match(stripped):
            flush_paragraph()
            blocks.append(
                MarkdownBlock(
                    id=make_id("raw_html", stripped),
                    kind=MarkdownBlockKind.raw_html,
                    text=stripped,
                    section_path=current_section_path(),
                )
            )
            line_index += 1
            continue

        current_paragraph.append(stripped)
        line_index += 1

    flush_paragraph()
    return blocks


def build_semantic_chunks(
    blocks: list[MarkdownBlock],
    *,
    max_tokens: int = 280,
    overlap_blocks: int = 1,
) -> list[SemanticChunk]:
    """Group consecutive blocks into semantic chunks while preserving special block boundaries."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    if overlap_blocks < 0:
        raise ValueError("overlap_blocks must be non-negative")

    chunks: list[SemanticChunk] = []
    current_blocks: list[MarkdownBlock] = []
    current_tokens = 0
    current_section_path: list[str] = []

    def flush_chunk() -> None:
        nonlocal current_blocks, current_tokens, current_section_path
        if not current_blocks:
            return
        chunk_text = "\n\n".join(block.text for block in current_blocks).strip()
        if not chunk_text:
            current_blocks = []
            current_tokens = 0
            return
        chunk_id = _make_chunk_id(current_blocks[0], len(chunks))
        chunks.append(
            SemanticChunk(
                id=chunk_id,
                section_path=list(current_section_path),
                block_ids=[block.id for block in current_blocks],
                block_kinds=[block.kind for block in current_blocks],
                text=chunk_text,
                token_count=_count_tokens(chunk_text),
                metadata={
                    "contains_atomic_block": any(block.kind in ATOMIC_BLOCK_KINDS for block in current_blocks),
                    "block_count": len(current_blocks),
                },
            )
        )
        current_blocks = current_blocks[-overlap_blocks:] if overlap_blocks else []
        current_tokens = _count_tokens("\n\n".join(block.text for block in current_blocks)) if current_blocks else 0

    for block in blocks:
        if block.kind == MarkdownBlockKind.heading:
            flush_chunk()
            current_section_path = list(block.section_path) + [block.text]
            continue

        if block.section_path != current_section_path:
            flush_chunk()
            current_section_path = list(block.section_path)

        block_tokens = _count_tokens(block.text)
        if block.kind in ATOMIC_BLOCK_KINDS:
            flush_chunk()
            chunks.append(
                SemanticChunk(
                    id=_make_chunk_id(block, len(chunks)),
                    section_path=list(current_section_path),
                    block_ids=[block.id],
                    block_kinds=[block.kind],
                    text=block.text.strip(),
                    token_count=block_tokens,
                    metadata={"atomic": True, "block_kind": block.kind.value},
                )
            )
            current_blocks = []
            current_tokens = 0
            continue

        if current_blocks and current_tokens + block_tokens > max_tokens:
            flush_chunk()

        current_blocks.append(block)
        current_tokens += block_tokens

    flush_chunk()
    return chunks


def semantic_chunk(sentences: list[str], *, max_chunk_size: int = 5) -> list[list[str]]:
    """Fallback sentence grouping for text-only use cases."""
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


def _compact_text(lines: list[str]) -> str:
    return " ".join(line.strip() for line in lines if line.strip()).strip()


def _extract_inline_image(text: str) -> tuple[str, str | None]:
    match = IMAGE_PATTERN.search(text)
    if not match:
        return text, None
    image_path = match.group("path").strip()
    cleaned = IMAGE_PATTERN.sub(match.group("alt").strip(), text).strip()
    return cleaned, image_path


def _looks_like_table_row(line: str, lines: list[str], index: int) -> bool:
    if "|" not in line:
        return False
    if TABLE_SEPARATOR_PATTERN.match(line):
        return True
    next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
    return bool(next_line and TABLE_SEPARATOR_PATTERN.match(next_line))


def _count_tokens(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _count_enum_values(values: Iterable[Enum]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _make_chunk_id(block: MarkdownBlock, index: int) -> str:
    slug_source = block.caption or block.text
    slug = re.sub(r"[^a-z0-9]+", "_", slug_source.lower()).strip("_")[:48] or "chunk"
    return f"{block.id}::chunk::{index}::{slug}"


def build_chunk_inspection_report(
    markdown_text: str,
    *,
    source_file: str | Path,
    max_tokens: int = 280,
    overlap_blocks: int = 1,
) -> ChunkInspectionReport:
    blocks = parse_markdown_blocks(markdown_text, source_file=source_file)
    chunks = build_semantic_chunks(blocks, max_tokens=max_tokens, overlap_blocks=overlap_blocks)
    source_file_str = str(source_file)

    return ChunkInspectionReport(
        source_file=source_file_str,
        block_count=len(blocks),
        chunk_count=len(chunks),
        blocks=blocks,
        chunks=chunks,
        summary={
            "source_name": Path(source_file_str).name,
            "block_kinds": _count_enum_values(block.kind for block in blocks),
            "chunk_token_counts": [chunk.token_count for chunk in chunks],
            "atomic_chunks": sum(1 for chunk in chunks if chunk.metadata.get("atomic")),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect how a markdown document is split into blocks and semantic chunks."
    )
    parser.add_argument("markdown_path", type=Path, help="Path to the extracted markdown file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the chunk inspection JSON report (defaults to <markdown_stem>_chunks.json)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=280,
        help="Target token budget per chunk",
    )
    parser.add_argument(
        "--overlap-blocks",
        type=int,
        default=1,
        help="How many previous blocks to carry into the next chunk",
    )
    args = parser.parse_args()

    markdown_path = args.markdown_path
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    markdown_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
    report = build_chunk_inspection_report(
        markdown_text,
        source_file=markdown_path,
        max_tokens=args.max_tokens,
        overlap_blocks=args.overlap_blocks,
    )

    output_path = args.output or markdown_path.with_name(f"{markdown_path.stem}_chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Saved chunk inspection report to {output_path} "
        f"({report.block_count} blocks, {report.chunk_count} chunks)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())