from __future__ import annotations

import argparse
import os

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ParsedDocument:
    sections: list[str]
    paragraphs: list[str]
    figures: list[str]
    captions: list[str]


_SECTION_PATTERN = re.compile(
    r"^(?:"
    r"(?P<markdown>#{1,6})\s+(?P<markdown_title>.+?)"
    r"|(?P<numbered>\d+(?:\.\d+)*\.?)[\s\-:]+(?P<numbered_title>.+?)"
    r"|(?P<all_caps>[A-Z][A-Z0-9\s\-]{4,})"
    r")$"
)

_FIGURE_PATTERN = re.compile(r"^(?:figure|fig\.)\s*\d+[\s:\-–—]*(?P<caption>.+)?$", re.IGNORECASE)
_IMAGE_PATTERN = re.compile(r"\[image:\s*(?P<label>.+?)\]", re.IGNORECASE)


def parse_document(document_path: str | Path) -> ParsedDocument:
    """Parse a PDF, Markdown, or text document into a structured representation."""
    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf_text(path)
    elif suffix in {".md", ".markdown"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    lines = _normalize_lines(text.splitlines())
    sections, paragraphs, figures, captions = _extract_structure(lines)
    return ParsedDocument(
        sections=sections,
        paragraphs=paragraphs,
        figures=figures,
        captions=captions,
    )


def main(document_path: str | Path) -> ParsedDocument:
    """Parse a document and return the structured result.

    This wrapper makes the parser easy to call from unit tests.
    """
    return parse_document(document_path)


def _read_pdf_text(path: Path) -> str:
    """Extract Markdown from a PDF using MinerU, then fall back to local text extraction."""
    print("[parser] Attempting to extract PDF text with MinerU...")
    pdf_text = _extract_pdf_with_mineru(path)
    if pdf_text is not None:
        return pdf_text

    print("[parser] Falling back to local PDF text extraction...")
    pdf_text = _extract_pdf_with_pymupdf(path)
    if pdf_text is not None:
        return pdf_text

    print("[parser] Falling back to pypdf extraction...")
    pdf_text = _extract_pdf_with_pypdf(path)
    if pdf_text is not None:
        return pdf_text

    raise RuntimeError(
        "PDF parsing requires MinerU, 'pymupdf', or 'pypdf'. Install one of them to enable PDF support."
    )


def _extract_pdf_with_mineru(path: Path) -> str | None:
    """Convert a PDF to Markdown using MinerU CLI and return the generated markdown text."""
    mineru_executable = shutil.which("mineru")
    if mineru_executable is None:
        return None

    with tempfile.TemporaryDirectory(prefix="mineru_parse_") as temporary_dir:
        output_dir = Path(temporary_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            mineru_executable,
            "-p",
            str(path),
            "-o",
            str(output_dir),
            "-b",
            os.environ.get("MINERU_BACKEND", "pipeline"),
        ]

        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return None
        except subprocess.CalledProcessError:
            return None

        markdown_path = _find_mineru_markdown(output_dir, path.stem)
        if markdown_path is None or not markdown_path.exists():
            return None

        markdown_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
        if not markdown_text.strip():
            return None

        return markdown_text


def _find_mineru_markdown(output_dir: Path, document_stem: str) -> Path | None:
    preferred_names = [
        f"{document_stem}.md",
        f"{document_stem}_content.md",
    ]

    for candidate_name in preferred_names:
        matches = list(output_dir.rglob(candidate_name))
        if matches:
            return matches[0]

    markdown_files = [path for path in output_dir.rglob("*.md") if path.is_file()]
    if not markdown_files:
        return None

    for markdown_path in markdown_files:
        if markdown_path.name.endswith(".md"):
            return markdown_path

    return markdown_files[0]


def _extract_pdf_with_pypdf(path: Path) -> str | None:
    try:
        from pypdf import PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found]
        except Exception:
            return None

    reader = PdfReader(str(path))
    page_texts: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted.strip():
            page_texts.append(extracted)
    return "\n\n".join(page_texts)


def _extract_pdf_with_pymupdf(path: Path) -> str | None:
    try:
        import fitz  # type: ignore
    except Exception:
        return None

    doc = fitz.open(str(path))
    try:
        page_texts = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    text = "\n\n".join(page_texts)
    return text if text.strip() else ""


def _normalize_lines(lines: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    previous_blank = False
    for raw_line in lines:
        line = raw_line.replace("\u00a0", " ").strip()
        if not line:
            if not previous_blank:
                normalized.append("")
            previous_blank = True
            continue
        normalized.append(line)
        previous_blank = False
    return normalized


def _extract_structure(lines: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    sections: list[str] = []
    paragraphs: list[str] = []
    figures: list[str] = []
    captions: list[str] = []

    current_paragraph: list[str] = []

    def flush_paragraph() -> None:
        if current_paragraph:
            paragraph = " ".join(current_paragraph).strip()
            if paragraph:
                paragraphs.append(paragraph)
            current_paragraph.clear()

    for line in lines:
        if not line:
            flush_paragraph()
            continue

        section_title = _match_section_title(line)
        if section_title:
            flush_paragraph()
            sections.append(section_title)
            continue

        figure_caption = _match_figure_caption(line)
        if figure_caption:
            flush_paragraph()
            figures.append(line)
            captions.append(figure_caption)
            continue

        image_label = _match_image_label(line)
        if image_label:
            flush_paragraph()
            figures.append(image_label)
            captions.append(image_label)
            continue

        current_paragraph.append(line)

    flush_paragraph()
    return sections, paragraphs, figures, captions


def _match_section_title(line: str) -> str | None:
    if not line or len(line) > 180:
        return None

    match = _SECTION_PATTERN.match(line)
    if not match:
        return None

    title = match.group("markdown_title") or match.group("numbered_title") or match.group("all_caps")
    if title is None:
        return None
    title = title.strip().rstrip(".")
    if len(title.split()) <= 1 and not match.group("markdown"):
        return None
    return title


def _match_figure_caption(line: str) -> str | None:
    match = _FIGURE_PATTERN.match(line)
    if not match:
        return None
    caption = (match.group("caption") or "").strip()
    return caption or line


def _match_image_label(line: str) -> str | None:
    match = _IMAGE_PATTERN.search(line)
    if not match:
        return None
    return match.group("label").strip()


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Parse a PDF, Markdown, or text document.")
    parser.add_argument("document_path", help="Path to the input document")
    args = parser.parse_args()

    result = main(args.document_path)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
