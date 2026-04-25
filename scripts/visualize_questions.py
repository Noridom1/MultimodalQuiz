from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import sys
import webbrowser
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_image_dir(payload: Any, json_path: Path, fallback_image_dir: Path) -> Path:
    if isinstance(payload, dict):
        explicit_image_dir = payload.get("image_dir")
        if explicit_image_dir:
            return (PROJECT_ROOT / Path(str(explicit_image_dir))).resolve()

        run_id = payload.get("run_id")
        if run_id:
            return (fallback_image_dir / str(run_id)).resolve()

    return fallback_image_dir.resolve()


def normalize_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [normalize_record(item, i) for i, item in enumerate(payload, start=1)]
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return [normalize_record(item, i) for i, item in enumerate(payload["results"], start=1)]
        return [normalize_record(payload, 1)]
    raise TypeError("Expected the input JSON to be a record, a list of records, or a dict with a 'results' list.")


def normalize_record(item: Any, fallback_index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TypeError(f"Each question entry must be an object. Received: {type(item).__name__}")

    question = item.get("question", item)
    if not isinstance(question, dict):
        raise TypeError("Each entry must either be a question object or contain a 'question' object.")

    metadata = question.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    options = question.get("options")
    if not isinstance(options, list):
        options = []

    return {
        "index": item.get("index", fallback_index),
        "id": question.get("id"),
        "question_text": question.get("question_text", ""),
        "options": [str(option) for option in options],
        "correct_answer": question.get("correct_answer"),
        "explanation": question.get("explanation"),
        "question_type": question.get("question_type"),
        "target_concept": question.get("target_concept"),
        "difficulty": question.get("difficulty"),
        "image_grounded": question.get("image_grounded"),
        "reasoning_type": metadata.get("reasoning_type"),
        "image_ref": question.get("associated_image") or item.get("image_url"),
    }


def encode_local_image(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_image_uri(image_ref: Any, json_path: Path, image_dir: Path) -> tuple[str | None, str | None]:
    if not image_ref:
        return None, "No image path"

    image_text = str(image_ref).strip()
    if not image_text:
        return None, "No image path"

    lowered = image_text.lower()
    if lowered.startswith(("http://", "https://", "data:")):
        return image_text, None
    if lowered.startswith("mock://"):
        return None, image_text

    image_path = Path(image_text)
    if not image_path.is_absolute():
      # Avoid duplicating path segments when image_text already contains
      # a parent folder that's the same as json_path.parent (e.g.:
      # json at .../generation and image_ref 'generation/images/1.png'
      # which would otherwise produce .../generation/generation/...)
      rel_parts = image_path.parts
      parent = json_path.parent
      if rel_parts and parent.name == rel_parts[0]:
        candidate1 = (parent / Path(*rel_parts[1:])).resolve()
      else:
        candidate1 = (parent / image_path).resolve()

      candidate2 = (image_dir / image_path.name).resolve()
      image_path = next((candidate for candidate in (candidate1, candidate2) if candidate.exists()), candidate1)

    if not image_path.exists():
        return None, f"Missing file: {image_path}"

    return encode_local_image(image_path), None


def format_value(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def build_html(records: list[dict[str, Any]], json_path: Path, image_dir: Path) -> str:
    cards: list[str] = []
    for record in records:
        image_uri, image_note = resolve_image_uri(record["image_ref"], json_path, image_dir)
        image_html = (
            f'<img src="{html.escape(image_uri, quote=True)}" '
            f'alt="Question {record["index"]} image" loading="lazy">'
            if image_uri
            else f'<div class="image-missing">{html.escape(image_note or "Image unavailable")}</div>'
        )

        option_items = "".join(
            f"<li>{html.escape(option)}</li>" for option in record["options"]
        ) or "<li>No options provided</li>"

        cards.append(
            f"""
            <article class="card">
              <div class="image-panel">
                {image_html}
              </div>
              <div class="content-panel">
                <div class="meta-row">
                  <span>#{html.escape(str(record["index"]))}</span>
                  <span>{html.escape(format_value(record["question_type"]))}</span>
                  <span>{html.escape(format_value(record["difficulty"]))}</span>
                </div>
                <h2>{html.escape(record["question_text"])}</h2>
                <dl class="details">
                  <div><dt>ID</dt><dd>{html.escape(format_value(record["id"]))}</dd></div>
                  <div><dt>Concept</dt><dd>{html.escape(format_value(record["target_concept"]))}</dd></div>
                  <div><dt>Reasoning</dt><dd>{html.escape(format_value(record["reasoning_type"]))}</dd></div>
                  <div><dt>Image Grounded</dt><dd>{html.escape(format_value(record["image_grounded"]))}</dd></div>
                  <div><dt>Answer</dt><dd>{html.escape(format_value(record["correct_answer"]))}</dd></div>
                </dl>
                <section>
                  <h3>Options</h3>
                  <ol>
                    {option_items}
                  </ol>
                </section>
                <section>
                  <h3>Explanation</h3>
                  <p>{html.escape(format_value(record["explanation"]))}</p>
                </section>
              </div>
            </article>
            """
        )

    title = f"Question Viewer: {json_path.name}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4efe7;
      --paper: #fffdf8;
      --ink: #1d1b18;
      --muted: #6b6258;
      --accent: #b14d2f;
      --line: #ddd1c2;
      --shadow: 0 18px 40px rgba(62, 39, 24, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(177, 77, 47, 0.14), transparent 24%),
        linear-gradient(180deg, #f8f3ec 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1200px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}
    header {{
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3.2rem);
      line-height: 1;
    }}
    header p {{
      margin: 0;
      color: var(--muted);
      max-width: 72ch;
    }}
    .stack {{
      display: grid;
      gap: 20px;
    }}
    .card {{
      display: grid;
      grid-template-columns: minmax(280px, 420px) 1fr;
      gap: 0;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 24px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    .image-panel {{
      min-height: 280px;
      background: #efe5d8;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 18px;
    }}
    .image-panel img {{
      width: 100%;
      height: auto;
      max-height: 75vh;
      object-fit: contain;
      border-radius: 14px;
      background: #fff;
    }}
    .image-missing {{
      width: 100%;
      border: 2px dashed #c3ae98;
      border-radius: 16px;
      padding: 24px;
      color: var(--muted);
      text-align: center;
      font-style: italic;
    }}
    .content-panel {{
      padding: 24px;
    }}
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }}
    .meta-row span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    h2 {{
      margin: 0 0 18px;
      font-size: clamp(1.35rem, 2vw, 1.8rem);
      line-height: 1.25;
    }}
    h3 {{
      margin: 18px 0 10px;
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent);
    }}
    .details {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px 16px;
      margin: 0;
    }}
    .details div {{
      padding: 10px 12px;
      border-radius: 14px;
      background: #faf5ee;
      border: 1px solid #efe3d4;
    }}
    dt {{
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    dd {{
      margin: 0;
      font-weight: 600;
    }}
    ol {{
      margin: 0;
      padding-left: 22px;
    }}
    li + li {{
      margin-top: 8px;
    }}
    p {{
      margin: 0;
      line-height: 1.6;
    }}
    @media (max-width: 860px) {{
      .card {{
        grid-template-columns: 1fr;
      }}
      .details {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{html.escape(title)}</h1>
      <p>Loaded {len(records)} question(s) from {html.escape(str(json_path))}. Each card shows the image, prompt text, answer metadata, and explanation.</p>
    </header>
    <section class="stack">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render generated quiz questions and their associated images into an HTML viewer."
    )
    parser.add_argument("questions_json", type=Path, help="Path to the generated question JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML output path. Defaults to <questions_json_stem>_viewer.html next to the input JSON.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "images",
        help="Directory to search when associated_image stores only the image filename.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML file in the default browser after writing it.",
    )
    args = parser.parse_args()

    json_path = args.questions_json.resolve()
    payload = read_json(json_path)
    records = normalize_records(payload)
    image_dir = extract_image_dir(payload, json_path, args.image_dir)

    output_path = args.output.resolve() if args.output else json_path.with_name(f"{json_path.stem}_viewer.html")
    output_path.write_text(build_html(records, json_path, image_dir), encoding="utf-8")

    print(f"Wrote viewer to: {output_path}")
    if args.open:
        webbrowser.open(output_path.as_uri())


if __name__ == "__main__":
    main()
