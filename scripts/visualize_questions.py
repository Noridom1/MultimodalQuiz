"""Generate a standalone interactive quiz HTML from a questions.json file.

Usage
-----
python scripts/visualize_questions.py --json_path outputs/.../generation/questions.json
python scripts/visualize_questions.py --json_path questions.json --image_dir /path/to/images --output quiz.html --open
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import webbrowser
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
        items = payload["results"]
    elif isinstance(payload, dict):
        items = [payload]
    else:
        raise TypeError("Expected a list of question objects or a dict with a 'results' list.")

    return [_normalize_one(item, i) for i, item in enumerate(items, start=1)]


def _normalize_one(item: Any, fallback_index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TypeError(f"Each question entry must be a JSON object. Got: {type(item).__name__}")

    # Support both flat records and wrapped {"question": {...}} records
    q = item.get("question", item)
    if not isinstance(q, dict):
        raise TypeError("Each entry must be a question object or contain a 'question' key.")

    meta = q.get("metadata") or {}
    options: list[str] = [str(o) for o in (q.get("options") or [])]
    correct_raw: str = str(q.get("correct_answer", "")).strip()

    return {
        "index": item.get("index", fallback_index),
        "id": q.get("id") or f"q_{fallback_index}",
        "question_text": q.get("question_text", ""),
        "options": options,
        "correct_answer_raw": correct_raw,
        "correct_index": _resolve_correct_index(correct_raw, options),
        "explanation": q.get("explanation") or "",
        "question_type": q.get("question_type") or "multiple_choice",
        "target_concept": q.get("target_concept") or "",
        "difficulty": q.get("difficulty") or "",
        "image_grounded": bool(q.get("image_grounded")),
        "reasoning_type": meta.get("reasoning_type") or "",
        "image_ref": q.get("associated_image") or item.get("image_url") or "",
    }


def _resolve_correct_index(correct_raw: str, options: list[str]) -> int:
    """Return the 0-based index of the correct option.

    Handles three formats:
    - Letter label:  "A", "B", "C", "D"
    - Numeric label: "1", "2", "3", "4"
    - Full text:     the exact (or contained) option text
    """
    if not correct_raw or not options:
        return -1

    # Letter label – strip trailing punctuation/space just in case
    upper = correct_raw.upper().rstrip(").: ")
    if len(upper) == 1 and upper.isalpha():
        idx = ord(upper) - ord("A")
        return idx if 0 <= idx < len(options) else -1

    # Numeric label
    if correct_raw.isdigit():
        idx = int(correct_raw) - 1
        return idx if 0 <= idx < len(options) else -1

    # Full text – try exact match, then containment
    for i, opt in enumerate(options):
        if opt.strip() == correct_raw:
            return i
    for i, opt in enumerate(options):
        if correct_raw in opt or opt in correct_raw:
            return i

    return -1


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _encode_local_image(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def resolve_image_uri(image_ref: str, json_path: Path, image_dir: Path) -> str | None:
    """Return a data-URI or http URL for the image, or None if unavailable."""
    if not image_ref:
        return None

    low = image_ref.lower()
    if low.startswith(("http://", "https://", "data:")):
        return image_ref
    if low.startswith("mock://"):
        return None

    img_path = Path(image_ref)

    if not img_path.is_absolute():
        # Avoid duplicating the parent segment that may already appear in the ref
        # e.g. json at .../generation/ and ref = "generation/images/1.png"
        rel_parts = img_path.parts
        parent = json_path.parent
        if rel_parts and parent.name == rel_parts[0]:
            candidate1 = (parent / Path(*rel_parts[1:])).resolve()
        else:
            candidate1 = (parent / img_path).resolve()

        candidate2 = (image_dir / img_path.name).resolve()
        img_path = next((c for c in (candidate1, candidate2) if c.exists()), candidate1)

    if not img_path.exists():
        return None

    return _encode_local_image(img_path)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def build_html(records: list[dict[str, Any]], json_path: Path, image_dir: Path) -> str:
    # Attach resolved image data-URIs and strip the large binary blobs from the
    # JS payload (keep only the uri reference keyed as "image_uri").
    for rec in records:
        rec["image_uri"] = resolve_image_uri(rec.pop("image_ref"), json_path, image_dir) or ""

    questions_json = json.dumps(records, ensure_ascii=False)
    total = len(records)
    source_label = json_path.name

    return f"""\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Quiz – {source_label}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    /* Smooth fade between questions */
    #question-card {{ transition: opacity 0.25s ease; }}
    #question-card.fade-out {{ opacity: 0; }}

    /* Option button selected state */
    .opt-btn.selected {{
      @apply ring-2 ring-indigo-500 bg-indigo-50;
    }}
    /* Correct / wrong feedback */
    .opt-btn.correct  {{ background-color:#d1fae5; border-color:#34d399; }}
    .opt-btn.wrong    {{ background-color:#fee2e2; border-color:#f87171; }}
    .opt-btn.reveal   {{ background-color:#d1fae5; border-color:#34d399; }}
  </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-100 to-indigo-50 flex flex-col items-center py-10 px-4">

  <!-- ── Progress bar ──────────────────────────────────────────────── -->
  <div class="w-full max-w-3xl mb-6">
    <div class="flex justify-between text-sm text-slate-500 mb-1">
      <span id="progress-label">Question 1 of {total}</span>
      <label class="flex items-center gap-2 cursor-pointer select-none">
        <input id="show-explanation" type="checkbox" class="accent-indigo-600 w-4 h-4" />
        <span>Show explanation after answering</span>
      </label>
    </div>
    <div class="h-2 bg-slate-200 rounded-full overflow-hidden">
      <div id="progress-bar" class="h-full bg-indigo-500 rounded-full transition-all duration-500"
           style="width:{round(1/total*100,2) if total else 0}%"></div>
    </div>
  </div>

  <!-- ── Question card ─────────────────────────────────────────────── -->
  <div id="question-card"
       class="w-full max-w-3xl bg-white rounded-3xl shadow-xl overflow-hidden">

    <!-- Image panel -->
    <div id="img-panel" class="hidden bg-slate-100 flex justify-center items-center p-4 border-b border-slate-200">
      <img id="q-image" src="" alt="Question image"
           class="max-h-72 w-auto object-contain rounded-xl shadow" />
    </div>

    <!-- Body -->
    <div class="p-8">
      <!-- Meta chips -->
      <div class="flex flex-wrap gap-2 mb-4" id="meta-chips"></div>

      <!-- Question text -->
      <h2 id="q-text" class="text-xl font-semibold text-slate-800 mb-6 leading-snug"></h2>

      <!-- Options -->
      <div id="options-list" class="flex flex-col gap-3"></div>

      <!-- Feedback / explanation -->
      <div id="feedback-box" class="hidden mt-6 p-4 rounded-2xl border text-sm leading-relaxed"></div>

      <!-- Action buttons -->
      <div class="mt-8 flex gap-3">
        <button id="submit-btn"
                class="px-6 py-2.5 rounded-xl bg-indigo-600 text-white font-medium
                       hover:bg-indigo-700 active:scale-95 transition disabled:opacity-40"
                disabled>
          Submit
        </button>
        <button id="next-btn"
                class="hidden px-6 py-2.5 rounded-xl bg-slate-700 text-white font-medium
                       hover:bg-slate-800 active:scale-95 transition">
          Next →
        </button>
      </div>
    </div>
  </div>

  <!-- ── Results screen (hidden until quiz ends) ───────────────────── -->
  <div id="results-screen" class="hidden w-full max-w-3xl mt-8">
    <div class="bg-white rounded-3xl shadow-xl p-8 text-center">
      <div class="text-6xl mb-4">🎉</div>
      <h2 class="text-2xl font-bold text-slate-800 mb-2">Quiz Complete</h2>
      <p id="score-text" class="text-slate-500 mb-8 text-lg"></p>
      <div class="overflow-x-auto mb-8">
        <table class="w-full text-sm text-left border-collapse">
          <thead>
            <tr class="border-b border-slate-200 text-slate-500 text-xs uppercase tracking-wide">
              <th class="py-2 pr-4">#</th>
              <th class="py-2 pr-4">Question</th>
              <th class="py-2 pr-4">Selected</th>
              <th class="py-2 pr-4">Result</th>
              <th class="py-2">Time (s)</th>
            </tr>
          </thead>
          <tbody id="results-table"></tbody>
        </table>
      </div>
      <button id="download-btn"
              class="px-8 py-3 rounded-xl bg-emerald-600 text-white font-semibold
                     hover:bg-emerald-700 active:scale-95 transition text-base">
        ⬇ Download Results (JSON)
      </button>
    </div>
  </div>

  <script>
  // ── Embedded question data ─────────────────────────────────────────
  const QUESTIONS = {questions_json};

  // ── State ──────────────────────────────────────────────────────────
  let current   = 0;
  let selected  = null;   // index of chosen option
  let submitted = false;
  let qStartMs  = Date.now();
  const results = [];     // collected responses

  // ── DOM refs ───────────────────────────────────────────────────────
  const progressLabel = document.getElementById('progress-label');
  const progressBar   = document.getElementById('progress-bar');
  const imgPanel      = document.getElementById('img-panel');
  const qImage        = document.getElementById('q-image');
  const metaChips     = document.getElementById('meta-chips');
  const qText         = document.getElementById('q-text');
  const optionsList   = document.getElementById('options-list');
  const feedbackBox   = document.getElementById('feedback-box');
  const submitBtn     = document.getElementById('submit-btn');
  const nextBtn       = document.getElementById('next-btn');
  const questionCard  = document.getElementById('question-card');
  const resultsScreen = document.getElementById('results-screen');
  const showExpCb     = document.getElementById('show-explanation');
  const downloadBtn   = document.getElementById('download-btn');

  // ── Render a question ──────────────────────────────────────────────
  function renderQuestion(idx) {{
    const q = QUESTIONS[idx];
    selected  = null;
    submitted = false;
    qStartMs  = Date.now();

    // Progress
    const pct = Math.round((idx / QUESTIONS.length) * 100);
    progressBar.style.width = pct + '%';
    progressLabel.textContent = `Question ${{idx + 1}} of ${{QUESTIONS.length}}`;

    // Image
    if (q.image_uri) {{
      qImage.src = q.image_uri;
      imgPanel.classList.remove('hidden');
      imgPanel.classList.add('flex');
    }} else {{
      imgPanel.classList.add('hidden');
      imgPanel.classList.remove('flex');
    }}

    // Meta chips
    metaChips.innerHTML = '';
    [
      ['#' + (idx + 1),           'bg-slate-100 text-slate-500'],
      [q.difficulty || 'unknown', difficultyColor(q.difficulty)],
      [q.question_type,           'bg-violet-50 text-violet-600'],
      [q.target_concept,          'bg-amber-50 text-amber-700'],
    ].forEach(([label, cls]) => {{
      if (!label) return;
      const chip = document.createElement('span');
      chip.className = `text-xs font-medium px-3 py-1 rounded-full ${{cls}}`;
      chip.textContent = label;
      metaChips.appendChild(chip);
    }});

    // Question text
    qText.textContent = q.question_text;

    // Options
    optionsList.innerHTML = '';
    q.options.forEach((opt, i) => {{
      const btn = document.createElement('button');
      btn.className = `opt-btn w-full text-left px-4 py-3 rounded-xl border border-slate-200
                       bg-white hover:bg-indigo-50 hover:border-indigo-300 transition text-slate-700 font-medium`;
      btn.textContent = opt;
      btn.dataset.index = i;
      btn.addEventListener('click', () => selectOption(i));
      optionsList.appendChild(btn);
    }});

    // Reset UI
    feedbackBox.classList.add('hidden');
    feedbackBox.textContent = '';
    submitBtn.disabled = true;
    submitBtn.classList.remove('hidden');
    nextBtn.classList.add('hidden');
  }}

  function difficultyColor(d) {{
    return {{ easy: 'bg-green-50 text-green-700',
              medium: 'bg-yellow-50 text-yellow-700',
              hard: 'bg-red-50 text-red-700' }}[d] || 'bg-slate-100 text-slate-500';
  }}

  // ── Option selection ───────────────────────────────────────────────
  function selectOption(idx) {{
    if (submitted) return;
    selected = idx;
    optionsList.querySelectorAll('.opt-btn').forEach((btn, i) => {{
      btn.classList.toggle('selected', i === idx);
      // Tailwind @apply doesn't work at runtime, so set inline ring manually
      if (i === idx) {{
        btn.style.outline = '2px solid #6366f1';
        btn.style.background = '#eef2ff';
      }} else {{
        btn.style.outline = '';
        btn.style.background = '';
      }}
    }});
    submitBtn.disabled = false;
  }}

  // ── Submit ─────────────────────────────────────────────────────────
  submitBtn.addEventListener('click', () => {{
    if (selected === null || submitted) return;
    submitted = true;

    const q       = QUESTIONS[current];
    const timeMs  = Date.now() - qStartMs;
    const correct = selected === q.correct_index;

    results.push({{
      question_id:     q.id,
      question_text:   q.question_text,
      selected_option: q.options[selected] ?? null,
      selected_index:  selected,
      correct_index:   q.correct_index,
      is_correct:      correct,
      time_spent_ms:   timeMs,
      time_spent_s:    parseFloat((timeMs / 1000).toFixed(2)),
    }});

    // Highlight options
    optionsList.querySelectorAll('.opt-btn').forEach((btn, i) => {{
      btn.style.outline = '';
      btn.style.background = '';
      if (i === q.correct_index) {{
        btn.style.background = '#d1fae5';
        btn.style.borderColor = '#34d399';
      }} else if (i === selected && !correct) {{
        btn.style.background = '#fee2e2';
        btn.style.borderColor = '#f87171';
      }}
    }});

    // Feedback
    if (showExpCb.checked) {{
      feedbackBox.classList.remove('hidden');
      feedbackBox.className = `mt-6 p-4 rounded-2xl border text-sm leading-relaxed ${{
        correct ? 'bg-green-50 border-green-300 text-green-800'
                : 'bg-red-50 border-red-300 text-red-800'
      }}`;
      const verdict = correct ? '✅ Correct!' : `❌ Incorrect. Correct answer: ${{q.options[q.correct_index] ?? q.correct_answer_raw}}`;
      feedbackBox.innerHTML = `<strong>${{verdict}}</strong>${{
        q.explanation ? `<br><br><span class="text-slate-600">${{escHtml(q.explanation)}}</span>` : ''
      }}`;
    }}

    submitBtn.classList.add('hidden');
    nextBtn.classList.remove('hidden');
  }});

  // ── Next ───────────────────────────────────────────────────────────
  nextBtn.addEventListener('click', () => {{
    questionCard.classList.add('fade-out');
    setTimeout(() => {{
      current++;
      if (current < QUESTIONS.length) {{
        renderQuestion(current);
      }} else {{
        showResults();
      }}
      questionCard.classList.remove('fade-out');
    }}, 220);
  }});

  // ── Results ────────────────────────────────────────────────────────
  function showResults() {{
    progressBar.style.width = '100%';
    progressLabel.textContent = 'Complete!';
    questionCard.classList.add('hidden');
    resultsScreen.classList.remove('hidden');

    const correct = results.filter(r => r.is_correct).length;
    document.getElementById('score-text').textContent =
      `You got ${{correct}} out of ${{results.length}} correct (${{Math.round(correct/results.length*100)}}%)`;

    const tbody = document.getElementById('results-table');
    results.forEach((r, i) => {{
      const tr = document.createElement('tr');
      tr.className = 'border-b border-slate-100 hover:bg-slate-50';
      tr.innerHTML = `
        <td class="py-2 pr-4 text-slate-400">${{i + 1}}</td>
        <td class="py-2 pr-4 text-slate-700 max-w-xs truncate" title="${{escHtml(r.question_text)}}">${{escHtml(r.question_text.slice(0, 60) + (r.question_text.length > 60 ? '…' : ''))}}</td>
        <td class="py-2 pr-4 text-slate-600">${{escHtml(r.selected_option ?? '—')}}</td>
        <td class="py-2 pr-4 font-semibold ${{r.is_correct ? 'text-emerald-600' : 'text-red-500'}}">${{r.is_correct ? '✓ Correct' : '✗ Wrong'}}</td>
        <td class="py-2 text-slate-500">${{r.time_spent_s}}</td>
      `;
      tbody.appendChild(tr);
    }});
  }}

  // ── Download ───────────────────────────────────────────────────────
  downloadBtn.addEventListener('click', () => {{
    const payload = {{
      source_file: "{source_label}",
      total_questions: QUESTIONS.length,
      correct: results.filter(r => r.is_correct).length,
      responses: results,
    }};
    const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json' }});
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'user_study_results.json';
    a.click();
    URL.revokeObjectURL(url);
  }});

  // ── Utility ────────────────────────────────────────────────────────
  function escHtml(str) {{
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }}

  // ── Boot ───────────────────────────────────────────────────────────
  if (QUESTIONS.length === 0) {{
    qText.textContent = 'No questions found in the JSON file.';
    submitBtn.disabled = true;
  }} else {{
    renderQuestion(0);
  }}
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a standalone interactive quiz HTML from a questions.json file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json_path",
        type=Path,
        required=True,
        help="Path to the questions.json file.",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "images",
        help="Fallback directory to look for images when the ref is a bare filename.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to index.html next to the JSON file.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML in the default browser.",
    )
    args = parser.parse_args()

    json_path: Path = args.json_path.resolve()
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    payload  = read_json(json_path)
    records  = normalize_records(payload)
    html     = build_html(records, json_path, args.image_dir.resolve())

    out: Path = args.output.resolve() if args.output else json_path.parent / "index.html"
    out.write_text(html, encoding="utf-8")

    print(f"✓ Wrote interactive quiz ({len(records)} questions) → {out}")
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
