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

QUIZ_UPLOAD_URL="https://script.google.com/macros/s/AKfycbzfO0Ndr9C1t5YDpda7_wSOkVKprgZvMtZUoEACSOcBiWskbIqwQHFrSUuvMt_G0Pw/exec"

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

def build_html(
  records: list[dict[str, Any]],
  json_path: Path,
  image_dir: Path,
  show_score: bool = True,
  upload_url: str | None = None,
) -> str:
    # Attach resolved image data-URIs
    for rec in records:
        rec["image_uri"] = resolve_image_uri(rec.pop("image_ref"), json_path, image_dir) or ""

    questions_json = json.dumps(records, ensure_ascii=False)
    show_score_js = 'true' if show_score else 'false'
    upload_url_js = 'null' if not upload_url else json.dumps(str(upload_url))
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
    #question-card {{ transition: opacity 0.25s ease; }}
    #question-card.fade-out {{ opacity: 0; }}
    .opt-btn.selected {{ @apply ring-2 ring-indigo-500 bg-indigo-50; }}
  </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-100 to-indigo-50 flex flex-col items-center py-10 px-4">

  <div class="w-full max-w-3xl mb-6">
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center text-sm text-slate-500 mb-4 gap-3">
      <span id="progress-label" class="font-medium">Question 1 of {total}</span>
      
      <div class="flex flex-wrap gap-4">
        <label class="flex items-center gap-2 cursor-pointer select-none">
          <input id="show-validation" type="checkbox" class="accent-indigo-600 w-4 h-4" />
          <span>Show correct/wrong marks</span>
        </label>

        <label class="flex items-center gap-2 cursor-pointer select-none">
          <input id="show-explanation" type="checkbox" class="accent-indigo-600 w-4 h-4" />
          <span>Show explanation</span>
        </label>
      </div>
    </div>
    
    <div class="h-2 bg-slate-200 rounded-full overflow-hidden">
      <div id="progress-bar" class="h-full bg-indigo-500 rounded-full transition-all duration-500"
           style="width:{round(1/total*100,2) if total else 0}%"></div>
    </div>
  </div>

  <div id="question-card" class="w-full max-w-3xl bg-white rounded-3xl shadow-xl overflow-hidden">
    <div id="img-panel" class="hidden bg-slate-100 flex justify-center items-center p-4 border-b border-slate-200">
      <img id="q-image" src="" alt="Question image" class="max-h-72 w-auto object-contain rounded-xl shadow" />
    </div>

    <div class="p-8">
      <div class="flex flex-wrap gap-2 mb-4" id="meta-chips"></div>
      <h2 id="q-text" class="text-xl font-semibold text-slate-800 mb-6 leading-snug"></h2>
      <div id="options-list" class="flex flex-col gap-3"></div>
      <div id="feedback-box" class="hidden mt-6 p-4 rounded-2xl border text-sm leading-relaxed"></div>

      <div class="mt-8 flex gap-3">
        <button id="submit-btn" class="px-6 py-2.5 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700 active:scale-95 transition disabled:opacity-40" disabled>
          Submit
        </button>
        <button id="next-btn" class="hidden px-6 py-2.5 rounded-xl bg-slate-700 text-white font-medium hover:bg-slate-800 active:scale-95 transition">
          Next →
        </button>
      </div>
    </div>
  </div>

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
      <div id="username-panel" class="hidden mb-4 text-left">
        <label for="username-input" class="block text-sm font-medium text-slate-700 mb-2">Your name</label>
        <input id="username-input" type="text" autocomplete="name" placeholder="Enter your name" class="w-full rounded-xl border border-slate-300 px-4 py-3 text-slate-800 outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500" />
      </div>
      <button id="download-btn" class="px-8 py-3 rounded-xl bg-emerald-600 text-white font-semibold hover:bg-emerald-700 active:scale-95 transition text-base">
        ⬇ Download Results (JSON)
      </button>
    </div>
  </div>

  <script>
  const QUESTIONS = {questions_json};
  const SHOW_SCORE = {show_score_js};
  const UPLOAD_URL = {upload_url_js};
  let current = 0, selected = null, submitted = false, qStartMs = Date.now();
  const results = [];

  const progressBar = document.getElementById('progress-bar'),
        progressLabel = document.getElementById('progress-label'),
        imgPanel = document.getElementById('img-panel'),
        qImage = document.getElementById('q-image'),
        metaChips = document.getElementById('meta-chips'),
        qText = document.getElementById('q-text'),
        optionsList = document.getElementById('options-list'),
        feedbackBox = document.getElementById('feedback-box'),
        submitBtn = document.getElementById('submit-btn'),
        nextBtn = document.getElementById('next-btn'),
        questionCard = document.getElementById('question-card'),
        resultsScreen = document.getElementById('results-screen'),
        showExpCb = document.getElementById('show-explanation'),
        showValCb = document.getElementById('show-validation'),
        usernamePanel = document.getElementById('username-panel'),
        usernameInput = document.getElementById('username-input'),
        downloadBtn = document.getElementById('download-btn');

  function renderQuestion(idx) {{
    const q = QUESTIONS[idx];
    selected = null; submitted = false; qStartMs = Date.now();
    progressBar.style.width = Math.round((idx / QUESTIONS.length) * 100) + '%';
    progressLabel.textContent = `Question ${{idx + 1}} of ${{QUESTIONS.length}}`;

    if (q.image_uri) {{ qImage.src = q.image_uri; imgPanel.classList.replace('hidden', 'flex'); }} 
    else {{ imgPanel.classList.replace('flex', 'hidden'); }}

    metaChips.innerHTML = '';
    [['#' + (idx + 1), 'bg-slate-100 text-slate-500'], [q.difficulty, difficultyColor(q.difficulty)], [q.question_type, 'bg-violet-50 text-violet-600']].forEach(([l, c]) => {{
      if (!l) return;
      const span = document.createElement('span');
      span.className = `text-xs font-medium px-3 py-1 rounded-full ${{c}}`;
      span.textContent = l;
      metaChips.appendChild(span);
    }});

    qText.textContent = q.question_text;
    optionsList.innerHTML = '';
    q.options.forEach((opt, i) => {{
      const btn = document.createElement('button');
      btn.className = "opt-btn w-full text-left px-4 py-3 rounded-xl border border-slate-200 bg-white hover:bg-indigo-50 transition text-slate-700 font-medium";
      btn.textContent = opt;
      btn.onclick = () => selectOption(i);
      optionsList.appendChild(btn);
    }});

    feedbackBox.classList.add('hidden');
    submitBtn.disabled = true;
    submitBtn.classList.remove('hidden');
    nextBtn.classList.add('hidden');
  }}

  function difficultyColor(d) {{
    return {{ easy: 'bg-green-50 text-green-700', medium: 'bg-yellow-50 text-yellow-700', hard: 'bg-red-50 text-red-700' }}[d] || 'bg-slate-100 text-slate-500';
  }}

  function selectOption(idx) {{
    if (submitted) return;
    selected = idx;
    Array.from(optionsList.children).forEach((btn, i) => {{
      btn.style.outline = (i === idx) ? '2px solid #6366f1' : '';
      btn.style.background = (i === idx) ? '#eef2ff' : '';
    }});
    submitBtn.disabled = false;
  }}

  submitBtn.onclick = () => {{
    if (selected === null || submitted) return;
    submitted = true;
    const q = QUESTIONS[current], correct = selected === q.correct_index;

    results.push({{
      question_id: q.id, is_correct: correct, selected_index: selected, 
      time_spent_s: parseFloat(((Date.now() - qStartMs) / 1000).toFixed(2))
    }});

    // Only highlight if the "Show validation" toggle is checked
    if (showValCb.checked) {{
      Array.from(optionsList.children).forEach((btn, i) => {{
        btn.style.outline = '';
        if (i === q.correct_index) {{ btn.style.background = '#d1fae5'; btn.style.borderColor = '#34d399'; }}
        else if (i === selected && !correct) {{ btn.style.background = '#fee2e2'; btn.style.borderColor = '#f87171'; }}
      }});
    }}

    if (showExpCb.checked) {{
      feedbackBox.classList.remove('hidden');
      feedbackBox.className = `mt-6 p-4 rounded-2xl border text-sm ${{correct ? 'bg-green-50 border-green-300 text-green-800' : 'bg-red-50 border-red-300 text-red-800'}}`;
      feedbackBox.innerHTML = `<strong>${{correct ? '✅ Correct!' : '❌ Incorrect.'}}</strong><br>${{escHtml(q.explanation)}}`;
    }}

    submitBtn.classList.add('hidden');
    nextBtn.classList.remove('hidden');
  }};

  nextBtn.onclick = () => {{
    questionCard.classList.add('fade-out');
    setTimeout(() => {{
      current++;
      if (current < QUESTIONS.length) renderQuestion(current);
      else showResults();
      questionCard.classList.remove('fade-out');
    }}, 220);
  }};

  function showResults() {{
    questionCard.classList.add('hidden');
    resultsScreen.classList.remove('hidden');
    const score = results.filter(r => r.is_correct).length;
    document.getElementById('score-text').textContent = SHOW_SCORE ? `Score: ${{score}} / ${{results.length}}` : '';
    if (UPLOAD_URL) {{
      usernamePanel.classList.remove('hidden');
      usernameInput.focus();
    }}
  }}

  function downloadResults() {{
    const blob = new Blob([JSON.stringify(results, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'quiz_results.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }}

  function uploadResults() {{
    if (!UPLOAD_URL) return Promise.reject(new Error('No upload URL configured'));
    const username = usernameInput.value.trim() || 'Anonymous';
    // Use no-cors mode for Google Apps Script endpoints that expect simple POSTs
    return fetch(UPLOAD_URL, {{
      method: 'POST',
      mode: 'no-cors',
      cache: 'no-cache',
      headers: {{ 'Content-Type': 'text/plain' }},
      body: JSON.stringify({{
        username: username,
        results: results,
      }})
    }});
  }}

  // Wire the download/upload button depending on configuration
  if (UPLOAD_URL) {{
    downloadBtn.textContent = '⬆ Upload Results (Google Sheets)';
    downloadBtn.onclick = () => {{
      if (!usernameInput.value.trim()) {{
        usernameInput.focus();
        usernameInput.placeholder = 'Please enter your name first';
        return;
      }}
      downloadBtn.disabled = true;
      downloadBtn.textContent = 'Uploading...';
      uploadResults()
        .then(() => {{ downloadBtn.textContent = 'Uploaded ✓'; }})
        .catch(err => {{ console.error('Upload error:', err); downloadBtn.textContent = 'Upload Failed'; }})
        .finally(() => {{ setTimeout(() => {{ downloadBtn.disabled = false; downloadBtn.textContent = '⬆ Upload Results (Google Sheets)'; }}, 2000); }});
    }};
  }} else {{
    downloadBtn.onclick = downloadResults;
  }}

  function escHtml(str) {{ return String(str).replace(/[&<>"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[m]); }}

  QUESTIONS.length ? renderQuestion(0) : qText.textContent = 'No questions found.';
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
      "--no_score",
      action="store_true",
      help="Do not display final score on the results screen.",
    )
    parser.add_argument(
        "--upload_url",
        type=str,
        default=QUIZ_UPLOAD_URL,
        help="Optional URL (e.g. Google Apps Script) to POST results to instead of downloading.",
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
    html     = build_html(
      records,
      json_path,
      args.image_dir.resolve(),
      show_score=not args.no_score,
      upload_url=args.upload_url,
    )

    out: Path = args.output.resolve() if args.output else json_path.parent / "index.html"
    out.write_text(html, encoding="utf-8")

    print(f"✓ Wrote interactive quiz ({len(records)} questions) → {out}")
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
