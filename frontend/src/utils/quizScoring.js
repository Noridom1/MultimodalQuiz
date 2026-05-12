/**
 * Client-side grading for multimodal quiz items (aligned with backend canonical types).
 */

const TYPE_ALIASES = {
  multiple_choice: "multiple_choice",
  "multiple-choice": "multiple_choice",
  mcq: "multiple_choice",
  true_false: "true_false",
  "true-false": "true_false",
  truefalse: "true_false",
  fill_in_blank: "fill_in_blank",
  "fill-in-blank": "fill_in_blank",
  fib: "fill_in_blank",
  matching: "matching",
  match: "matching",
};

export function normalizeQuestionType(question) {
  const raw = String(question?.question_type ?? "multiple_choice")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/-/g, "_");
  return TYPE_ALIASES[raw] || raw || "multiple_choice";
}

function normalizeTrueFalse(s) {
  const x = String(s ?? "").trim().toLowerCase();
  if (x === "true" || x === "t" || x === "yes" || x === "1") return "true";
  if (x === "false" || x === "f" || x === "no" || x === "0") return "false";
  return x;
}

function fibMatches(user, correct) {
  const u = String(user ?? "").trim();
  const c = String(correct ?? "").trim();
  if (!u || !c) return false;
  if (u === c) return true;
  return u.toLowerCase() === c.toLowerCase();
}

/** Build left index -> correct right index from matching_solution pairs */
export function expectedMatchingMap(metadata) {
  const sol = metadata?.matching_solution;
  if (!Array.isArray(sol)) return null;
  const map = {};
  for (const pair of sol) {
    if (!Array.isArray(pair) || pair.length < 2) continue;
    const li = Number(pair[0]);
    const ri = Number(pair[1]);
    if (!Number.isFinite(li) || !Number.isFinite(ri)) continue;
    map[li] = ri;
  }
  return map;
}

/**
 * @param {Record<string, unknown>} question - summary result item
 * @param {unknown} rawAnswer - string for mcq/tf/fib; { t: 'matching', p: number[] } for matching
 */
/** Whether the learner submitted enough input to count as answered */
export function isAnswerProvided(question, rawAnswer) {
  const qt = normalizeQuestionType(question);
  if (rawAnswer === undefined || rawAnswer === null) return false;
  if (qt === "matching") {
    if (typeof rawAnswer !== "object" || rawAnswer.t !== "matching" || !Array.isArray(rawAnswer.p)) return false;
    const md = question?.metadata && typeof question.metadata === "object" ? question.metadata : {};
    const left = md.matching_left;
    const n = Array.isArray(left) ? left.length : 0;
    return (
      n > 0 &&
      rawAnswer.p.length === n &&
      rawAnswer.p.every((x) => typeof x === "number" && Number.isFinite(x))
    );
  }
  return typeof rawAnswer === "string" && rawAnswer.trim().length > 0;
}

export function isAnswerCorrect(question, rawAnswer) {
  const qt = normalizeQuestionType(question);
  if (qt === "multiple_choice") {
    return String(rawAnswer ?? "").trim() === String(question?.correct_answer ?? "").trim();
  }
  if (qt === "true_false") {
    const ca = String(question?.correct_answer ?? "").trim();
    const ua = String(rawAnswer ?? "").trim();
    if (!ua || !ca) return false;
    return ua.toLowerCase() === ca.toLowerCase() || normalizeTrueFalse(ua) === normalizeTrueFalse(ca);
  }
  if (qt === "fill_in_blank") {
    return fibMatches(rawAnswer, question?.correct_answer);
  }
  if (qt === "matching") {
    const picks = rawAnswer && typeof rawAnswer === "object" && Array.isArray(rawAnswer.p) ? rawAnswer.p : null;
    if (!picks) return false;
    const md = question?.metadata && typeof question.metadata === "object" ? question.metadata : {};
    const expected = expectedMatchingMap(md);
    const left = md.matching_left;
    const n = Array.isArray(left) ? left.length : (expected ? Object.keys(expected).length : 0);
    if (!expected || n === 0) return false;
    if (picks.length !== n) return false;
    for (let i = 0; i < n; i++) {
      if (expected[i] !== picks[i]) return false;
    }
    return true;
  }
  return String(rawAnswer ?? "").trim() === String(question?.correct_answer ?? "").trim();
}
