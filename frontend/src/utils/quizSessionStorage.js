const PREFIX = "multimodalQuiz.quizSession.v1";

function storageKey(notebookId, runId) {
  return `${PREFIX}.${notebookId}.${runId}`;
}

function clampIndex(value, maxIndex) {
  const n = Number.parseInt(String(value), 10);
  if (Number.isNaN(n)) return 0;
  return Math.min(Math.max(0, n), Math.max(0, maxIndex));
}

/**
 * @param {string} notebookId
 * @param {string} runId
 * @param {number} questionCount
 * @returns {{ selectedAnswers: Record<number, string|object>, resultsOpen: boolean, activeQuestionIndex: number } | null}
 */
export function loadQuizSession(notebookId, runId, questionCount) {
  if (!notebookId || !runId || questionCount <= 0) return null;
  try {
    const raw = localStorage.getItem(storageKey(notebookId, runId));
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (data.questionCount !== questionCount) return null;
    const rawAnswers = data.selectedAnswers && typeof data.selectedAnswers === "object" ? data.selectedAnswers : {};
    const selectedAnswers = {};
    for (const [key, val] of Object.entries(rawAnswers)) {
      const idx = Number.parseInt(key, 10);
      if (Number.isNaN(idx) || idx < 0 || idx >= questionCount) continue;
      if (typeof val === "string") {
        if (val.length > 0) selectedAnswers[idx] = val;
      } else if (typeof val === "object" && val !== null && val.t === "matching" && Array.isArray(val.p)) {
        selectedAnswers[idx] = val;
      }
    }
    return {
      selectedAnswers,
      resultsOpen: Boolean(data.resultsOpen),
      activeQuestionIndex: clampIndex(data.activeQuestionIndex, questionCount - 1),
    };
  } catch {
    return null;
  }
}

/**
 * @param {string} notebookId
 * @param {string} runId
 * @param {{ selectedAnswers: Record<number, unknown>, resultsOpen: boolean, activeQuestionIndex: number, questionCount: number }} payload
 */
export function saveQuizSession(notebookId, runId, payload) {
  if (!notebookId || !runId || !payload.questionCount) return;
  try {
    localStorage.setItem(
      storageKey(notebookId, runId),
      JSON.stringify({
        questionCount: payload.questionCount,
        selectedAnswers: payload.selectedAnswers || {},
        resultsOpen: Boolean(payload.resultsOpen),
        activeQuestionIndex: payload.activeQuestionIndex ?? 0,
      }),
    );
  } catch {
    /* quota / private mode */
  }
}
