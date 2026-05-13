import {
  BarChart2,
  Bookmark,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleDashed,
  CircleX,
  Download,
  FileText,
  GraduationCap,
  LoaderCircle,
  MoreVertical,
  Sparkles,
  X,
} from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { API_BASE } from "../../api";
import {
  expectedMatchingMap,
  isAnswerCorrect,
  isAnswerProvided,
  normalizeQuestionType,
} from "../../utils/quizScoring";

function joinUrl(base, path) {
  const safeBase = (base || "").replace(/\/+$/, "");
  const safePath = (path || "").replace(/^\/+/, "");
  return safeBase ? `${safeBase}/${safePath}` : `/${safePath}`;
}

function isAbsoluteHttpUrl(value) {
  return value.startsWith("http://") || value.startsWith("https://");
}

function decodeIfEncodedUrl(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function extractFilename(value) {
  if (!value) return "";
  const decoded = decodeIfEncodedUrl(String(value).trim());
  if (!decoded) return "";
  if (isAbsoluteHttpUrl(decoded)) {
    try {
      const parsed = new URL(decoded);
      return decodeIfEncodedUrl(parsed.pathname.split("/").pop() || "");
    } catch {
      return "";
    }
  }
  return decoded.replace(/\\/g, "/").split("/").pop() || "";
}

function resolveQuestionImageUrl(imageUrl, selectedRun) {
  const url = String(imageUrl || "").trim();
  if (!url) return "";
  if (isAbsoluteHttpUrl(url)) return url;

  const decoded = decodeIfEncodedUrl(url).trim();
  if (isAbsoluteHttpUrl(decoded)) return decoded;

  if (url.startsWith("/")) return `${API_BASE}${url}`;

  const normalizedRelative = url.replace(/\\/g, "/").replace(/^\.\/+/, "");
  const hasOutputsPrefix = normalizedRelative.startsWith("outputs/");
  const hasGenerationPrefix = normalizedRelative.startsWith("generation/");

  if (selectedRun?.run_id) {
    if (hasOutputsPrefix) {
      return `${API_BASE}/api/artifacts/${normalizedRelative}`;
    }
    if (hasGenerationPrefix) {
      return `${API_BASE}/api/artifacts/outputs/${selectedRun.run_id}/${normalizedRelative}`;
    }
    return `${API_BASE}/api/artifacts/outputs/${selectedRun.run_id}/generation/${normalizedRelative}`;
  }

  return joinUrl(API_BASE, normalizedRelative);
}

function buildQuestionImageCandidates(imageUrl, selectedRun) {
  const raw = String(imageUrl || "").trim();
  if (!raw) return [];
  const decoded = decodeIfEncodedUrl(raw).trim();
  const filename = extractFilename(raw);
  const localFallback = selectedRun?.run_id && filename
    ? `${API_BASE}/api/artifacts/outputs/${selectedRun.run_id}/generation/images/${filename}`
    : "";

  if (isAbsoluteHttpUrl(raw) || isAbsoluteHttpUrl(decoded)) {
    return [isAbsoluteHttpUrl(raw) ? raw : decoded, localFallback].filter(Boolean);
  }

  const normalized = raw.replace(/\\/g, "/").replace(/^\.\/+/, "");
  const candidates = [resolveQuestionImageUrl(raw, selectedRun)];

  if (selectedRun?.run_id && filename) {
    candidates.push(`${API_BASE}/api/artifacts/outputs/${selectedRun.run_id}/generation/images/${filename}`);
    candidates.push(`${API_BASE}/api/artifacts/outputs/${selectedRun.run_id}/${normalized}`);
  }

  const deduped = [];
  for (const item of candidates) {
    const value = String(item || "").trim();
    if (value && !deduped.includes(value)) deduped.push(value);
  }
  return deduped;
}

/** Local-only override: fixed image paths per question index for one run folder. */
const HARDCODED_RUN_GENERATION_IMAGES = {
  "20260512_173748_book-riscv-rev5-chap8_16673c": {
    1: "outputs/20260512_173748_book-riscv-rev5-chap8_16673c/generation/images/1.png",
    2: "outputs/20260512_173748_book-riscv-rev5-chap8_16673c/generation/images/2.jpg",
    3: "outputs/20260512_173748_book-riscv-rev5-chap8_16673c/generation/images/3.jpg",
    4: "outputs/20260512_173748_book-riscv-rev5-chap8_16673c/generation/images/4.png",
    5: "outputs/20260512_173748_book-riscv-rev5-chap8_16673c/generation/images/5.png",
  },
};

function hardcodedQuestionImageRef(runId, questionIndex1Based) {
  if (!runId || questionIndex1Based == null || Number.isNaN(Number(questionIndex1Based))) return "";
  const table = HARDCODED_RUN_GENERATION_IMAGES[runId];
  if (!table) return "";
  const n = Number(questionIndex1Based);
  return table[n] || "";
}

function formatQuizTitle(run) {
  const rawTitle = String(run?.summary?.title || run?.title || "Untitled quiz").trim();
  if (!rawTitle) return "Untitled quiz";
  return rawTitle
    .replace(/^run[\s:_-]*/i, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function formatDifficultyLabel(raw) {
  const s = String(raw ?? "medium").trim() || "medium";
  return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
}

function findFirstReviewQuestionIndex(quizResults, selectedAnswers) {
  for (let i = 0; i < quizResults.length; i++) {
    const item = quizResults[i];
    if (!isAnswerProvided(item, selectedAnswers[i])) return i;
    if (!isAnswerCorrect(item, selectedAnswers[i])) return i;
  }
  return 0;
}

function QuizScoreRing({ correct, total }) {
  const gradId = `quizRingGrad-${useId().replace(/:/g, "")}`;
  const r = 58;
  const stroke = 10;
  const c = 2 * Math.PI * r;
  const fraction = total > 0 ? correct / total : 0;
  const dash = c * fraction;
  const pct = total > 0 ? Math.round(fraction * 100) : 0;
  const size = (r + stroke) * 2;

  return (
    <div className="quiz-score-ring" aria-hidden>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="quiz-score-ring-svg">
        <defs>
          <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#c4b5fd" />
            <stop offset="45%" stopColor="#8b5cf6" />
            <stop offset="100%" stopColor="#5b21b6" />
          </linearGradient>
        </defs>
        <g transform={`translate(${size / 2} ${size / 2}) rotate(-90)`}>
          <circle className="quiz-score-ring-track" r={r} fill="none" strokeWidth={stroke} />
          <circle
            r={r}
            fill="none"
            stroke={`url(#${gradId})`}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={`${dash} ${c}`}
          />
        </g>
      </svg>
      <div className="quiz-score-ring-label">
        <strong>{correct}/{total}</strong>
        <span>{pct}%</span>
      </div>
    </div>
  );
}

function formatRunBlurb(run) {
  const concepts = run?.summary?.concepts;
  if (Array.isArray(concepts) && concepts.length > 0) {
    return concepts
      .slice(0, 3)
      .map((c) => String(c).trim())
      .filter(Boolean)
      .join(" · ");
  }
  return "Multimodal quiz with image-backed questions from your sources.";
}

function RunListRow({
  run,
  formatTitle,
  isMenuOpen,
  onToggleMenu,
  onSelectCompleted,
  onRequestExport,
  onRename,
  onDelete,
  onSaveToList,
}) {
  const completed = run.status === "completed";

  useEffect(() => {
    if (!isMenuOpen) return undefined;
    const onDocMouseDown = (event) => {
      const root = event.target.closest("[data-run-menu]");
      if (root && root.getAttribute("data-run-menu") === run.run_id) return;
      onToggleMenu(null);
    };
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [isMenuOpen, run.run_id, onToggleMenu]);

  return (
    <div
      className={`run-list-item${completed ? "" : " run-list-item--inactive"}${isMenuOpen ? " run-list-item--menu-open" : ""}`}
    >
      <button
        type="button"
        className="run-list-item-main"
        onClick={() => completed && onSelectCompleted(run.run_id)}
        disabled={!completed}
      >
        <div className="run-list-icon-tile" aria-hidden>
          <FileText size={18} strokeWidth={1.75} />
        </div>
        <div className="run-list-item-body">
          <strong>{formatTitle(run)}</strong>
          <div className="run-list-blurb-row">
            <span className="run-question-badge">
              {run.summary?.question_count || run.num_questions || 0} questions
            </span>
            <span className="run-list-desc">{formatRunBlurb(run)}</span>
          </div>
        </div>
        {!completed ? <CircleDashed size={16} className="run-list-status-icon" aria-hidden /> : null}
        {completed && !isMenuOpen ? (
          <ChevronRight size={20} className="run-list-chevron" aria-hidden />
        ) : null}
      </button>
      <div className="run-list-item-actions" data-run-menu={run.run_id}>
        <button
          type="button"
          className="run-list-menu-trigger ghost-inline compact"
          aria-expanded={isMenuOpen}
          aria-haspopup="true"
          aria-label="Quiz actions"
          onClick={(event) => {
            event.stopPropagation();
            onToggleMenu(isMenuOpen ? null : run.run_id);
          }}
        >
          <MoreVertical size={18} />
        </button>
        {isMenuOpen ? (
          <div className="run-list-menu" role="menu">
            {completed && onSaveToList ? (
              <button
                type="button"
                className="run-list-menu-item"
                role="menuitem"
                onClick={() => {
                  onSaveToList(run.run_id);
                  onToggleMenu(null);
                }}
              >
                Save to list…
              </button>
            ) : null}
            {completed ? (
              <button
                type="button"
                className="run-list-menu-item"
                role="menuitem"
                onClick={() => {
                  onRequestExport(run.run_id);
                  onToggleMenu(null);
                }}
              >
                Export…
              </button>
            ) : null}
            <button
              type="button"
              className="run-list-menu-item"
              role="menuitem"
              onClick={() => {
                const next = window.prompt("Quiz name", formatTitle(run));
                if (next != null && next.trim()) {
                  onRename(run.run_id, next.trim());
                }
                onToggleMenu(null);
              }}
            >
              Rename
            </button>
            <button
              type="button"
              className="run-list-menu-item run-list-menu-item--danger"
              role="menuitem"
              onClick={() => {
                if (window.confirm("Delete this quiz? This cannot be undone.")) {
                  onDelete(run.run_id);
                }
                onToggleMenu(null);
              }}
            >
              Delete
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function QuizPanel({
  activeQuestionIndex,
  canGenerate,
  notebookId,
  pendingRun,
  resultStats,
  resultsOpen,
  selectedAnswers,
  selectedRun,
  totalQuestions,
  workspaceRuns,
  setActiveQuestionIndex,
  setResultsOpen,
  setSelectedAnswers,
  setSelectedRunId,
  onDeleteRun,
  onRequestExport,
  onExitQuiz,
  onOpenQuizBuilder,
  onRedoQuiz,
  onRenameRun,
  onRequestSaveToList,
}) {
  const [openMenuRunId, setOpenMenuRunId] = useState(null);
  const [quizDetailMenuOpen, setQuizDetailMenuOpen] = useState(false);
  const quizDetailMenuRef = useRef(null);
  /** Draft inputs before Submit for fill-in-blank / matching (per question index). */
  const [fibDraftByIndex, setFibDraftByIndex] = useState({});
  const [matchingDraftByIndex, setMatchingDraftByIndex] = useState({});
  const quizResults = selectedRun?.summary?.results || [];
  const currentQuestion = quizResults[activeQuestionIndex] || null;
  const questionIndex1Based =
    currentQuestion?.index != null && String(currentQuestion.index).trim() !== ""
      ? Number(currentQuestion.index)
      : activeQuestionIndex + 1;
  const hardcodedImageRef = hardcodedQuestionImageRef(selectedRun?.run_id, questionIndex1Based);
  const effectiveQuestionImageUrl = hardcodedImageRef || currentQuestion?.image_url || "";
  const imageCandidates = buildQuestionImageCandidates(effectiveQuestionImageUrl, selectedRun);
  const showQuestionImage = imageCandidates.length > 0;
  const hasCompletedRuns = workspaceRuns.some((run) => run.status === "completed");
  const currentChoice = selectedAnswers[activeQuestionIndex];
  const currentType = normalizeQuestionType(currentQuestion || {});
  const fibSubmitted =
    currentType === "fill_in_blank" && isAnswerProvided(currentQuestion, currentChoice);
  const matchingSubmitted =
    currentType === "matching" && isAnswerProvided(currentQuestion, currentChoice);
  const hasAnswer =
    currentType === "fill_in_blank"
      ? fibSubmitted
      : currentType === "matching"
        ? matchingSubmitted
        : isAnswerProvided(currentQuestion, currentChoice);
  const choiceLocked = hasAnswer;

  const mdMatching =
    currentQuestion?.metadata && typeof currentQuestion.metadata === "object"
      ? currentQuestion.metadata
      : {};
  const matchingLeft = Array.isArray(mdMatching.matching_left) ? mdMatching.matching_left : [];
  const matchingRight = Array.isArray(mdMatching.matching_right) ? mdMatching.matching_right : [];

  useEffect(() => {
    if (!quizDetailMenuOpen) return undefined;
    const onDocMouseDown = (event) => {
      if (quizDetailMenuRef.current?.contains(event.target)) return;
      setQuizDetailMenuOpen(false);
    };
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [quizDetailMenuOpen]);

  useEffect(() => {
    if (currentType !== "matching" || matchingSubmitted || matchingLeft.length === 0) return;
    setMatchingDraftByIndex((prev) => {
      const existing = prev[activeQuestionIndex];
      if (existing?.p?.length === matchingLeft.length) return prev;
      return {
        ...prev,
        [activeQuestionIndex]: { t: "matching", p: Array(matchingLeft.length).fill(-1) },
      };
    });
  }, [activeQuestionIndex, currentType, matchingLeft.length, matchingSubmitted]);

  function submitFib() {
    const raw = fibDraftByIndex[activeQuestionIndex] ?? "";
    const trimmed = raw.trim();
    if (!trimmed) return;
    setSelectedAnswers((prev) => ({ ...prev, [activeQuestionIndex]: trimmed }));
  }

  function submitMatching() {
    const draft = matchingDraftByIndex[activeQuestionIndex];
    const picks = draft?.p;
    const payload = picks ? { t: "matching", p: picks } : null;
    if (!payload || !isAnswerProvided(currentQuestion, payload)) return;
    setSelectedAnswers((prev) => ({ ...prev, [activeQuestionIndex]: payload }));
  }

  const fibInputValue = fibSubmitted
    ? String(currentChoice ?? "").trim()
    : fibDraftByIndex[activeQuestionIndex] ?? "";
  const fibCanSubmit = fibInputValue.trim().length > 0;

  const matchingPicks =
    matchingSubmitted && currentChoice?.t === "matching" && Array.isArray(currentChoice.p)
      ? currentChoice.p
      : matchingDraftByIndex[activeQuestionIndex]?.p ??
        Array.from({ length: matchingLeft.length }, () => -1);
  const matchingDraftPayload =
    matchingPicks.length === matchingLeft.length ? { t: "matching", p: matchingPicks } : null;
  const matchingCanSubmit =
    Boolean(matchingDraftPayload) && isAnswerProvided(currentQuestion, matchingDraftPayload);

  const artifactPaths = selectedRun?.summary?.artifact_paths ?? {};

  return (
    <section className="panel studio-panel">
      <div
        className={`panel-header panel-header--notebook quiz-panel-head${selectedRun ? " quiz-panel-head--detail" : ""}`}
      >
        <div className="panel-header-lead">
          <div className="panel-header-icon-wrap" aria-hidden>
            <GraduationCap size={19} strokeWidth={1.75} />
          </div>
          <div className="panel-header-text">
            <h2>Quizzes</h2>
            <p className="panel-subtitle">
              {selectedRun
                ? "Test your knowledge from your sources."
                : "Generate quizzes from your sources."}
            </p>
          </div>
        </div>
        {selectedRun ? (
          <button
            type="button"
            className="ghost-pill compact quiz-panel-exit-quiz"
            onClick={() => onExitQuiz()}
          >
            <X size={15} aria-hidden />
            Exit quiz
          </button>
        ) : (
          <button
            type="button"
            className="primary-pill compact quiz-panel-generate-header"
            onClick={() => onOpenQuizBuilder()}
            disabled={!canGenerate || pendingRun}
          >
            {pendingRun ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}
            Generate quiz
          </button>
        )}
      </div>
      {!selectedRun ? (
        <div className="quiz-panel-list">
          <div className="quiz-run-list-scroll">
            <div className="quiz-run-list">
              {workspaceRuns.map((run) => (
                <RunListRow
                  key={run.run_id}
                  run={run}
                  formatTitle={formatQuizTitle}
                  isMenuOpen={openMenuRunId === run.run_id}
                  onToggleMenu={setOpenMenuRunId}
                  onSelectCompleted={(runId) => setSelectedRunId(runId)}
                  onRequestExport={(runId) => onRequestExport(runId)}
                  onRename={(runId, title) => {
                    void onRenameRun(runId, title).catch((err) =>
                      alert(err instanceof Error ? err.message : "Rename failed"),
                    );
                  }}
                  onDelete={(runId) => {
                    void onDeleteRun(runId).catch((err) =>
                      alert(err instanceof Error ? err.message : "Delete failed"),
                    );
                  }}
                  onSaveToList={onRequestSaveToList}
                />
              ))}
              {workspaceRuns.length === 0 ? (
                <div className="empty-panel studio-empty">
                  <p>Finished quizzes will appear here after the first run.</p>
                </div>
              ) : null}
              {hasCompletedRuns && workspaceRuns.length > 0 ? (
                <p className="quiz-list-hint">Tap a completed quiz to open it and start answering.</p>
              ) : null}
            </div>
          </div>
        </div>
      ) : (
        <div className="quiz-panel-content quiz-panel-content--detail">
          <div className="quiz-player">
            <div className="quiz-player-toolbar">
              <button
                type="button"
                className="quiz-toolbar-back"
                onClick={() => onExitQuiz()}
                aria-label="Back to quizzes"
              >
                <ChevronLeft size={18} strokeWidth={2} aria-hidden />
              </button>
              <h3 className="quiz-toolbar-title">{formatQuizTitle(selectedRun)}</h3>
              <div className="quiz-toolbar-actions" ref={quizDetailMenuRef}>
                {onRequestSaveToList && notebookId ? (
                  <button
                    type="button"
                    className="quiz-toolbar-text-btn"
                    onClick={() => onRequestSaveToList(selectedRun.run_id)}
                  >
                    <Bookmark size={15} strokeWidth={2} aria-hidden />
                    Save to list
                  </button>
                ) : null}
                <div className="quiz-toolbar-menu-wrap">
                  <button
                    type="button"
                    className="quiz-toolbar-more"
                    aria-expanded={quizDetailMenuOpen}
                    aria-haspopup="true"
                    aria-label="Quiz options"
                    onClick={() => setQuizDetailMenuOpen((open) => !open)}
                  >
                    <MoreVertical size={18} strokeWidth={2} aria-hidden />
                  </button>
                  {quizDetailMenuOpen ? (
                    <div className="run-list-menu quiz-toolbar-dropdown" role="menu">
                      <button
                        type="button"
                        className="run-list-menu-item"
                        role="menuitem"
                        onClick={() => {
                          onRequestExport(selectedRun.run_id);
                          setQuizDetailMenuOpen(false);
                        }}
                      >
                        <Download size={16} aria-hidden />
                        Export…
                      </button>
                      {artifactPaths.quiz_package ? (
                        <a
                          href={artifactPaths.quiz_package}
                          className="run-list-menu-item"
                          role="menuitem"
                          target="_blank"
                          rel="noreferrer"
                          onClick={() => setQuizDetailMenuOpen(false)}
                        >
                          Quiz package
                        </a>
                      ) : null}
                      {artifactPaths.graph_html ? (
                        <a
                          href={artifactPaths.graph_html}
                          className="run-list-menu-item"
                          role="menuitem"
                          target="_blank"
                          rel="noreferrer"
                          onClick={() => setQuizDetailMenuOpen(false)}
                        >
                          Graph
                        </a>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              </div>
            </div>

            {resultsOpen ? (
              <div className="quiz-player-scroll">
                <div className="quiz-results-card quiz-results-card--summary">
                  <header className="quiz-results-celebrate">
                    <span className="quiz-results-badge">Quiz complete</span>
                    <h2 className="quiz-results-hero">You did it!</h2>
                  </header>

                  <div className="quiz-results-ring-wrap">
                    <QuizScoreRing correct={resultStats.correct} total={resultStats.total} />
                  </div>

                  <div className="quiz-results-stats" role="list">
                    <div className="quiz-results-stat-card quiz-results-stat-card--correct" role="listitem">
                      <span className="quiz-results-stat-label">Correct</span>
                      <span className="quiz-results-stat-value" aria-label={`${resultStats.correct} correct`}>
                        {resultStats.correct}
                      </span>
                    </div>
                    <div className="quiz-results-stat-card quiz-results-stat-card--wrong" role="listitem">
                      <span className="quiz-results-stat-label">Wrong</span>
                      <span className="quiz-results-stat-value" aria-label={`${resultStats.wrong} wrong`}>
                        {resultStats.wrong}
                      </span>
                    </div>
                    <div className="quiz-results-stat-card quiz-results-stat-card--unanswered" role="listitem">
                      <span className="quiz-results-stat-label">Unanswered</span>
                      <span className="quiz-results-stat-value" aria-label={`${resultStats.unanswered} unanswered`}>
                        {resultStats.unanswered}
                      </span>
                    </div>
                  </div>

                  <p className="quiz-results-sub">
                    Tap a question number to review, or use the actions below.
                  </p>
                  <div className="quiz-question-nav compact-nav quiz-results-question-nav">
                    {quizResults.map((item, index) => {
                      let dotClass = "question-nav-dot";
                      if (!isAnswerProvided(item, selectedAnswers[index])) {
                        dotClass += " skipped";
                      } else if (isAnswerCorrect(item, selectedAnswers[index])) {
                        dotClass += " correct";
                      } else {
                        dotClass += " incorrect";
                      }
                      return (
                        <button
                          type="button"
                          key={item.index}
                          className={dotClass}
                          onClick={() => {
                            setResultsOpen(false);
                            setActiveQuestionIndex(index);
                          }}
                        >
                          {index + 1}
                        </button>
                      );
                    })}
                  </div>
                  <div className="quiz-results-actions">
                    <button
                      type="button"
                      className="quiz-results-btn-outline"
                      onClick={() => {
                        const idx = findFirstReviewQuestionIndex(quizResults, selectedAnswers);
                        setResultsOpen(false);
                        setActiveQuestionIndex(idx);
                      }}
                    >
                      Review answers
                    </button>
                    <button type="button" className="primary-pill" onClick={() => onRedoQuiz()}>
                      Retry quiz
                    </button>
                    <button
                      type="button"
                      className="ghost-pill quiz-results-btn-ghost"
                      onClick={() => onRequestExport(selectedRun.run_id)}
                    >
                      <Download size={16} />
                      Export…
                    </button>
                  </div>
                </div>
              </div>
            ) : currentQuestion ? (
              <>
                <div className="quiz-player-scroll">
                  <article className="quiz-question-card">
                    <div className="quiz-question-card-head">
                      <div className="quiz-progress-stack">
                        <span className="quiz-progress-label quiz-progress-label--strong">
                          Question {activeQuestionIndex + 1} of {totalQuestions}
                        </span>
                        <div
                          className="quiz-progress-bar"
                          role="progressbar"
                          aria-valuemin={1}
                          aria-valuenow={Math.min(activeQuestionIndex + 1, Math.max(totalQuestions, 1))}
                          aria-valuemax={Math.max(totalQuestions, 1)}
                          aria-label={`Question ${activeQuestionIndex + 1} of ${totalQuestions}`}
                        >
                          <div
                            className="quiz-progress-bar-fill"
                            style={{
                              width: `${totalQuestions > 0 ? ((activeQuestionIndex + 1) / totalQuestions) * 100 : 0}%`,
                            }}
                          />
                        </div>
                      </div>
                      <div className="quiz-difficulty-badge">
                        <BarChart2 size={14} strokeWidth={2} aria-hidden />
                        {formatDifficultyLabel(currentQuestion.difficulty)}
                      </div>
                    </div>
                    {showQuestionImage ? (
                      <div className="quiz-player-image quiz-player-image--card">
                        <img
                          key={currentQuestion.index ?? activeQuestionIndex}
                          src={imageCandidates[0] || ""}
                          onError={(event) => {
                            const imageEl = event.currentTarget;
                            const currentIndex = Number(imageEl.dataset.fallbackIndex || 0);
                            const nextIndex = currentIndex + 1;
                            if (nextIndex >= imageCandidates.length) return;
                            imageEl.dataset.fallbackIndex = String(nextIndex);
                            imageEl.src = imageCandidates[nextIndex];
                          }}
                          alt={`Question ${activeQuestionIndex + 1} illustration`}
                          className="quiz-player-image-element"
                        />
                      </div>
                    ) : null}
                    <h4 className="quiz-question-title">{currentQuestion.question_text}</h4>
                    {currentQuestion.target_concept ? (
                      <p className="quiz-question-concept">{String(currentQuestion.target_concept)}</p>
                    ) : null}
                    {currentType === "fill_in_blank" ? (
                      <div className="quiz-options quiz-options--with-submit">
                        <input
                          type="text"
                          className="quiz-fib-input"
                          placeholder="Type your answer"
                          value={fibInputValue}
                          disabled={fibSubmitted}
                          onChange={(event) => {
                            if (fibSubmitted) return;
                            setFibDraftByIndex((prev) => ({
                              ...prev,
                              [activeQuestionIndex]: event.target.value,
                            }));
                          }}
                        />
                        {!fibSubmitted ? (
                          <button
                            type="button"
                            className="primary-pill quiz-submit-answer"
                            disabled={!fibCanSubmit}
                            onClick={submitFib}
                          >
                            Submit answer
                          </button>
                        ) : null}
                        {fibSubmitted ? (
                          <div className="quiz-option-feedback">
                            <div
                              className={`quiz-option-verdict ${isAnswerCorrect(currentQuestion, currentChoice) ? "correct" : "incorrect"}`}
                            >
                              {isAnswerCorrect(currentQuestion, currentChoice) ? "Correct!" : "Not quite right!"}
                            </div>
                            <p className="quiz-option-explanation">
                              Correct answer: {String(currentQuestion.correct_answer || "(none)")}
                            </p>
                            {currentQuestion.explanation ? (
                              <p className="quiz-option-explanation">{currentQuestion.explanation}</p>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    ) : currentType === "matching" ? (
                      <div className="quiz-options quiz-options--with-submit">
                        {matchingLeft.map((leftItem, li) => (
                          <label key={`${li}-${String(leftItem)}`} className="quiz-matching-row">
                            <span className="quiz-matching-left">{String(leftItem)}</span>
                            <select
                              className="quiz-matching-select"
                              disabled={matchingSubmitted}
                              value={
                                Number.isFinite(matchingPicks[li]) && matchingPicks[li] >= 0
                                  ? String(matchingPicks[li])
                                  : ""
                              }
                              onChange={(event) => {
                                if (matchingSubmitted) return;
                                const ri = Number.parseInt(event.target.value, 10);
                                setMatchingDraftByIndex((prev) => {
                                  const raw = prev[activeQuestionIndex];
                                  const baseP =
                                    raw?.p?.length === matchingLeft.length
                                      ? [...raw.p]
                                      : Array.from({ length: matchingLeft.length }, () => -1);
                                  baseP[li] = Number.isFinite(ri) ? ri : -1;
                                  return {
                                    ...prev,
                                    [activeQuestionIndex]: { t: "matching", p: baseP },
                                  };
                                });
                              }}
                            >
                              <option value="">Select match</option>
                              {matchingRight.map((rightItem, ri) => (
                                <option key={`${ri}-${String(rightItem)}`} value={ri}>
                                  {String(rightItem)}
                                </option>
                              ))}
                            </select>
                          </label>
                        ))}
                        {!matchingSubmitted ? (
                          <button
                            type="button"
                            className="primary-pill quiz-submit-answer"
                            disabled={!matchingCanSubmit}
                            onClick={submitMatching}
                          >
                            Submit matches
                          </button>
                        ) : null}
                        {matchingSubmitted ? (
                          <div className="quiz-option-feedback">
                            <div
                              className={`quiz-option-verdict ${isAnswerCorrect(currentQuestion, currentChoice) ? "correct" : "incorrect"}`}
                            >
                              {isAnswerCorrect(currentQuestion, currentChoice) ? "Correct!" : "Not quite right!"}
                            </div>
                            {!isAnswerCorrect(currentQuestion, currentChoice) ? (
                              <p className="quiz-option-explanation">
                                Correct mapping:{" "}
                                {(() => {
                                  const expected = expectedMatchingMap(mdMatching) || {};
                                  return matchingLeft
                                    .map((item, leftIdx) => {
                                      const rightIdx = expected[leftIdx];
                                      const rightLabel = Number.isInteger(rightIdx)
                                        ? matchingRight[rightIdx]
                                        : null;
                                      return `${String(item)} -> ${String(rightLabel ?? "?")}`;
                                    })
                                    .join("; ");
                                })()}
                              </p>
                            ) : null}
                            {currentQuestion.explanation ? (
                              <p className="quiz-option-explanation">{currentQuestion.explanation}</p>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    ) : (
                      <div className="quiz-options">
                        {currentQuestion.options.map((option, optionIndex) => {
                          const isSelected = currentChoice === option;
                          const showFeedback = isSelected && hasAnswer;
                          const isCorrectChoice = option === currentQuestion.correct_answer;
                          const letter =
                            optionIndex < 26 ? String.fromCharCode(65 + optionIndex) : String(optionIndex + 1);
                          return (
                            <button
                              type="button"
                              key={option}
                              className={`quiz-option ${isSelected ? "selected" : ""} ${showFeedback && isCorrectChoice ? "quiz-option--correct" : ""
                                } ${showFeedback && !isCorrectChoice ? "quiz-option--incorrect" : ""}`}
                              disabled={choiceLocked}
                              onClick={() =>
                                setSelectedAnswers((current) => {
                                  if (Object.prototype.hasOwnProperty.call(current, activeQuestionIndex)) {
                                    return current;
                                  }
                                  return { ...current, [activeQuestionIndex]: option };
                                })
                              }
                            >
                              <span className="quiz-option-row">
                                <span className="quiz-option-letter" aria-hidden>
                                  {letter}
                                </span>
                                <span className="quiz-option-text">{option}</span>
                              </span>
                              {showFeedback ? (
                                <div className="quiz-option-feedback">
                                  <div className={`quiz-option-verdict ${isCorrectChoice ? "correct" : "incorrect"}`}>
                                    {isCorrectChoice ? (
                                      <>
                                        <CheckCircle2 size={18} aria-hidden />
                                        <span>Correct!</span>
                                      </>
                                    ) : (
                                      <>
                                        <CircleX size={18} aria-hidden />
                                        <span>Not quite right!</span>
                                      </>
                                    )}
                                  </div>
                                  {currentQuestion.explanation ? (
                                    <p className="quiz-option-explanation">{currentQuestion.explanation}</p>
                                  ) : null}
                                </div>
                              ) : null}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </article>
                </div>

                <div className="quiz-player-actions quiz-player-actions--dock">
                  <button
                    type="button"
                    className="ghost-pill quiz-nav-prev"
                    onClick={() => setActiveQuestionIndex((current) => Math.max(0, current - 1))}
                    disabled={activeQuestionIndex === 0}
                  >
                    <ChevronLeft size={18} aria-hidden />
                    Previous
                  </button>
                  {activeQuestionIndex === totalQuestions - 1 ? (
                    <button
                      type="button"
                      className="primary-pill quiz-nav-next"
                      onClick={() => setResultsOpen(true)}
                      disabled={totalQuestions === 0}
                    >
                      View results
                      <ChevronRight size={18} aria-hidden />
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="primary-pill quiz-nav-next"
                      onClick={() =>
                        setActiveQuestionIndex((current) => Math.min(totalQuestions - 1, current + 1))
                      }
                    >
                      Next
                      <ChevronRight size={18} aria-hidden />
                    </button>
                  )}
                </div>
              </>
            ) : null}
          </div>
        </div>
      )}
    </section>
  );
}

export default QuizPanel;
