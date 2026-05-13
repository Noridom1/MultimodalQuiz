import { useEffect, useLayoutEffect, useRef, useState, useTransition } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Pencil } from "lucide-react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api";
import LoadingPanel from "../components/common/LoadingPanel";
import VisionQBrandLink from "../components/common/VisionQBrandLink";
import UserAccountMenu from "../components/common/UserAccountMenu";
import ExportOptionsModal from "../components/notebook/ExportOptionsModal";
import QuizBuilderModal from "../components/notebook/QuizBuilderModal";
import QuizPanel from "../components/notebook/QuizPanel";
import SaveToListModal from "../components/notebook/SaveToListModal";
import SourcePanel from "../components/notebook/SourcePanel";
import { useAuth } from "../context/AuthContext";
import { useWorkspaceColumnResize } from "../hooks/useWorkspaceColumnResize";
import {
  buildEqualDistribution,
  QUIZ_QUESTION_TYPE_LABELS,
} from "../utils/quizFormatPresets";
import { isAnswerCorrect, isAnswerProvided } from "../utils/quizScoring";
import { loadQuizSession, saveQuizSession } from "../utils/quizSessionStorage";

function NotebookPage() {
  const { notebookId } = useParams();
  const [searchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const notebookQueryKey = ["notebook", notebookId, user?.id ?? "anon"];

  const workspaceQuery = useQuery({
    queryKey: notebookQueryKey,
    queryFn: () => api.getNotebook(notebookId),
    enabled: Boolean(notebookId),
  });

  const workspace = workspaceQuery.data;
  const showNotebookLoading = Boolean(notebookId) && (workspaceQuery.isPending || !workspace);
  const [quizBuilderOpen, setQuizBuilderOpen] = useState(false);
  const [saveListRunId, setSaveListRunId] = useState("");
  const [exportRunId, setExportRunId] = useState("");
  const [exportPending, setExportPending] = useState(false);
  const [exportPrefs, setExportPrefs] = useState({ format: "zip", dataFormat: "json" });
  const [selectedSourceId, setSelectedSourceId] = useState("");
  const [questions, setQuestions] = useState(10);
  const [selectedQuestionTypes, setSelectedQuestionTypes] = useState(["multiple_choice"]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [activeQuestionIndex, setActiveQuestionIndex] = useState(0);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [resultsOpen, setResultsOpen] = useState(false);
  const [pendingRun, startRunTransition] = useTransition();
  const [pendingUpload, startUploadTransition] = useTransition();
  const [notebookTitleDraft, setNotebookTitleDraft] = useState("");
  const workspaceRef = useRef(null);
  const { widths: columnWidths, startDrag: startColumnDrag } = useWorkspaceColumnResize();

  useEffect(() => {
    if (!workspace?.notebook) return;
    setNotebookTitleDraft(workspace.notebook.title ?? "");
  }, [workspace?.notebook?.title]);

  useEffect(() => {
    if (!workspace?.sources?.length) {
      setSelectedSourceId("");
      return;
    }
    setSelectedSourceId((current) => {
      if (current && workspace.sources.some((source) => source.id === current)) {
        return current;
      }
      return workspace.sources[0].id;
    });
  }, [workspace?.sources]);

  const runFromUrl = searchParams.get("run") || "";

  useEffect(() => {
    const completedRuns = (workspace?.runs || []).filter((run) => run.status === "completed");
    if (!completedRuns.length) {
      setSelectedRunId("");
      return;
    }
    setSelectedRunId((current) => {
      if (runFromUrl && completedRuns.some((run) => run.run_id === runFromUrl)) {
        return runFromUrl;
      }
      if (current && completedRuns.some((run) => run.run_id === current)) {
        return current;
      }
      return "";
    });
  }, [workspace?.runs, runFromUrl]);

  const selectedRunForSession =
    workspace?.runs?.filter((run) => run.status === "completed").find((run) => run.run_id === selectedRunId) || null;
  const sessionQuestionCount = selectedRunForSession?.summary?.results?.length ?? 0;

  useLayoutEffect(() => {
    if (!notebookId || !selectedRunId || sessionQuestionCount === 0) {
      return;
    }
    const loaded = loadQuizSession(notebookId, selectedRunId, sessionQuestionCount);
    if (loaded) {
      setSelectedAnswers(loaded.selectedAnswers);
      setResultsOpen(loaded.resultsOpen);
      setActiveQuestionIndex(loaded.activeQuestionIndex);
    } else {
      setSelectedAnswers({});
      setResultsOpen(false);
      setActiveQuestionIndex(0);
    }
  }, [notebookId, selectedRunId, sessionQuestionCount]);

  useEffect(() => {
    if (!notebookId || !selectedRunId || sessionQuestionCount === 0) {
      return;
    }
    saveQuizSession(notebookId, selectedRunId, {
      selectedAnswers,
      resultsOpen,
      activeQuestionIndex,
      questionCount: sessionQuestionCount,
    });
  }, [notebookId, selectedRunId, sessionQuestionCount, selectedAnswers, resultsOpen, activeQuestionIndex]);

  if (showNotebookLoading) {
    return (
      <div className="page-shell notebook-page">
        <LoadingPanel label="Loading workspace..." />
      </div>
    );
  }

  const primarySource = workspace.sources[0];
  const completedRuns = workspace.runs.filter((run) => run.status === "completed");
  const selectedRun =
    completedRuns.find((run) => run.run_id === selectedRunId) || null;
  const quizResults = selectedRun?.summary?.results || [];
  const totalQuestions = quizResults.length;

  const score = quizResults.reduce(
    (total, item, index) => total + (isAnswerCorrect(item, selectedAnswers[index]) ? 1 : 0),
    0,
  );

  const answeredCount = quizResults.reduce(
    (total, item, index) => total + (isAnswerProvided(item, selectedAnswers[index]) ? 1 : 0),
    0,
  );

  const wrongCount = quizResults.reduce((total, item, index) => {
    const answer = selectedAnswers[index];
    if (!isAnswerProvided(item, answer)) return total;
    return total + (isAnswerCorrect(item, answer) ? 0 : 1);
  }, 0);

  const unansweredCount = Math.max(0, totalQuestions - answeredCount);

  const resultStats = { correct: score, wrong: wrongCount, unanswered: unansweredCount, total: totalQuestions };

  function flushQuizSessionToStorage() {
    if (!notebookId || !selectedRunId || !selectedRun) {
      return;
    }
    const n = selectedRun.summary?.results?.length ?? 0;
    if (n <= 0) return;
    saveQuizSession(notebookId, selectedRunId, {
      selectedAnswers,
      resultsOpen,
      activeQuestionIndex,
      questionCount: n,
    });
  }

  function flushAndExitQuiz() {
    flushQuizSessionToStorage();
    setSelectedRunId("");
  }

  function handleRedoQuiz() {
    setSelectedAnswers({});
    setResultsOpen(false);
    setActiveQuestionIndex(0);
    if (notebookId && selectedRunId && selectedRun) {
      const n = selectedRun.summary?.results?.length ?? 0;
      if (n > 0) {
        saveQuizSession(notebookId, selectedRunId, {
          selectedAnswers: {},
          resultsOpen: false,
          activeQuestionIndex: 0,
          questionCount: n,
        });
      }
    }
  }

  function refreshWorkspace() {
    if (!notebookId) {
      return Promise.resolve(null);
    }
    return queryClient.fetchQuery({
      queryKey: notebookQueryKey,
      queryFn: () => api.getNotebook(notebookId),
    });
  }

  function handleUploadFile(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    startUploadTransition(async () => {
      await api.uploadSource(notebookId, { file, title: file.name.replace(/\.[^/.]+$/, "") });
      refreshWorkspace();
      event.target.value = "";
    });
  }

  function handleGenerateQuiz() {
    if (!selectedSourceId) {
      return;
    }
    startRunTransition(async () => {
      const question_format_distribution = buildEqualDistribution(selectedQuestionTypes);
      const createdRun = await api.generateQuiz(notebookId, {
        source_id: selectedSourceId,
        num_questions: questions,
        mock_image: false,
        mock_question: false,
        question_format_distribution,
      });
      const nextWorkspace = await refreshWorkspace();
      setQuizBuilderOpen(false);
      setSelectedRunId(createdRun.run_id || nextWorkspace?.latest_run?.run_id || "");
    });
  }

  async function handleNotebookTitleCommit() {
    if (!notebookId || !workspace) return;
    const trimmed = notebookTitleDraft.trim();
    if (!trimmed) {
      setNotebookTitleDraft(workspace.notebook.title);
      return;
    }
    if (trimmed === workspace.notebook.title) return;
    try {
      const updated = await api.patchNotebook(notebookId, { title: trimmed });
      queryClient.setQueryData(notebookQueryKey, (old) =>
        old ? { ...old, notebook: { ...old.notebook, ...updated } } : old,
      );
    } catch (err) {
      setNotebookTitleDraft(workspace.notebook.title);
      alert(err instanceof Error ? err.message : "Could not save title");
    }
  }

  async function handleRenameRun(runId, title) {
    if (!notebookId) return;
    await api.patchRun(notebookId, runId, { title });
    await refreshWorkspace();
  }

  async function handleDeleteRun(runId) {
    if (!notebookId) return;
    if (selectedRunId === runId) {
      setSelectedRunId("");
      setSelectedAnswers({});
      setResultsOpen(false);
      setActiveQuestionIndex(0);
    }
    await api.deleteRun(notebookId, runId);
    await refreshWorkspace();
  }

  function handleRequestExport(runId) {
    if (!runId) return;
    setExportRunId(runId);
  }

  async function handleConfirmExport({ format, dataFormat }) {
    if (!notebookId || !exportRunId) return;
    setExportPending(true);
    try {
      await api.exportRun(notebookId, exportRunId, { format, dataFormat });
      setExportPrefs({ format, dataFormat });
      setExportRunId("");
    } catch (err) {
      alert(err instanceof Error ? err.message : "Export failed");
    } finally {
      setExportPending(false);
    }
  }

  return (
    <div className="page-shell notebook-page">
      <header className="topbar notebook-topbar">
        <VisionQBrandLink aria-label="Back to all workspaces" />
        <div className="notebook-topbar-title">
          <input
            className="notebook-title-input"
            value={notebookTitleDraft}
            onChange={(event) => setNotebookTitleDraft(event.target.value)}
            onBlur={() => handleNotebookTitleCommit()}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                event.currentTarget.blur();
              }
            }}
            maxLength={120}
            spellCheck={false}
            aria-label="Workspace name"
          />
          <Pencil size={16} className="notebook-title-pencil" aria-hidden />
        </div>
        <div className="notebook-topbar-actions">
          <Link className="notebook-saved-pill" to="/saved">
            <CheckCircle2 size={18} className="notebook-saved-check" aria-hidden />
            Saved
          </Link>
          {user ? <UserAccountMenu /> : null}
        </div>
      </header>

      <QuizBuilderModal
        open={quizBuilderOpen}
        pendingRun={pendingRun}
        questions={questions}
        selectedQuestionTypes={selectedQuestionTypes}
        selectedSourceId={selectedSourceId}
        sources={workspace.sources}
        questionTypeLabels={QUIZ_QUESTION_TYPE_LABELS}
        setQuestions={setQuestions}
        setSelectedQuestionTypes={setSelectedQuestionTypes}
        setSelectedSourceId={setSelectedSourceId}
        onClose={() => setQuizBuilderOpen(false)}
        onCreate={handleGenerateQuiz}
      />

      <SaveToListModal
        open={Boolean(saveListRunId)}
        notebookId={notebookId}
        runId={saveListRunId}
        onClose={() => setSaveListRunId("")}
        onAdded={() => { }}
      />

      <ExportOptionsModal
        open={Boolean(exportRunId)}
        pending={exportPending}
        defaultFormat={exportPrefs.format}
        defaultDataFormat={exportPrefs.dataFormat}
        onCancel={() => {
          if (exportPending) return;
          setExportRunId("");
        }}
        onConfirm={handleConfirmExport}
      />

      <main ref={workspaceRef} className="workspace-flex">
        <div
          className="workspace-pane workspace-pane--left"
          style={{ width: columnWidths.left, flex: "0 0 auto" }}
        >
          <SourcePanel
            pendingUpload={pendingUpload}
            selectedSourceId={selectedSourceId}
            sources={workspace.sources}
            onUploadFile={handleUploadFile}
            setSelectedSourceId={setSelectedSourceId}
          />
        </div>
        <div
          className="workspace-resizer"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize columns"
          tabIndex={-1}
          onMouseDown={(event) => startColumnDrag(workspaceRef.current)(event)}
        />
        <div className="workspace-pane workspace-pane--right">
          <QuizPanel
            activeQuestionIndex={activeQuestionIndex}
            canGenerate={Boolean(primarySource)}
            notebookId={notebookId}
            pendingRun={pendingRun}
            resultStats={resultStats}
            resultsOpen={resultsOpen}
            selectedAnswers={selectedAnswers}
            selectedRun={selectedRun}
            totalQuestions={totalQuestions}
            workspaceRuns={workspace.runs}
            setActiveQuestionIndex={setActiveQuestionIndex}
            setResultsOpen={setResultsOpen}
            setSelectedAnswers={setSelectedAnswers}
            setSelectedRunId={setSelectedRunId}
            onDeleteRun={handleDeleteRun}
            onRequestExport={handleRequestExport}
            onExitQuiz={flushAndExitQuiz}
            onOpenQuizBuilder={() => setQuizBuilderOpen(true)}
            onRedoQuiz={handleRedoQuiz}
            onRenameRun={handleRenameRun}
            onRequestSaveToList={(runId) => setSaveListRunId(runId)}
          />
        </div>
      </main>
    </div>
  );
}

export default NotebookPage;
