import { useEffect, useLayoutEffect, useRef, useState, useTransition } from "react";
import { BookOpen } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import LoadingPanel from "../components/common/LoadingPanel";
import UserAccountMenu from "../components/common/UserAccountMenu";
import ConversationPanel from "../components/notebook/ConversationPanel";
import QuizBuilderModal from "../components/notebook/QuizBuilderModal";
import QuizPanel from "../components/notebook/QuizPanel";
import SourcePanel from "../components/notebook/SourcePanel";
import { useAuth } from "../context/AuthContext";
import { useWorkspaceColumnResize } from "../hooks/useWorkspaceColumnResize";
import { loadQuizSession, saveQuizSession } from "../utils/quizSessionStorage";

function NotebookPage() {
  const { notebookId } = useParams();
  const { user } = useAuth();
  const [workspace, setWorkspace] = useState(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [quizBuilderOpen, setQuizBuilderOpen] = useState(false);
  const [selectedSourceId, setSelectedSourceId] = useState("");
  const [questions, setQuestions] = useState(10);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [activeQuestionIndex, setActiveQuestionIndex] = useState(0);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [resultsOpen, setResultsOpen] = useState(false);
  const [pendingMessage, startMessageTransition] = useTransition();
  const [pendingRun, startRunTransition] = useTransition();
  const [pendingUpload, startUploadTransition] = useTransition();
  const [notebookTitleDraft, setNotebookTitleDraft] = useState("");
  const workspaceRef = useRef(null);
  const { widths: columnWidths, startDrag: startColumnDrag } = useWorkspaceColumnResize();

  useEffect(() => {
    if (!notebookId) {
      return;
    }
    let active = true;
    setLoading(true);
    api
      .getNotebook(notebookId)
      .then((data) => {
        if (active) {
          setWorkspace(data);
        }
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [notebookId, user?.id]);

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

  useEffect(() => {
    const completedRuns = (workspace?.runs || []).filter((run) => run.status === "completed");
    if (!completedRuns.length) {
      setSelectedRunId("");
      return;
    }
    setSelectedRunId((current) => {
      if (current && completedRuns.some((run) => run.run_id === current)) {
        return current;
      }
      return "";
    });
  }, [workspace?.runs]);

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

  if (loading || !workspace) {
    return (
      <div className="page-shell notebook-page">
        <LoadingPanel />
      </div>
    );
  }

  const primarySource = workspace.sources[0];
  const completedRuns = workspace.runs.filter((run) => run.status === "completed");
  const selectedRun =
    completedRuns.find((run) => run.run_id === selectedRunId) || null;
  const quizResults = selectedRun?.summary?.results || [];
  const totalQuestions = quizResults.length;

  const score = quizResults.reduce((total, item, index) => {
    return total + (selectedAnswers[index] === item.correct_answer ? 1 : 0);
  }, 0);

  const answeredCount = Array.from({ length: totalQuestions }, (_, index) => index).filter((index) =>
    Object.prototype.hasOwnProperty.call(selectedAnswers, index),
  ).length;

  const wrongCount = quizResults.reduce((total, item, index) => {
    if (!Object.prototype.hasOwnProperty.call(selectedAnswers, index)) return total;
    return total + (selectedAnswers[index] !== item.correct_answer ? 1 : 0);
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
    return api.getNotebook(notebookId).then((data) => {
      setWorkspace(data);
      return data;
    });
  }

  function handleSendMessage(event) {
    event.preventDefault();
    if (!message.trim()) {
      return;
    }
    startMessageTransition(async () => {
      await api.sendMessage(notebookId, message.trim());
      setMessage("");
      refreshWorkspace();
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
      const createdRun = await api.generateQuiz(notebookId, {
        source_id: selectedSourceId,
        num_questions: questions,
        mock_image: false,
        mock_question: false,
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
      setWorkspace((w) => (w ? { ...w, notebook: { ...w.notebook, ...updated } } : w));
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

  async function handleExportRun(runId) {
    if (!notebookId) return;
    await api.exportRun(notebookId, runId);
  }

  return (
    <div className="page-shell notebook-page">
      <header className="topbar notebook-topbar">
        <div className="notebook-topbar-brand">
          <Link className="brand-mark-link" to="/" aria-label="Back to notebooks">
            <div className="brand-mark">
              <BookOpen size={20} />
            </div>
          </Link>
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
            aria-label="Notebook title"
          />
        </div>
        <div className="notebook-topbar-actions">{user ? <UserAccountMenu /> : null}</div>
      </header>

      <QuizBuilderModal
        open={quizBuilderOpen}
        pendingRun={pendingRun}
        questions={questions}
        selectedSourceId={selectedSourceId}
        sources={workspace.sources}
        setQuestions={setQuestions}
        setSelectedSourceId={setSelectedSourceId}
        onClose={() => setQuizBuilderOpen(false)}
        onCreate={handleGenerateQuiz}
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
          aria-label="Resize sources column"
          tabIndex={-1}
          onMouseDown={(event) => startColumnDrag(0, workspaceRef.current)(event)}
        />
        <div className="workspace-pane workspace-pane--center">
          <ConversationPanel
            message={message}
            messages={workspace.messages}
            pendingMessage={pendingMessage}
            setMessage={setMessage}
            onSendMessage={handleSendMessage}
          />
        </div>
        <div
          className="workspace-resizer"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize quizzes column"
          tabIndex={-1}
          onMouseDown={(event) => startColumnDrag(1, workspaceRef.current)(event)}
        />
        <div
          className="workspace-pane workspace-pane--right"
          style={{ width: columnWidths.right, flex: "0 0 auto" }}
        >
          <QuizPanel
            activeQuestionIndex={activeQuestionIndex}
            canGenerate={Boolean(primarySource)}
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
            onExportRun={handleExportRun}
            onExitQuiz={flushAndExitQuiz}
            onOpenQuizBuilder={() => setQuizBuilderOpen(true)}
            onRedoQuiz={handleRedoQuiz}
            onRenameRun={handleRenameRun}
          />
        </div>
      </main>
    </div>
  );
}

export default NotebookPage;
