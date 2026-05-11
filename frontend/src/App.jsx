import { useEffect, useState, useTransition } from "react";
import {
  ArrowRight,
  BookOpen,
  Bot,
  ChevronRight,
  Grid2X2,
  LayoutPanelLeft,
  LoaderCircle,
  Plus,
  Search,
  SendHorizontal,
  Sparkles,
  Upload,
} from "lucide-react";
import { Link, Route, Routes, useNavigate, useParams } from "react-router-dom";
import { api } from "./api";

function App() {
  return (
    <Routes>
      <Route path="/" element={<DashboardPage />} />
      <Route path="/notebooks/:notebookId" element={<NotebookPage />} />
    </Routes>
  );
}

function DashboardPage() {
  const navigate = useNavigate();
  const [notebooks, setNotebooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [creating, startCreating] = useTransition();

  useEffect(() => {
    let active = true;
    api
      .listNotebooks()
      .then((data) => {
        if (active) {
          setNotebooks(data);
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
  }, []);

  const featured = notebooks.slice(0, 3);

  function handleCreateNotebook() {
    startCreating(async () => {
      const created = await api.createNotebook({
        title: `Untitled notebook ${notebooks.length + 1}`,
        description: "A workspace for multimodal quiz generation.",
      });
      navigate(`/notebooks/${created.id}`);
    });
  }

  return (
    <div className="page-shell dashboard-page">
      <header className="topbar">
        <Link className="brand" to="/">
          <div className="brand-mark">
            <BookOpen size={22} />
          </div>
          <span>NotebookQuiz</span>
        </Link>
        <div className="topbar-actions">
          <button className="ghost-pill">
            <Search size={18} />
            Search
          </button>
          <button className="primary-pill" onClick={handleCreateNotebook} disabled={creating}>
            {creating ? <LoaderCircle className="spin" size={18} /> : <Plus size={18} />}
            Create new
          </button>
        </div>
      </header>

      <section className="dashboard-controls">
        <div className="chip-row">
          <button className="chip active">All</button>
          <button className="chip">My notebooks</button>
          <button className="chip">Highlights</button>
          <button className="chip">Shared with me</button>
        </div>
        <div className="chip-row">
          <button className="icon-chip active">
            <Grid2X2 size={18} />
          </button>
          <button className="icon-chip">
            <LayoutPanelLeft size={18} />
          </button>
        </div>
      </section>

      <section className="dashboard-section">
        <div className="section-heading">
          <h2>Featured notebooks</h2>
          <button className="ghost-inline">See all <ChevronRight size={16} /></button>
        </div>
        {loading ? (
          <LoadingPanel />
        ) : (
          <div className="featured-grid">
            {(featured.length ? featured : [{ title: "Start from a source" }]).map((notebook, index) => (
              <button
                key={notebook.id || index}
                className="hero-card"
                onClick={() => notebook.id ? navigate(`/notebooks/${notebook.id}`) : handleCreateNotebook()}
                style={heroStyle(notebook, index)}
              >
                <div className="hero-overlay" />
                <div className="hero-content">
                  <span className="eyebrow">{notebook.source_count || 0} sources</span>
                  <h3>{notebook.title}</h3>
                  <p>{notebook.description || "Notebook-ready workspace for generation, source review, and quiz output."}</p>
                </div>
              </button>
            ))}
          </div>
        )}
      </section>

      <section className="dashboard-section">
        <div className="section-heading">
          <h2>Recent notebooks</h2>
        </div>
        <div className="notebook-grid">
          {notebooks.map((notebook, index) => (
            <button
              key={notebook.id}
              className="notebook-card"
              onClick={() => navigate(`/notebooks/${notebook.id}`)}
            >
              <div className="notebook-thumb" style={heroStyle(notebook, index)} />
              <div className="notebook-meta">
                <h3>{notebook.title}</h3>
                <p>{notebook.question_count || 0} questions - {notebook.source_count || 0} sources</p>
              </div>
            </button>
          ))}
          {!loading && notebooks.length === 0 ? (
            <button className="notebook-card empty-card" onClick={handleCreateNotebook}>
              <div className="empty-orb">
                <Plus size={28} />
              </div>
              <div className="notebook-meta">
                <h3>Create your first notebook</h3>
                <p>Upload a source and generate a multimodal quiz.</p>
              </div>
            </button>
          ) : null}
        </div>
      </section>
    </div>
  );
}

function NotebookPage() {
  const { notebookId } = useParams();
  const [workspace, setWorkspace] = useState(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [questions, setQuestions] = useState(5);
  const [pendingMessage, startMessageTransition] = useTransition();
  const [pendingRun, startRunTransition] = useTransition();
  const [pendingUpload, startUploadTransition] = useTransition();

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
  }, [notebookId]);

  if (loading || !workspace) {
    return (
      <div className="page-shell notebook-page">
        <LoadingPanel />
      </div>
    );
  }

  const primarySource = workspace.sources[0];
  const latestRun = workspace.latest_run;

  function refreshWorkspace() {
    if (!notebookId) {
      return;
    }
    api.getNotebook(notebookId).then(setWorkspace);
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

  function handleGenerateQuiz(sourceId) {
    if (!sourceId) {
      return;
    }
    startRunTransition(async () => {
      await api.generateQuiz(notebookId, {
        source_id: sourceId,
        num_questions: questions,
        mock_image: false,
        mock_question: false,
      });
      refreshWorkspace();
    });
  }

  return (
    <div className="page-shell notebook-page">
      <header className="topbar notebook-topbar">
        <Link className="brand" to="/">
          <div className="brand-mark">
            <BookOpen size={20} />
          </div>
          <span>{workspace.notebook.title}</span>
        </Link>
        <div className="topbar-actions">
          <button className="primary-pill compact" onClick={() => handleGenerateQuiz(primarySource?.id)} disabled={!primarySource || pendingRun}>
            {pendingRun ? <LoaderCircle className="spin" size={18} /> : <Plus size={18} />}
            Generate quiz
          </button>
        </div>
      </header>

      <main className="workspace-grid">
        <section className="panel source-panel">
          <div className="panel-header">
            <h2>Sources</h2>
          </div>
          <label className="upload-cta">
            <input type="file" hidden onChange={handleUploadFile} />
            <Upload size={18} />
            {pendingUpload ? "Uploading..." : "Add source"}
          </label>
          <div className="search-card">
            <div className="search-label">
              <Search size={18} />
              <span>Find source material for this notebook</span>
            </div>
            <p>Upload PDFs, markdown, or text files. The generator uses the local copy and persists metadata in Supabase-backed storage.</p>
          </div>
          <div className="source-list">
            {workspace.sources.map((source) => (
              <button key={source.id} className="source-item" onClick={() => handleGenerateQuiz(source.id)}>
                <div>
                  <strong>{source.title}</strong>
                  <span>{formatBytes(source.size_bytes)} - {source.content_type}</span>
                </div>
                <ArrowRight size={18} />
              </button>
            ))}
            {workspace.sources.length === 0 ? (
              <div className="empty-panel">
                <p>Saved sources will appear here.</p>
              </div>
            ) : null}
          </div>
        </section>

        <section className="panel conversation-panel">
          <div className="panel-header">
            <h2>Conversation</h2>
          </div>
          <div className="conversation-scroll">
            {workspace.messages.length <= 1 ? (
              <div className="conversation-hero">
                <div className="hero-icon">*</div>
                <h1>Start building your quiz notebook</h1>
                <p>
                  Upload source material, then generate a multimodal quiz package with image-backed question cards.
                </p>
                <div className="prompt-chips">
                  <button className="prompt-chip" onClick={() => setMessage("Summarize the current sources")}>Summarize sources</button>
                  <button className="prompt-chip" onClick={() => setMessage("What does the latest quiz cover?")}>Review latest run</button>
                  <button className="prompt-chip" onClick={() => setMessage("How many source files are ready?")}>Check readiness</button>
                </div>
              </div>
            ) : null}

            <div className="message-list">
              {workspace.messages.map((entry) => (
                <article key={entry.id} className={`message ${entry.role}`}>
                  <div className="message-avatar">{entry.role === "assistant" ? <Bot size={16} /> : "You"}</div>
                  <div className="message-body">
                    <p>{entry.content}</p>
                  </div>
                </article>
              ))}
            </div>
          </div>
          <form className="composer" onSubmit={handleSendMessage}>
            <div className="composer-controls">
              <label className="question-count">
                Questions
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={questions}
                  onChange={(event) => setQuestions(Number(event.target.value))}
                />
              </label>
              <button
                type="button"
                className="inline-action"
                onClick={() => handleGenerateQuiz(primarySource?.id)}
                disabled={!primarySource || pendingRun}
              >
                <Sparkles size={16} />
                {pendingRun ? "Generating..." : "Run pipeline"}
              </button>
            </div>
            <div className="composer-bar">
              <input
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                placeholder="Ask about sources, runs, or quiz coverage..."
              />
              <button type="submit" disabled={pendingMessage}>
                {pendingMessage ? <LoaderCircle className="spin" size={18} /> : <SendHorizontal size={18} />}
              </button>
            </div>
          </form>
        </section>

        <section className="panel studio-panel">
          <div className="panel-header">
            <h2>Studio</h2>
          </div>
          <div className="studio-actions">
            <button className="studio-tile" onClick={() => handleGenerateQuiz(primarySource?.id)} disabled={!primarySource}>
              <Sparkles size={18} />
              <span>Generate quiz</span>
            </button>
            <button className="studio-tile" onClick={() => setMessage("What does the latest quiz cover?")}>
              <BookOpen size={18} />
              <span>Summarize run</span>
            </button>
            <button className="studio-tile" onClick={() => setMessage("How many source files are ready?")}>
              <LayoutPanelLeft size={18} />
              <span>Review sources</span>
            </button>
          </div>

          {latestRun ? (
            <div className="run-card">
              <div className="run-card-media" style={heroStyle({ hero_image: latestRun.summary.hero_image }, 0)} />
              <div className="run-card-body">
                <span className="eyebrow">{latestRun.status}</span>
                <h3>{latestRun.summary.title}</h3>
                <p>{latestRun.summary.question_count} questions - {latestRun.summary.concepts.join(", ")}</p>
                <div className="artifact-links">
                  <a href={latestRun.summary.artifact_paths.quiz_package} target="_blank" rel="noreferrer">Quiz package</a>
                  <a href={latestRun.summary.artifact_paths.graph_html} target="_blank" rel="noreferrer">Graph</a>
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-panel studio-empty">
              <p>Studio output will appear here after the first run.</p>
            </div>
          )}

          <div className="question-stack">
            {(latestRun?.summary?.results || []).slice(0, 4).map((item) => (
              <article key={item.index} className="question-card">
                <div className="question-card-image" style={heroStyle({ hero_image: item.image_url }, item.index)} />
                <div className="question-card-body">
                  <span className="eyebrow">{item.difficulty} - {item.target_concept}</span>
                  <h4>{item.question_text}</h4>
                  <p>{item.explanation}</p>
                </div>
              </article>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

function LoadingPanel() {
  return (
    <div className="loading-panel">
      <LoaderCircle className="spin" size={26} />
      <span>Loading workspace...</span>
    </div>
  );
}

function heroStyle(item, index) {
  const gradients = [
    "linear-gradient(135deg, rgba(90,103,255,.8), rgba(12,18,44,.95))",
    "linear-gradient(135deg, rgba(223,120,62,.85), rgba(41,24,14,.92))",
    "linear-gradient(135deg, rgba(78,179,148,.85), rgba(11,35,34,.95))",
  ];
  return item?.hero_image
    ? { backgroundImage: `linear-gradient(180deg, rgba(19,23,31,.05), rgba(19,23,31,.9)), url(${item.hero_image})` }
    : { backgroundImage: gradients[index % gradients.length] };
}

function formatBytes(value) {
  if (!value) {
    return "0 B";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export default App;

