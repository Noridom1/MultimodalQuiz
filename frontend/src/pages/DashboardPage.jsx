import { useEffect, useState, useTransition } from "react";
import {
  BookOpen,
  ChevronRight,
  Grid2X2,
  LayoutPanelLeft,
  LoaderCircle,
  Plus,
  Search,
  Star,
} from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import LoadingPanel from "../components/common/LoadingPanel";
import UserAccountMenu from "../components/common/UserAccountMenu";
import { api } from "../api";
import { useAuth } from "../context/AuthContext";
import { heroStyle } from "../utils/ui";

function DashboardPage() {
  const navigate = useNavigate();
  const { user, supabaseConfigured } = useAuth();
  const [notebooks, setNotebooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [creating, startCreating] = useTransition();

  useEffect(() => {
    let active = true;
    setLoading(true);
    api
      .listNotebooks()
      .then((data) => {
        if (active) {
          setNotebooks(data);
        }
      })
      .catch(() => {
        if (active) {
          setNotebooks([]);
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
  }, [user?.id, supabaseConfigured]);

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
          <Link className="ghost-pill" to="/search">
            <Search size={18} />
            Search
          </Link>
          <Link className="ghost-pill" to="/saved">
            <Star size={18} />
            Saved
          </Link>
          {user ? (
            <UserAccountMenu />
          ) : supabaseConfigured ? (
            <Link className="ghost-pill" to="/login">
              Sign in
            </Link>
          ) : null}
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
          <button className="ghost-inline">
            See all <ChevronRight size={16} />
          </button>
        </div>
        {loading ? (
          <LoadingPanel />
        ) : (
          <div className="featured-grid">
            {(featured.length ? featured : [{ title: "Start from a source" }]).map((notebook, index) => (
              <button
                key={notebook.id || index}
                className="hero-card"
                onClick={() => (notebook.id ? navigate(`/notebooks/${notebook.id}`) : handleCreateNotebook())}
                style={heroStyle(notebook, index)}
              >
                <div className="hero-overlay" />
                <div className="hero-content">
                  <span className="eyebrow">{notebook.source_count || 0} sources</span>
                  <h3>{notebook.title}</h3>
                  <p>
                    {notebook.description ||
                      "Notebook-ready workspace for generation, source review, and quiz output."}
                  </p>
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

export default DashboardPage;
