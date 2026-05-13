import { useCallback, useEffect, useMemo, useState, useTransition } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BookOpen,
  ChevronRight,
  Clock,
  ExternalLink,
  Grid2X2,
  LayoutGrid,
  LayoutList,
  LoaderCircle,
  MoreVertical,
  Plus,
  Search,
  Sparkles,
  Star,
  Users,
} from "lucide-react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import LoadingPanel from "../components/common/LoadingPanel";
import UserAccountMenu from "../components/common/UserAccountMenu";
import { api } from "../api";
import { useAuth } from "../context/AuthContext";
import { heroStyle } from "../utils/ui";

function displayUserName(user) {
  const meta = user?.user_metadata || {};
  if (typeof meta.full_name === "string" && meta.full_name.trim()) {
    return meta.full_name.trim();
  }
  const email = user?.email || "";
  if (email) return email.split("@")[0] || "Account";
  return "Account";
}

function formatEditedAgo(iso) {
  if (!iso) return "Recently";
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "Recently";
  const sec = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (sec < 60) return "just now";
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 48) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  return `${day}d ago`;
}

function initialsFromUser(user) {
  const email = user?.email || "";
  const meta = user?.user_metadata || {};
  const name = typeof meta.full_name === "string" ? meta.full_name : "";
  if (name.trim()) {
    const parts = name.trim().split(/\s+/);
    const a = parts[0]?.[0] || "";
    const b = parts[1]?.[0] || "";
    return (a + b).toUpperCase() || a.toUpperCase() || "?";
  }
  if (email) return email.slice(0, 2).toUpperCase();
  return "?";
}

export default function DashboardPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const { user, supabaseConfigured } = useAuth();
  const [creating, startCreating] = useTransition();
  const [filter, setFilter] = useState("all");
  const [viewMode, setViewMode] = useState("grid");

  const notebooksQuery = useQuery({
    queryKey: ["notebooks", user?.id ?? "anon", supabaseConfigured ? "sb" : "local"],
    queryFn: async () => {
      try {
        return await api.listNotebooks();
      } catch {
        return [];
      }
    },
  });

  const notebooks = notebooksQuery.data ?? [];
  const showInitialLoading = notebooksQuery.isPending && notebooks.length === 0;

  useEffect(() => {
    function onKeyDown(event) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        navigate("/search");
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [navigate]);

  const onDashboard = location.pathname === "/";

  const sortedNotebooks = useMemo(() => {
    const copy = [...notebooks];
    copy.sort((a, b) => {
      const ta = Date.parse(a.updated_at || a.created_at || 0) || 0;
      const tb = Date.parse(b.updated_at || b.created_at || 0) || 0;
      return tb - ta;
    });
    return copy;
  }, [notebooks]);

  const filteredNotebooks = useMemo(() => {
    if (filter === "highlights") {
      return sortedNotebooks.filter((n) => (n.question_count || 0) > 0);
    }
    if (filter === "shared") {
      return [];
    }
    return sortedNotebooks;
  }, [sortedNotebooks, filter]);

  const featuredPrimary = filteredNotebooks[0];
  const featuredSecondary = filteredNotebooks.slice(1, 3);

  const handleCreateNotebook = useCallback(() => {
    startCreating(async () => {
      const created = await api.createNotebook({
        title: `Untitled workspace ${notebooks.length + 1}`,
        description: "A workspace for multimodal quiz generation.",
      });
      await queryClient.invalidateQueries({ queryKey: ["notebooks"] });
      navigate(`/notebooks/${created.id}`);
    });
  }, [navigate, notebooks.length, queryClient]);

  function navItemClass(active) {
    return `dash-nav-item${active ? " is-active" : ""}`;
  }

  return (
    <div className="dashboard-page">
      <aside className="dashboard-sidebar" aria-label="Workspace">
        <Link to="/" className="dash-sidebar-brand">
          <span className="visionq-logo-mark" aria-hidden>
            VQ
          </span>
          <span className="visionq-wordmark">VisionQ</span>
        </Link>

        <nav className="dash-nav" aria-label="Primary">
          <button
            type="button"
            className={navItemClass(onDashboard && filter === "all")}
            onClick={() => {
              navigate("/");
              setFilter("all");
            }}
          >
            <LayoutGrid size={18} strokeWidth={1.75} aria-hidden />
            All workspaces
          </button>
          <button
            type="button"
            className={navItemClass(onDashboard && filter === "my")}
            onClick={() => {
              navigate("/");
              setFilter("my");
            }}
          >
            <BookOpen size={18} strokeWidth={1.75} aria-hidden />
            My workspaces
          </button>
          <NavLink to="/saved" className={({ isActive }) => navItemClass(isActive)}>
            <Star size={18} strokeWidth={1.75} aria-hidden />
            Highlights
          </NavLink>
          <button
            type="button"
            className={navItemClass(onDashboard && filter === "shared")}
            onClick={() => {
              navigate("/");
              setFilter("shared");
            }}
          >
            <Users size={18} strokeWidth={1.75} aria-hidden />
            Shared with me
          </button>
        </nav>

        <div className="dash-nav-label">Views</div>
        <div className="dash-nav dash-nav--compact">
          <button
            type="button"
            className={navItemClass(viewMode === "grid")}
            onClick={() => setViewMode("grid")}
          >
            <Grid2X2 size={18} strokeWidth={1.75} aria-hidden />
            Grid view
          </button>
          <button
            type="button"
            className={navItemClass(viewMode === "list")}
            onClick={() => setViewMode("list")}
          >
            <LayoutList size={18} strokeWidth={1.75} aria-hidden />
            List view
          </button>
        </div>

        <div className="dash-sidebar-spacer" />

        <div className="dash-sidebar-promo">
          <Sparkles size={20} className="dash-sidebar-promo-icon" strokeWidth={1.75} aria-hidden />
          <strong>Create your first workspace</strong>
          <p>Upload sources and generate multimodal quizzes in minutes.</p>
          <button type="button" className="primary-pill compact dash-sidebar-promo-btn" onClick={handleCreateNotebook}>
            <Plus size={16} aria-hidden />
            Create new
          </button>
        </div>

        {user ? (
          <div className="dash-sidebar-user">
            <span className="dash-sidebar-user-avatar" aria-hidden>
              {initialsFromUser(user)}
            </span>
            <div className="dash-sidebar-user-text">
              <span className="dash-sidebar-user-name">{displayUserName(user)}</span>
              <span className="dash-sidebar-user-email">{user.email}</span>
            </div>
            <UserAccountMenu />
          </div>
        ) : null}
      </aside>

      <div className="dashboard-main">
        <header className="dashboard-main-topbar">
          <div className="dashboard-topbar-lead" aria-hidden="true" />
          <button type="button" className="dashboard-search-pill" onClick={() => navigate("/search")}>
            <Search size={17} strokeWidth={2} aria-hidden />
            <span className="dashboard-search-placeholder">Search</span>
            <kbd className="dashboard-kbd">⌘K</kbd>
          </button>
          <div className="dashboard-main-topbar-actions">
            <Link className="ghost-pill compact dashboard-saved-link" to="/saved">
              <Star size={16} strokeWidth={2} aria-hidden />
              Saved
            </Link>
            {user ? (
              <UserAccountMenu />
            ) : supabaseConfigured ? (
              <Link className="ghost-pill compact" to="/login">
                Sign in
              </Link>
            ) : null}
            <button type="button" className="primary-pill compact" onClick={handleCreateNotebook} disabled={creating}>
              {creating ? <LoaderCircle className="spin" size={18} /> : <Plus size={18} />}
              Create new
            </button>
          </div>
        </header>

        <div className="dashboard-main-scroll">
          <header className="dashboard-page-head">
            <h1 className="dashboard-page-title">All workspaces</h1>
            <p className="dashboard-page-sub">Your workspace for multimodal quiz generation.</p>
          </header>

          <section className="dashboard-toolbar" aria-label="Filters and display">
            <div className="dashboard-toolbar-chips">
              <button
                type="button"
                className={`dashboard-chip${filter === "all" ? " is-active" : ""}`}
                onClick={() => setFilter("all")}
              >
                All
              </button>
              <button
                type="button"
                className={`dashboard-chip${filter === "my" ? " is-active" : ""}`}
                onClick={() => setFilter("my")}
              >
                My workspaces
              </button>
              <button
                type="button"
                className={`dashboard-chip${filter === "highlights" ? " is-active" : ""}`}
                onClick={() => setFilter("highlights")}
              >
                Highlights
              </button>
              <button
                type="button"
                className={`dashboard-chip${filter === "shared" ? " is-active" : ""}`}
                onClick={() => setFilter("shared")}
              >
                Shared with me
              </button>
            </div>
            <div className="dashboard-toolbar-right">
              {notebooksQuery.isFetching && !notebooksQuery.isPending ? (
                <span className="dashboard-sync-hint" aria-live="polite">
                  <LoaderCircle className="spin" size={14} aria-hidden />
                  Updating
                </span>
              ) : null}
              <div className="dashboard-view-toggle" role="group" aria-label="View mode">
                <button
                  type="button"
                  className={viewMode === "grid" ? "is-active" : ""}
                  onClick={() => setViewMode("grid")}
                  aria-pressed={viewMode === "grid"}
                >
                  Grid
                </button>
                <button
                  type="button"
                  className={viewMode === "list" ? "is-active" : ""}
                  onClick={() => setViewMode("list")}
                  aria-pressed={viewMode === "list"}
                >
                  List
                </button>
              </div>
              <button type="button" className="dashboard-sort-pill">
                Sort: Recently updated
                <ChevronRight size={14} className="dashboard-sort-chevron" aria-hidden />
              </button>
            </div>
          </section>

          <section className="dashboard-section dash-section">
            <div className="dash-section-head">
              <h2 className="dash-section-title">
                <Star size={17} strokeWidth={2} className="dash-section-title-icon" aria-hidden />
                Featured workspaces
              </h2>
              <Link to="/search" className="dash-section-link">
                See all <ChevronRight size={15} aria-hidden />
              </Link>
            </div>
            {showInitialLoading ? (
              <LoadingPanel label="Loading workspaces..." />
            ) : (
              <div className="dash-featured-layout">
                <div className="dash-feature-hero-wrap">
                  {featuredPrimary ? (
                    <button
                      type="button"
                      className="dash-feature-hero"
                      onClick={() => navigate(`/notebooks/${featuredPrimary.id}`)}
                    >
                      <div
                        className="dash-feature-hero-media"
                        style={heroStyle(featuredPrimary, 0)}
                        aria-hidden
                      />
                      <div className="dash-feature-hero-body">
                        <h3>{featuredPrimary.title}</h3>
                        <p>
                          {featuredPrimary.description ||
                            "Quiz-ready workspace for generation, source review, and quiz output."}
                        </p>
                        <div className="dash-notebook-meta">
                          <span>
                            {featuredPrimary.question_count || 0} questions • {featuredPrimary.source_count || 0} source
                            {(featuredPrimary.source_count || 0) === 1 ? "" : "s"}
                          </span>
                        </div>
                        <div className="dash-feature-hero-footer">
                          <span className="dash-feature-edited">You edited {formatEditedAgo(featuredPrimary.updated_at)}</span>
                          <button
                            type="button"
                            className="dash-feature-open"
                            onClick={(e) => {
                              e.stopPropagation();
                              navigate(`/notebooks/${featuredPrimary.id}`);
                            }}
                          >
                            Open
                          </button>
                        </div>
                      </div>
                    </button>
                  ) : (
                    <button type="button" className="dash-feature-hero dash-feature-hero--empty" onClick={handleCreateNotebook}>
                      <div className="dash-feature-hero-media dash-feature-hero-media--empty" aria-hidden />
                      <div className="dash-feature-hero-body">
                        <h3>Start from a source</h3>
                        <p>Create a workspace, add a file, and generate your first quiz.</p>
                        <div className="dash-feature-hero-footer">
                          <span className="dash-feature-edited">New workspace</span>
                          <span className="dash-feature-open">Create</span>
                        </div>
                      </div>
                    </button>
                  )}
                </div>
                <div className="dash-feature-stack">
                  {featuredSecondary[0] ? (
                    <button
                      type="button"
                      className="dash-feature-mini"
                      onClick={() => navigate(`/notebooks/${featuredSecondary[0].id}`)}
                    >
                      <div className="dash-feature-mini-icon" aria-hidden>
                        <BookOpen size={20} />
                      </div>
                      <div className="dash-feature-mini-text">
                        <strong>{featuredSecondary[0].title}</strong>
                        <span className="dash-notebook-meta">
                          {featuredSecondary[0].question_count || 0} questions • {featuredSecondary[0].source_count || 0}{" "}
                          sources
                        </span>
                      </div>
                      <ChevronRight size={18} className="dash-feature-mini-chevron" aria-hidden />
                    </button>
                  ) : (
                    <div className="dash-feature-mini dash-feature-mini--static">
                      <div className="dash-feature-mini-icon" aria-hidden>
                        <Sparkles size={20} />
                      </div>
                      <div className="dash-feature-mini-text">
                        <strong>Getting started</strong>
                        <span>Tips for your first quiz</span>
                      </div>
                      <ChevronRight size={18} className="dash-feature-mini-chevron" aria-hidden />
                    </div>
                  )}
                  {featuredSecondary[1] ? (
                    <button
                      type="button"
                      className="dash-feature-mini"
                      onClick={() => navigate(`/notebooks/${featuredSecondary[1].id}`)}
                    >
                      <div className="dash-feature-mini-icon" aria-hidden>
                        <LayoutGrid size={20} />
                      </div>
                      <div className="dash-feature-mini-text">
                        <strong>{featuredSecondary[1].title}</strong>
                        <span className="dash-notebook-meta">
                          {featuredSecondary[1].question_count || 0} questions • {featuredSecondary[1].source_count || 0}{" "}
                          sources
                        </span>
                      </div>
                      <ChevronRight size={18} className="dash-feature-mini-chevron" aria-hidden />
                    </button>
                  ) : (
                    <div className="dash-feature-mini dash-feature-mini--static">
                      <div className="dash-feature-mini-icon" aria-hidden>
                        <BookOpen size={20} />
                      </div>
                      <div className="dash-feature-mini-text">
                        <strong>Example workspaces</strong>
                        <span>Ideas and layouts</span>
                      </div>
                      <ChevronRight size={18} className="dash-feature-mini-chevron" aria-hidden />
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>

          <section className="dashboard-section dash-section">
            <div className="dash-section-head">
              <h2 className="dash-section-title">
                <Clock size={17} strokeWidth={2} className="dash-section-title-icon" aria-hidden />
                Recent workspaces
              </h2>
              <Link to="/search" className="dash-section-link">
                See all <ChevronRight size={15} aria-hidden />
              </Link>
            </div>
            <div className={viewMode === "list" ? "notebook-grid notebook-grid--list" : "notebook-grid"}>
              {filteredNotebooks.map((notebook, index) => (
                <button
                  key={notebook.id}
                  type="button"
                  className="notebook-card dash-notebook-card"
                  onClick={() => navigate(`/notebooks/${notebook.id}`)}
                >
                  <div className="notebook-thumb" style={heroStyle(notebook, index)} />
                  <div className="notebook-meta">
                    <h3>{notebook.title}</h3>
                    <p className="dash-notebook-meta">
                      {notebook.question_count || 0} questions <span aria-hidden>•</span> {notebook.source_count || 0}{" "}
                      source{(notebook.source_count || 0) === 1 ? "" : "s"}
                    </p>
                    <div className="dash-notebook-card-footer">
                      <span className="dash-notebook-card-avatar" aria-hidden>
                        {user ? initialsFromUser(user) : "•"}
                      </span>
                      <span className="dash-notebook-card-edited">Edited {formatEditedAgo(notebook.updated_at)}</span>
                      <span
                        className="dash-notebook-card-menu"
                        role="presentation"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <MoreVertical size={16} aria-hidden />
                      </span>
                    </div>
                  </div>
                </button>
              ))}
              {!showInitialLoading && filteredNotebooks.length === 0 ? (
                <button type="button" className="notebook-card empty-card dash-notebook-card" onClick={handleCreateNotebook}>
                  <div className="empty-orb">
                    <Plus size={28} />
                  </div>
                  <div className="notebook-meta">
                    <h3>Create your first workspace</h3>
                    <p>Upload a source and generate a multimodal quiz.</p>
                  </div>
                </button>
              ) : null}
            </div>
          </section>

          <footer className="dashboard-info-banner">
            <BookOpen size={22} strokeWidth={1.75} className="dashboard-info-banner-icon" aria-hidden />
            <p>
              Workspaces help you organize and generate better quizzes. Combine text, images, audio, and more to create
              engaging assessments.
            </p>
            <Link to="/search" className="dashboard-info-banner-link">
              Learn more <ExternalLink size={15} aria-hidden />
            </Link>
          </footer>
        </div>
      </div>
    </div>
  );
}
