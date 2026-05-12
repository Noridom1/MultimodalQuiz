import { useEffect, useState, useTransition } from "react";
import { ChevronLeft, Folder, LoaderCircle, Plus, Star } from "lucide-react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import LoadingPanel from "../components/common/LoadingPanel";
import VisionQBrandLink from "../components/common/VisionQBrandLink";

const FOLDER_COLORS = ["#f0bf57", "#e37a46", "#9d6bff", "#5f6fff", "#4fb38a"];

function FolderGlyph({ color }) {
  return (
    <span className="saved-folder-icon" style={{ color }} aria-hidden>
      <Folder size={28} strokeWidth={1.75} fill="currentColor" fillOpacity={0.22} />
    </span>
  );
}

function CreateListGlyph() {
  return (
    <span className="saved-folder-icon saved-folder-icon--muted" aria-hidden>
      <Plus size={22} strokeWidth={2} />
    </span>
  );
}

function CreateListModal({ open, onClose, onCreated }) {
  const [title, setTitle] = useState("");
  const [color, setColor] = useState(FOLDER_COLORS[0]);
  const [pending, startTransition] = useTransition();

  useEffect(() => {
    if (open) {
      setTitle("");
      setColor(FOLDER_COLORS[0]);
    }
  }, [open]);

  if (!open) return null;

  function handleSubmit(event) {
    event.preventDefault();
    const trimmed = title.trim();
    if (!trimmed) return;
    startTransition(async () => {
      try {
        await api.createQuizList({ title: trimmed, folder_color: color });
        onCreated();
        onClose();
      } catch (err) {
        alert(err instanceof Error ? err.message : "Could not create list");
      }
    });
  }

  return (
    <div className="quiz-builder-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className="quiz-builder saved-create-modal"
        role="dialog"
        aria-labelledby="saved-create-title"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h2 id="saved-create-title">New list</h2>
        <form onSubmit={handleSubmit}>
          <label className="saved-create-label">
            Name
            <input
              className="saved-create-input"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Physics review"
              maxLength={120}
              autoFocus
            />
          </label>
          <div className="saved-color-chips" role="group" aria-label="Folder color">
            {FOLDER_COLORS.map((c) => (
              <button
                key={c}
                type="button"
                className={`saved-color-chip${color === c ? " saved-color-chip--active" : ""}`}
                style={{ background: c }}
                onClick={() => setColor(c)}
                aria-label={`Color ${c}`}
              />
            ))}
          </div>
          <div className="saved-create-actions">
            <button type="button" className="ghost-pill" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="primary-pill" disabled={pending || !title.trim()}>
              {pending ? <LoaderCircle className="spin" size={18} /> : null}
              Create
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}

function SavedPage() {
  const { listId } = useParams();
  const navigate = useNavigate();
  const [lists, setLists] = useState([]);
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [removingId, setRemovingId] = useState("");

  useEffect(() => {
    let active = true;
    if (listId) {
      setDetailLoading(true);
      setError("");
      api
        .getQuizList(listId)
        .then((data) => {
          if (active) setDetail(data);
        })
        .catch((err) => {
          if (active) {
            setError(err instanceof Error ? err.message : "Could not load list");
            setDetail(null);
          }
        })
        .finally(() => {
          if (active) setDetailLoading(false);
        });
    } else {
      setDetail(null);
    }
    return () => {
      active = false;
    };
  }, [listId]);

  useEffect(() => {
    if (listId) {
      setLoading(false);
      return undefined;
    }
    let active = true;
    setLoading(true);
    setError("");
    api
      .listQuizLists()
      .then((data) => {
        if (active) setLists(data);
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : "Could not load lists");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [listId]);

  function refreshLists() {
    api.listQuizLists().then(setLists).catch(() => {});
  }

  async function handleRemoveItem(itemId) {
    if (!listId || !window.confirm("Remove this quiz from the list?")) return;
    setRemovingId(itemId);
    try {
      await api.removeQuizListItem(listId, itemId);
      const next = await api.getQuizList(listId);
      setDetail(next);
      refreshLists();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Remove failed");
    } finally {
      setRemovingId("");
    }
  }

  if (listId) {
    return (
      <div className="page-shell saved-page">
        <header className="topbar saved-topbar">
          <button type="button" className="ghost-pill compact" onClick={() => navigate("/saved")}>
            <ChevronLeft size={18} />
            Saved
          </button>
          <VisionQBrandLink />
        </header>

        {detailLoading ? (
          <LoadingPanel />
        ) : error ? (
          <div className="saved-error">
            <p>{error}</p>
            <Link className="ghost-pill" to="/login">
              Sign in
            </Link>
          </div>
        ) : detail ? (
          <>
            <div className="saved-detail-head">
              <FolderGlyph color={detail.list?.folder_color || FOLDER_COLORS[0]} />
              <div>
                <h1 className="saved-detail-title">{detail.list?.title}</h1>
                <p className="saved-detail-meta">
                  {detail.item_count} quiz{detail.item_count === 1 ? "" : "es"} · {detail.question_count} questions
                </p>
              </div>
            </div>
            <ul className="saved-item-rows">
              {detail.items?.map((item) => (
                <li key={item.id} className="saved-item-row">
                  <Link
                    className="saved-item-link"
                    to={`/notebooks/${item.notebook_id}?run=${encodeURIComponent(item.run_id)}`}
                  >
                    <strong>{item.run_title}</strong>
                    <span>
                      {item.question_count} question{item.question_count === 1 ? "" : "s"}
                    </span>
                  </Link>
                  <button
                    type="button"
                    className="ghost-inline compact saved-item-remove"
                    disabled={removingId === item.id}
                    onClick={() => handleRemoveItem(item.id)}
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
            {!detail.items?.length ? <p className="saved-empty-hint">Add quizzes from a notebook using “Save to list”.</p> : null}
          </>
        ) : null}
      </div>
    );
  }

  return (
    <div className="page-shell saved-page">
      <header className="topbar">
        <VisionQBrandLink />
        <div className="topbar-actions">
          <Link className="ghost-pill" to="/">
            Notebooks
          </Link>
          <Link className="ghost-pill" to="/login">
            Sign in
          </Link>
        </div>
      </header>

      <div className="saved-screen-head">
        <h1 className="saved-screen-title">
          <Star size={22} className="saved-star" aria-hidden />
          Saved
        </h1>
      </div>

      {loading ? (
        <LoadingPanel />
      ) : error ? (
        <div className="saved-error">
          <p>{error}</p>
          <p className="saved-error-hint">Saved lists require an account when the API uses Supabase.</p>
          <Link className="primary-pill" to="/login">
            Sign in
          </Link>
        </div>
      ) : (
        <div className="saved-list-scroll">
          <div className="saved-card-stack">
            {lists.map((row) => (
              <button
                key={row.id}
                type="button"
                className="saved-collection-card"
                onClick={() => navigate(`/saved/${row.id}`)}
              >
                <FolderGlyph color={row.folder_color || FOLDER_COLORS[0]} />
                <div className="saved-collection-body">
                  <strong>{row.title}</strong>
                  <span>
                    {row.question_count || 0} question{(row.question_count || 0) === 1 ? "" : "s"} · {row.item_count || 0}{" "}
                    quiz{(row.item_count || 0) === 1 ? "" : "zes"}
                  </span>
                </div>
              </button>
            ))}
            <button type="button" className="saved-collection-card saved-collection-card--cta" onClick={() => setCreateOpen(true)}>
              <CreateListGlyph />
              <div className="saved-collection-body">
                <strong>Create list</strong>
                <span>Organize quizzes into a named collection</span>
              </div>
            </button>
          </div>
        </div>
      )}

      <CreateListModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={() => {
          refreshLists();
        }}
      />
    </div>
  );
}

export default SavedPage;
