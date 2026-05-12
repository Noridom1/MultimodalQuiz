import { useEffect, useState, useTransition } from "react";
import { LoaderCircle } from "lucide-react";
import { api } from "../../api";

const FOLDER_COLORS = ["#f0bf57", "#e37a46", "#9d6bff", "#5f6fff", "#4fb38a"];

export default function SaveToListModal({ open, notebookId, runId, onClose, onAdded }) {
  const [lists, setLists] = useState([]);
  const [selectedListId, setSelectedListId] = useState("");
  const [newTitle, setNewTitle] = useState("");
  const [newColor, setNewColor] = useState(FOLDER_COLORS[0]);
  const [mode, setMode] = useState("pick");
  const [pending, startTransition] = useTransition();
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    if (!open) return undefined;
    let active = true;
    setLoadError("");
    setMode("pick");
    setNewTitle("");
    setNewColor(FOLDER_COLORS[0]);
    setSelectedListId("");
    api
      .listQuizLists()
      .then((data) => {
        if (active) {
          setLists(data);
          if (data.length && data[0]?.id) {
            setSelectedListId(data[0].id);
          }
        }
      })
      .catch((err) => {
        if (active) setLoadError(err instanceof Error ? err.message : "Could not load lists");
      });
    return () => {
      active = false;
    };
  }, [open]);

  if (!open || !notebookId || !runId) {
    return null;
  }

  function handlePickSubmit(event) {
    event.preventDefault();
    if (!selectedListId) return;
    startTransition(async () => {
      try {
        await api.addQuizListItem(selectedListId, { notebook_id: notebookId, run_id: runId });
        onAdded?.();
        onClose();
      } catch (err) {
        alert(err instanceof Error ? err.message : "Could not save");
      }
    });
  }

  function handleCreateSubmit(event) {
    event.preventDefault();
    const trimmed = newTitle.trim();
    if (!trimmed) return;
    startTransition(async () => {
      try {
        const created = await api.createQuizList({ title: trimmed, folder_color: newColor });
        await api.addQuizListItem(created.id, { notebook_id: notebookId, run_id: runId });
        onAdded?.();
        onClose();
      } catch (err) {
        alert(err instanceof Error ? err.message : "Could not create list");
      }
    });
  }

  return (
    <div className="quiz-builder-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className="quiz-builder save-to-list-modal"
        role="dialog"
        aria-labelledby="save-to-list-title"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h2 id="save-to-list-title">Save to list</h2>
        <p className="save-to-list-sub">Choose a list or create a new one for this quiz.</p>

        {loadError ? (
          <p className="save-to-list-error">{loadError}</p>
        ) : null}

        <div className="save-to-list-tabs">
          <button
            type="button"
            className={`chip${mode === "pick" ? " active" : ""}`}
            onClick={() => setMode("pick")}
          >
            Existing
          </button>
          <button
            type="button"
            className={`chip${mode === "create" ? " active" : ""}`}
            onClick={() => setMode("create")}
          >
            New list
          </button>
        </div>

        {mode === "pick" ? (
          <form onSubmit={handlePickSubmit}>
            {lists.length ? (
              <label className="saved-create-label">
                List
                <select
                  className="saved-create-input"
                  value={selectedListId}
                  onChange={(e) => setSelectedListId(e.target.value)}
                >
                  {lists.map((row) => (
                    <option key={row.id} value={row.id}>
                      {row.title}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <p className="save-to-list-hint">You do not have any lists yet. Switch to “New list”.</p>
            )}
            <div className="saved-create-actions">
              <button type="button" className="ghost-pill" onClick={onClose}>
                Cancel
              </button>
              <button type="submit" className="primary-pill" disabled={pending || !lists.length}>
                {pending ? <LoaderCircle className="spin" size={18} /> : null}
                Save
              </button>
            </div>
          </form>
        ) : (
          <form onSubmit={handleCreateSubmit}>
            <label className="saved-create-label">
              List name
              <input
                className="saved-create-input"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                placeholder="e.g. Exam prep"
                maxLength={120}
              />
            </label>
            <div className="saved-color-chips" role="group" aria-label="Folder color">
              {FOLDER_COLORS.map((c) => (
                <button
                  key={c}
                  type="button"
                  className={`saved-color-chip${newColor === c ? " saved-color-chip--active" : ""}`}
                  style={{ background: c }}
                  onClick={() => setNewColor(c)}
                  aria-label={`Folder color ${c}`}
                />
              ))}
            </div>
            <div className="saved-create-actions">
              <button type="button" className="ghost-pill" onClick={onClose}>
                Cancel
              </button>
              <button type="submit" className="primary-pill" disabled={pending || !newTitle.trim()}>
                {pending ? <LoaderCircle className="spin" size={18} /> : null}
                Create and save
              </button>
            </div>
          </form>
        )}
      </section>
    </div>
  );
}
