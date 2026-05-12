import { CheckCircle2, FileText, Lightbulb, Plus } from "lucide-react";
import { formatBytes } from "../../utils/ui";

function sourceTypeLabel(contentType) {
  const c = String(contentType || "").toLowerCase();
  if (c.includes("markdown")) return "Text/Markdown";
  if (c.includes("plain")) return "Text";
  if (c.includes("pdf")) return "PDF";
  if (c.includes("html")) return "HTML";
  if (c.startsWith("image/")) return "Image";
  return contentType || "File";
}

function sourceFileBadge(contentType) {
  const c = String(contentType || "").toLowerCase();
  if (c.includes("pdf")) return "PDF";
  if (c.includes("markdown") || c.includes("text")) return "TXT";
  if (c.startsWith("image/")) return "IMG";
  return "FILE";
}

function SourcePanel({
  pendingUpload,
  selectedSourceId,
  sources,
  onUploadFile,
  setSelectedSourceId,
}) {
  return (
    <section className="panel source-panel">
      <div className="panel-header panel-header--notebook">
        <div className="panel-header-icon-wrap" aria-hidden>
          <FileText size={22} strokeWidth={1.75} />
        </div>
        <div className="panel-header-text">
          <h2>Sources</h2>
          <p className="panel-subtitle">Add reference materials for your quiz.</p>
        </div>
      </div>
      <label className="upload-cta notebook-add-source">
        <input type="file" hidden onChange={onUploadFile} />
        <Plus size={18} strokeWidth={2.5} />
        {pendingUpload ? "Uploading…" : "Add source"}
      </label>
      <div className="source-list-scroll">
        <div className="source-list">
          {sources.map((source) => (
            <button
              key={source.id}
              type="button"
              className={`source-item ${selectedSourceId === source.id ? "selected" : ""}`}
              onClick={() => setSelectedSourceId(source.id)}
            >
              <span className="source-file-badge">{sourceFileBadge(source.content_type)}</span>
              <div className="source-item-copy">
                <strong>{source.title}</strong>
                <span>
                  {formatBytes(source.size_bytes)} • {sourceTypeLabel(source.content_type)}
                </span>
              </div>
              {selectedSourceId === source.id ? (
                <CheckCircle2 className="source-item-check" size={22} aria-hidden />
              ) : null}
            </button>
          ))}
          {sources.length === 0 ? (
            <div className="empty-panel">
              <p>Saved sources will appear here.</p>
            </div>
          ) : null}
        </div>
      </div>
      <div className="source-tip-box">
        <Lightbulb size={20} className="source-tip-icon" aria-hidden />
        <div>
          <strong className="source-tip-title">Tip</strong>
          <p className="source-tip-text">
            Add more sources to generate better quizzes with diverse content.
          </p>
        </div>
      </div>
    </section>
  );
}

export default SourcePanel;
