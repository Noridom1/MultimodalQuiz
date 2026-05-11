import { ArrowRight, CheckCircle2, Upload } from "lucide-react";
import { formatBytes } from "../../utils/ui";

function SourcePanel({
  pendingUpload,
  selectedSourceId,
  sources,
  onUploadFile,
  setSelectedSourceId,
}) {
  return (
    <section className="panel source-panel">
      <div className="panel-header">
        <h2>Sources</h2>
      </div>
      <label className="upload-cta">
        <input type="file" hidden onChange={onUploadFile} />
        <Upload size={18} />
        {pendingUpload ? "Uploading..." : "Add source"}
      </label>
      <div className="source-list-scroll">
        <div className="source-list">
          {sources.map((source) => (
            <button
              key={source.id}
              className={`source-item ${selectedSourceId === source.id ? "selected" : ""}`}
              onClick={() => setSelectedSourceId(source.id)}
            >
              <div>
                <strong>{source.title}</strong>
                <span>{formatBytes(source.size_bytes)} - {source.content_type}</span>
              </div>
              {selectedSourceId === source.id ? <CheckCircle2 size={18} /> : <ArrowRight size={18} />}
            </button>
          ))}
          {sources.length === 0 ? (
            <div className="empty-panel">
              <p>Saved sources will appear here.</p>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

export default SourcePanel;
