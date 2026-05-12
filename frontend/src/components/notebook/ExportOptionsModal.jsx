import { useEffect, useState } from "react";
import { Download, FileArchive, FileText, LoaderCircle } from "lucide-react";

function ExportOptionsModal({ open, pending, defaultFormat, defaultDataFormat, onCancel, onConfirm }) {
  const [format, setFormat] = useState(defaultFormat || "zip");
  const [dataFormat, setDataFormat] = useState(defaultDataFormat || "json");

  useEffect(() => {
    if (open) {
      setFormat(defaultFormat || "zip");
      setDataFormat(defaultDataFormat || "json");
    }
  }, [open, defaultFormat, defaultDataFormat]);

  if (!open) {
    return null;
  }

  const handleSubmit = (event) => {
    event.preventDefault();
    onConfirm({ format, dataFormat });
  };

  return (
    <div className="quiz-builder-backdrop" onClick={pending ? undefined : onCancel}>
      <section
        className="quiz-builder export-modal"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="export-options-title"
      >
        <div className="quiz-builder-header">
          <div>
            <span className="eyebrow">Export quiz</span>
            <h2 id="export-options-title">Choose a format</h2>
          </div>
          <button
            className="ghost-inline quiz-builder-close"
            type="button"
            onClick={onCancel}
            disabled={pending}
          >
            Close
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <fieldset className="export-format-group" disabled={pending}>
            <legend className="export-format-legend">File type</legend>
            <div className="export-option-grid">
              <label className={`export-option-card${format === "pdf" ? " is-selected" : ""}`}>
                <input
                  type="radio"
                  name="export-format"
                  value="pdf"
                  checked={format === "pdf"}
                  onChange={() => setFormat("pdf")}
                />
                <FileText size={20} aria-hidden />
                <div>
                  <strong>PDF</strong>
                  <span>Printable document with images, options, answers and explanations.</span>
                </div>
              </label>
              <label className={`export-option-card${format === "zip" ? " is-selected" : ""}`}>
                <input
                  type="radio"
                  name="export-format"
                  value="zip"
                  checked={format === "zip"}
                  onChange={() => setFormat("zip")}
                />
                <FileArchive size={20} aria-hidden />
                <div>
                  <strong>ZIP archive</strong>
                  <span>Question data plus the original images.</span>
                </div>
              </label>
            </div>
          </fieldset>

          {format === "zip" ? (
            <fieldset className="export-format-group export-format-group--sub" disabled={pending}>
              <legend className="export-format-legend">Questions file format</legend>
              <div className="export-radio-row">
                <label className={`export-pill${dataFormat === "json" ? " is-selected" : ""}`}>
                  <input
                    type="radio"
                    name="export-data-format"
                    value="json"
                    checked={dataFormat === "json"}
                    onChange={() => setDataFormat("json")}
                  />
                  <span>JSON</span>
                </label>
                <label className={`export-pill${dataFormat === "csv" ? " is-selected" : ""}`}>
                  <input
                    type="radio"
                    name="export-data-format"
                    value="csv"
                    checked={dataFormat === "csv"}
                    onChange={() => setDataFormat("csv")}
                  />
                  <span>CSV</span>
                </label>
              </div>
            </fieldset>
          ) : null}

          <div className="quiz-builder-actions">
            <button className="ghost-pill" type="button" onClick={onCancel} disabled={pending}>
              Cancel
            </button>
            <button className="primary-pill" type="submit" disabled={pending}>
              {pending ? <LoaderCircle className="spin" size={18} /> : <Download size={18} />}
              Export
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}

export default ExportOptionsModal;
