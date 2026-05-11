import { LoaderCircle, Sparkles } from "lucide-react";

function QuizBuilderModal({
  open,
  pendingRun,
  questions,
  selectedSourceId,
  sources,
  setQuestions,
  setSelectedSourceId,
  onClose,
  onCreate,
}) {
  if (!open) {
    return null;
  }

  return (
    <div className="quiz-builder-backdrop" onClick={onClose}>
      <section className="quiz-builder" onClick={(event) => event.stopPropagation()}>
        <div className="quiz-builder-header">
          <div>
            <span className="eyebrow">Quiz setup</span>
            <h2>Create a quiz</h2>
          </div>
          <button className="ghost-inline quiz-builder-close" onClick={onClose} type="button">
            Close
          </button>
        </div>
        <div className="quiz-builder-grid">
          <label className="field">
            <span>Document</span>
            <select value={selectedSourceId} onChange={(event) => setSelectedSourceId(event.target.value)}>
              {sources.map((source) => (
                <option key={source.id} value={source.id}>
                  {source.title}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Questions</span>
            <input
              type="number"
              min="1"
              max="20"
              value={questions}
              onChange={(event) => setQuestions(Number(event.target.value) || 1)}
            />
          </label>
        </div>
        <div className="quiz-builder-actions">
          <button className="ghost-pill" type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            className="primary-pill"
            type="button"
            onClick={onCreate}
            disabled={!selectedSourceId || pendingRun}
          >
            {pendingRun ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}
            Create
          </button>
        </div>
      </section>
    </div>
  );
}

export default QuizBuilderModal;
