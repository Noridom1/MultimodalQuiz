import { LoaderCircle, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

function QuizBuilderModal({
  open,
  pendingRun,
  questions,
  selectedQuestionTypes,
  selectedSourceId,
  sources,
  setQuestions,
  setSelectedQuestionTypes,
  setSelectedSourceId,
  questionTypeLabels,
  onClose,
  onCreate,
}) {
  const [questionDraft, setQuestionDraft] = useState(String(questions ?? 1));

  useEffect(() => {
    setQuestionDraft(String(questions ?? 1));
  }, [questions]);

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
              value={questionDraft}
              onChange={(event) => {
                const next = event.target.value;
                setQuestionDraft(next);
                if (next === "") return;
                const parsed = Number.parseInt(next, 10);
                if (Number.isNaN(parsed)) return;
                const clamped = Math.min(20, Math.max(1, parsed));
                setQuestions(clamped);
              }}
              onBlur={() => {
                const parsed = Number.parseInt(String(questionDraft), 10);
                const clamped = Number.isNaN(parsed) ? 1 : Math.min(20, Math.max(1, parsed));
                setQuestions(clamped);
                setQuestionDraft(String(clamped));
              }}
            />
          </label>
          <fieldset className="field quiz-type-fieldset">
            <legend>Question types</legend>
            <div className="quiz-type-options">
              {Object.entries(questionTypeLabels || {}).map(([id, label]) => {
                const checked = Array.isArray(selectedQuestionTypes) && selectedQuestionTypes.includes(id);
                return (
                  <label key={id} className="quiz-type-option">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(event) => {
                        setSelectedQuestionTypes((current) => {
                          const list = Array.isArray(current) ? current : [];
                          if (event.target.checked) {
                            return list.includes(id) ? list : [...list, id];
                          }
                          return list.filter((item) => item !== id);
                        });
                      }}
                    />
                    <span>{label}</span>
                  </label>
                );
              })}
            </div>
          </fieldset>
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
