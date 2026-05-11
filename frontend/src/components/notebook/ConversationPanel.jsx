import { Bot, LoaderCircle, SendHorizontal } from "lucide-react";

function ConversationPanel({
  message,
  messages,
  pendingMessage,
  setMessage,
  onSendMessage,
}) {
  return (
    <section className="panel conversation-panel">
      <div className="panel-header">
        <h2>Conversation</h2>
      </div>
      <div className="conversation-scroll">
        {messages.length <= 1 ? (
          <div className="conversation-hero">
            <div className="hero-icon">*</div>
            <h1>Start building your quiz notebook</h1>
            <p>
              Upload source material, then generate a multimodal quiz package with image-backed
              question cards.
            </p>
            <div className="prompt-chips">
              <button type="button" className="prompt-chip" onClick={() => setMessage("Summarize the current sources")}>
                Summarize sources
              </button>
              <button type="button" className="prompt-chip" onClick={() => setMessage("What does the latest quiz cover?")}>
                Review latest run
              </button>
              <button type="button" className="prompt-chip" onClick={() => setMessage("How many source files are ready?")}>
                Check readiness
              </button>
            </div>
          </div>
        ) : null}

        <div className="message-list">
          {messages.map((entry) => {
            const isAssistant = entry.role === "assistant";
            const isSystem = entry.role === "system" || entry.kind === "system";

            if (isSystem) {
              return (
                <article key={entry.id} className="message-row message-row--system">
                  <div className="message-bubble message-bubble--system">
                    <p>{entry.content}</p>
                  </div>
                </article>
              );
            }

            return (
              <article key={entry.id} className={`message-row message-row--${entry.role}`}>
                <div className={`message-bubble message-bubble--${entry.role}`}>
                  {isAssistant ? (
                    <div className="message-bubble-head" aria-hidden>
                      <Bot size={16} />
                    </div>
                  ) : null}
                  <p>{entry.content}</p>
                </div>
              </article>
            );
          })}
        </div>
      </div>
      <form className="composer" onSubmit={onSendMessage}>
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
  );
}

export default ConversationPanel;
