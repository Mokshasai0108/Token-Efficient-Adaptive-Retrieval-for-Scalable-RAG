import { useState } from "react";

const EXAMPLE_QUERIES = [
  "Who wrote Hamlet?",
  "How does photosynthesis convert sunlight into energy?",
  "Compare the causes and consequences of World War I and World War II.",
  "Why did the 2008 financial crisis lead to a global recession and how did governments respond?",
  "What is the capital of France?",
];

export default function QueryPanel({ onSubmit, loading, streaming, disabled }) {
  const [query, setQuery] = useState("");
  const [tokenBudget, setTokenBudget] = useState(1024);
  const [useStream, setUseStream] = useState(false);

  const handleSubmit = () => {
    if (!query.trim() || loading || streaming) return;
    onSubmit(query.trim(), tokenBudget, useStream);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSubmit();
  };

  return (
    <div className="query-panel">
      <div className="panel-header">
        <h2 className="panel-title">Query</h2>
        <span className="panel-hint">⌘↵ to submit</span>
      </div>

      <textarea
        className="query-input"
        placeholder="Ask anything… TEAR will adapt retrieval depth to your query's complexity."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKey}
        rows={5}
        disabled={disabled}
      />

      <div className="controls-row">
        <div className="control-group">
          <label className="control-label">Token Budget</label>
          <div className="budget-control">
            <input
              type="range"
              min={128}
              max={4096}
              step={128}
              value={tokenBudget}
              onChange={(e) => setTokenBudget(Number(e.target.value))}
              className="budget-slider"
            />
            <span className="budget-value">{tokenBudget}</span>
          </div>
        </div>

        <div className="control-group">
          <label className="control-label">Streaming</label>
          <button
            className={`toggle-btn ${useStream ? "on" : "off"}`}
            onClick={() => setUseStream(!useStream)}
          >
            {useStream ? "On" : "Off"}
          </button>
        </div>
      </div>

      <button
        className={`submit-btn ${loading || streaming ? "loading" : ""}`}
        onClick={handleSubmit}
        disabled={!query.trim() || loading || streaming || disabled}
      >
        {loading ? (
          <span className="btn-loading"><span className="spinner" />Processing…</span>
        ) : streaming ? (
          <span className="btn-loading"><span className="spinner" />Streaming…</span>
        ) : (
          "Run TEAR →"
        )}
      </button>

      {disabled && (
        <div className="warning-banner">
          ⚠ Index not ready. Use the Index button in the header to build it first.
        </div>
      )}

      <div className="examples-section">
        <div className="examples-label">Try an example:</div>
        <div className="examples-list">
          {EXAMPLE_QUERIES.map((q) => (
            <button
              key={q}
              className="example-chip"
              onClick={() => setQuery(q)}
              disabled={disabled}
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
