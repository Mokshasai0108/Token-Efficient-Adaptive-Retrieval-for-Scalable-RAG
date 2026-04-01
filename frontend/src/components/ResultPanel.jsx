import { useState } from "react";

export default function ResultPanel({ result, loading, streamText, streaming }) {
  const [showPassages, setShowPassages] = useState(false);

  if (loading) return (
    <div className="result-panel loading-state">
      <div className="pipeline-animation">
        {[
          "Estimating complexity…",
          "Retrieving documents…",
          "Re-ranking…",
          "Applying token budget…",
          "Filtering redundancy…",
          "Compressing context…",
          "Generating answer…",
        ].map((step, i) => (
          <div key={i} className="pipeline-step" style={{ animationDelay: `${i * 0.4}s` }}>
            <div className="step-dot" />
            <span>{step}</span>
          </div>
        ))}
      </div>
    </div>
  );

  if (streaming) return (
    <div className="result-panel">
      <div className="panel-header">
        <h2 className="panel-title">Answer <span className="streaming-badge">● Live</span></h2>
      </div>
      <div className="answer-box streaming">
        <p>{streamText}<span className="cursor-blink">|</span></p>
      </div>
    </div>
  );

  if (!result) return (
    <div className="result-panel empty-result">
      <div className="empty-icon">◎</div>
      <p>Results will appear here after you run a query.</p>
    </div>
  );

  const tokenSavingsPct = result.tokens_retrieved > 0
    ? Math.round((1 - result.tokens_in_context / result.tokens_retrieved) * 100)
    : 0;

  return (
    <div className="result-panel">
      <div className="panel-header">
        <h2 className="panel-title">Answer</h2>
        <span className="latency-badge">{result.latency_seconds}s</span>
      </div>

      <div className="answer-box">
        <p>{result.answer}</p>
      </div>

      {/* Pipeline Telemetry */}
      <div className="telemetry-grid">
        <TelemetryCard
          label="Complexity"
          value={result.complexity_label.toUpperCase()}
          sub={`score: ${result.complexity_score}`}
          accent={
            result.complexity_label === "simple" ? "green" :
            result.complexity_label === "moderate" ? "amber" : "red"
          }
        />
        <TelemetryCard
          label="k Retrieved"
          value={result.k_used}
          sub={`${result.docs_in_context} in context`}
        />
        <TelemetryCard
          label="Tokens Used"
          value={result.tokens_in_context}
          sub={`budget: ${result.token_budget}`}
        />
        <TelemetryCard
          label="Tokens Saved"
          value={`${tokenSavingsPct}%`}
          sub={`${result.tokens_saved} tokens`}
          accent="green"
        />
        <TelemetryCard
          label="Prompt Tokens"
          value={result.prompt_tokens}
          sub={`total: ${result.total_tokens}`}
        />
        <TelemetryCard
          label="Latency"
          value={`${result.latency_seconds}s`}
          sub="end-to-end"
        />
      </div>

      {/* Complexity Feature Breakdown */}
      <div className="features-section">
        <div className="features-header">Complexity Features</div>
        <div className="features-bars">
          {Object.entries(result.complexity_features).map(([key, val]) => (
            <div key={key} className="feature-row">
              <span className="feature-label">
                {key.replace(/_/g, " ")}
              </span>
              <div className="feature-bar-track">
                <div
                  className="feature-bar-fill"
                  style={{ width: `${Math.round(val * 100)}%` }}
                />
              </div>
              <span className="feature-val">{(val * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Passages */}
      <div className="passages-section">
        <button
          className="passages-toggle"
          onClick={() => setShowPassages(!showPassages)}
        >
          {showPassages ? "▾" : "▸"} Retrieved Passages ({result.passages?.length})
        </button>
        {showPassages && (
          <div className="passages-list">
            {result.passages?.map((p, i) => (
              <div key={i} className="passage-card">
                <div className="passage-meta">
                  <span className="passage-source">{p.source}</span>
                  <span className="passage-score">score: {p.score}</span>
                  <span className="passage-tokens">{p.token_count} tokens</span>
                </div>
                <p className="passage-text">{p.text}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function TelemetryCard({ label, value, sub, accent }) {
  return (
    <div className={`telemetry-card ${accent || ""}`}>
      <div className="tc-label">{label}</div>
      <div className="tc-value">{value}</div>
      <div className="tc-sub">{sub}</div>
    </div>
  );
}
