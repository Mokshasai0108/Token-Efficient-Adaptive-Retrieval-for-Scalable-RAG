// ── StatusBar ─────────────────────────────────────────────────

import { useState } from "react";

export default function StatusBar({ status, onIndex }) {
  const [showIndexModal, setShowIndexModal] = useState(false);
  const [maxDocs, setMaxDocs] = useState(50000);

  const dot = status?.index_ready ? "green" : "amber";

  return (
    <div className="status-bar">
      <div className={`status-dot ${dot}`} />
      <span className="status-text">
        {!status
          ? "Connecting…"
          : status.index_ready
          ? `${status.index_doc_count.toLocaleString()} docs indexed`
          : "Index not built"}
      </span>

      <button
        className="index-btn"
        onClick={() => setShowIndexModal(true)}
      >
        {status?.indexing ? "Indexing…" : "⊕ Index"}
      </button>

      {showIndexModal && (
        <div className="modal-overlay" onClick={() => setShowIndexModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3 className="modal-title">Build Search Index</h3>
            <p className="modal-desc">
              This will download and index NaturalQuestions, TriviaQA, and SQuAD v2.
              This may take 10–30 minutes depending on hardware.
            </p>
            <div className="modal-control">
              <label>Max docs per dataset</label>
              <input
                type="number"
                value={maxDocs}
                onChange={(e) => setMaxDocs(Number(e.target.value))}
                min={1000}
                max={200000}
                step={1000}
              />
            </div>
            <div className="modal-actions">
              <button className="modal-cancel" onClick={() => setShowIndexModal(false)}>
                Cancel
              </button>
              <button
                className="modal-confirm"
                onClick={() => {
                  onIndex(maxDocs);
                  setShowIndexModal(false);
                }}
              >
                Start Indexing
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


// ── MetricsDashboard ──────────────────────────────────────────

export function MetricsDashboard({ history }) {
  if (!history.length)
    return (
      <div className="metrics-panel empty-state">
        <p>Run queries to see live metrics here.</p>
      </div>
    );

  const avgTokens = Math.round(
    history.reduce((s, h) => s + h.result.tokens_in_context, 0) / history.length
  );
  const avgSaved = Math.round(
    history.reduce((s, h) => s + (h.result.tokens_saved || 0), 0) / history.length
  );
  const avgLatency = (
    history.reduce((s, h) => s + h.result.latency_seconds, 0) / history.length
  ).toFixed(2);

  const complexityDist = { simple: 0, moderate: 0, complex: 0 };
  history.forEach((h) => complexityDist[h.result.complexity_label]++);

  const maxK = Math.max(...history.map((h) => h.result.k_used));
  const kDist = {};
  history.forEach((h) => {
    kDist[h.result.k_used] = (kDist[h.result.k_used] || 0) + 1;
  });

  return (
    <div className="metrics-panel">
      <h2 className="panel-title">Live Metrics Dashboard</h2>

      <div className="metrics-summary-grid">
        <SummaryCard label="Queries Run" value={history.length} />
        <SummaryCard label="Avg Tokens Used" value={avgTokens} />
        <SummaryCard label="Avg Tokens Saved" value={avgSaved} accent="green" />
        <SummaryCard label="Avg Latency" value={`${avgLatency}s`} />
      </div>

      <div className="charts-row">
        <div className="chart-card">
          <div className="chart-title">Complexity Distribution</div>
          <div className="dist-bars">
            {Object.entries(complexityDist).map(([label, count]) => (
              <div key={label} className="dist-row">
                <span className={`complexity-badge ${label}`}>{label}</span>
                <div className="dist-track">
                  <div
                    className={`dist-fill ${label}`}
                    style={{ width: `${(count / history.length) * 100}%` }}
                  />
                </div>
                <span className="dist-count">{count}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="chart-card">
          <div className="chart-title">Adaptive k Distribution</div>
          <div className="dist-bars">
            {Object.entries(kDist).sort((a, b) => Number(a[0]) - Number(b[0])).map(([k, count]) => (
              <div key={k} className="dist-row">
                <span className="k-label">k={k}</span>
                <div className="dist-track">
                  <div
                    className="dist-fill moderate"
                    style={{ width: `${(count / history.length) * 100}%` }}
                  />
                </div>
                <span className="dist-count">{count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="history-tokens-chart">
        <div className="chart-title">Token Usage Per Query</div>
        <div className="bar-chart">
          {history.slice(0, 20).reverse().map((h, i) => (
            <div key={i} className="bar-wrapper" title={h.query}>
              <div className="bar-group">
                <div
                  className="bar retrieved"
                  style={{ height: `${Math.min((h.result.tokens_retrieved / 1500) * 100, 100)}%` }}
                  title={`Retrieved: ${h.result.tokens_retrieved}`}
                />
                <div
                  className="bar used"
                  style={{ height: `${Math.min((h.result.tokens_in_context / 1500) * 100, 100)}%` }}
                  title={`Used: ${h.result.tokens_in_context}`}
                />
              </div>
              <div className="bar-label">{i + 1}</div>
            </div>
          ))}
        </div>
        <div className="bar-legend">
          <span className="legend-item"><span className="legend-dot retrieved" />Retrieved</span>
          <span className="legend-item"><span className="legend-dot used" />In Context</span>
        </div>
      </div>
    </div>
  );
}

function SummaryCard({ label, value, accent }) {
  return (
    <div className={`summary-card ${accent || ""}`}>
      <div className="sc-value">{value}</div>
      <div className="sc-label">{label}</div>
    </div>
  );
}
