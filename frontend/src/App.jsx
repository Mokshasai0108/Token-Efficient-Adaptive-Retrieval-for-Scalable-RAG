import { useState, useEffect, useRef } from "react";
import QueryPanel from "./components/QueryPanel";
import ResultPanel from "./components/ResultPanel";
import StatusBar, { MetricsDashboard } from "./components/StatusBar";
import "./index.css";

const API = "http://localhost:8000/api";

export default function App() {
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("query");
  const [history, setHistory] = useState([]);
  const [streamText, setStreamText] = useState("");
  const [streaming, setStreaming] = useState(false);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API}/status`);
      const data = await res.json();
      setStatus(data);
    } catch {
      setStatus(null);
    }
  };

  const handleQuery = async (query, tokenBudget, useStream) => {
    setLoading(true);
    setResult(null);
    setStreamText("");

    if (useStream) {
      setStreaming(true);
      setLoading(false);
      const evtSource = new EventSource(
        `${API}/stream?q=${encodeURIComponent(query)}`
      );
      let fullText = "";
      evtSource.onmessage = (e) => {
        if (e.data === "[DONE]") {
          evtSource.close();
          setStreaming(false);
          return;
        }
        const parsed = JSON.parse(e.data);
        if (parsed.token) {
          fullText += parsed.token;
          setStreamText(fullText);
        }
      };
      evtSource.onerror = () => {
        evtSource.close();
        setStreaming(false);
      };
      return;
    }

    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, token_budget: tokenBudget }),
      });
      const data = await res.json();
      setResult(data);
      setHistory((prev) => [{ query, result: data, ts: Date.now() }, ...prev.slice(0, 19)]);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleIndex = async (maxDocs) => {
    await fetch(`${API}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_docs: maxDocs }),
    });
    setTimeout(fetchStatus, 2000);
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-left">
          <div className="logo-mark">T</div>
          <div className="header-titles">
            <h1 className="app-title">TEAR</h1>
            <span className="app-subtitle">Token-Efficient Adaptive Retrieval</span>
          </div>
        </div>
        <nav className="app-nav">
          {["query", "metrics", "history"].map((tab) => (
            <button
              key={tab}
              className={`nav-btn ${activeTab === tab ? "active" : ""}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </nav>
        <StatusBar status={status} onIndex={handleIndex} />
      </header>

      <main className="app-main">
        {activeTab === "query" && (
          <div className="query-layout">
            <QueryPanel
              onSubmit={handleQuery}
              loading={loading}
              streaming={streaming}
              disabled={!status?.index_ready}
            />
            <ResultPanel
              result={result}
              loading={loading}
              streamText={streamText}
              streaming={streaming}
            />
          </div>
        )}
        {activeTab === "metrics" && (
          <MetricsDashboard history={history} />
        )}
        {activeTab === "history" && (
          <HistoryPanel history={history} onSelect={(r) => {
            setResult(r.result);
            setActiveTab("query");
          }} />
        )}
      </main>
    </div>
  );
}

function HistoryPanel({ history, onSelect }) {
  if (!history.length)
    return <div className="empty-state">No queries yet. Ask something!</div>;

  return (
    <div className="history-panel">
      <h2 className="panel-title">Query History</h2>
      <div className="history-list">
        {history.map((item) => (
          <div key={item.ts} className="history-item" onClick={() => onSelect(item)}>
            <div className="history-query">{item.query}</div>
            <div className="history-meta">
              <span className={`complexity-badge ${item.result.complexity_label}`}>
                {item.result.complexity_label}
              </span>
              <span className="history-tokens">{item.result.tokens_in_context} tokens</span>
              <span className="history-time">{new Date(item.ts).toLocaleTimeString()}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
