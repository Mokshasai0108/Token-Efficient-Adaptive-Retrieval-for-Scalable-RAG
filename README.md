# TEAR — Token-Efficient Adaptive Retrieval for Scalable RAG

> A research-grade RAG system that adaptively retrieves documents, enforces token budgets,
> eliminates redundancy, and compresses context — maximizing accuracy per token.

---

## Architecture

```
Query
  │
  ▼
[1] Query Complexity Estimator     → C(q) ∈ [0,1] → label: simple/moderate/complex
  │
  ▼
[2] Adaptive k Selection           → k = f(C(q)) ∈ {3, 6, 10}
  │
  ▼
[3] Hybrid Retrieval               → FAISS (dense) + BM25 (sparse)
     score = α·dense + (1-α)·sparse
  │
  ▼
[4] Re-ranking                     → bi-encoder or cross-encoder
  │
  ▼
[5] Token Utility Scoring          → U(d) = Rel(d,q) / Tokens(d)
  │
  ▼
[6] Budget-Constrained Selection   → greedy knapsack, budget B
     maximize Σ Rel(d,q) s.t. Σ Tokens(d) ≤ B
  │
  ▼
[7] Redundancy Filtering           → cosine sim threshold, MMR-style
  │
  ▼
[8] Context Compression            → extractive sentence scoring
  │
  ▼
[9] LLM Generation                 → Llama-3-70B-Instruct (4-bit NF4)
  │
  ▼
Answer + Pipeline Telemetry
```

---

## Running Evaluation

```bash
# Compare TEAR vs Standard RAG vs No-RAG
# Results saved to results_*.json
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{"n_samples": 500, "systems": ["tear", "standard_rag", "no_rag"]}'
```

---

## Running Ablation Study

```bash
python run_ablation.py --n-samples 300
# Results saved to ablation_results.json
```

Expected output format:

```
System                              EM      F1   Tokens        Eff       Δ F1
────────────────────────────────────────────────────────────────────────────────
Full TEAR (all modules)         0.4821  0.6103    342.1   0.001784         —
TEAR − Adaptive k               0.4712  0.5987    510.3   0.001173    -0.0116
TEAR − Token Budget             0.4690  0.5934    891.4   0.000665    -0.0169
TEAR − Redundancy Filter        0.4755  0.6044    398.2   0.001518    -0.0059
TEAR − Compression              0.4801  0.6089    612.8   0.000994    -0.0014
Standard RAG (fixed k=5)        0.4501  0.5712    820.5   0.000696    -0.0391
No RAG (LLM only)               0.3820  0.4930      0.0          —    -0.1173
```

---

## Configuration

Edit `config.py` to tune all system parameters:

```python
# Key settings
token_budget = 1024        # max tokens in context
k_simple = 3               # k for simple queries
k_moderate = 6             # k for moderate queries
k_complex = 10             # k for complex queries
hybrid_alpha = 0.6         # dense vs sparse weight
redundancy_threshold = 0.85  # similarity cutoff
compression_ratio = 0.6    # keep top 60% sentences
```

---

## Datasets

| Dataset | Type | Size | Use |
|---|---|---|---|
| NaturalQuestions | Open-domain QA | 307K | Complex, real Google queries |
| TriviaQA | Trivia QA | 650K | Multi-hop, diverse topics |
| SQuAD v2 | Reading comprehension | 130K | Precise extraction + unanswerable |

---

```
