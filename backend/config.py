# ============================================================
# TEAR - System Configuration
# ============================================================

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TEARConfig:
    # ── LLM Backend ──────────────────────────────────────────
    llm_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_load_in_4bit: bool = False           # quantize for local GPU
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.1
    llm_device_map: str = "auto"            # auto-splits across GPUs/CPU

    # ── Embedding Model ───────────────────────────────────────
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_batch_size: int = 64

    # ── Vector Stores ─────────────────────────────────────────
    vector_store: str = "faiss"             # "faiss" | "chroma"
    faiss_index_path: str = "data/faiss_index"
    chroma_persist_dir: str = "data/chroma_db"
    chroma_collection: str = "tear_docs"

    # ── Retrieval ─────────────────────────────────────────────
    use_hybrid: bool = True                 # BM25 + dense
    hybrid_alpha: float = 0.6              # weight for dense score
    min_k: int = 2
    max_k: int = 12

    # ── Complexity Estimator ──────────────────────────────────
    spacy_model: str = "en_core_web_sm"
    complexity_thresholds: tuple = (0.35, 0.65)
    k_simple: int = 3
    k_moderate: int = 6
    k_complex: int = 10

    # ── Token Budget ──────────────────────────────────────────
    token_budget: int = 1024               # max context tokens
    rerank_top_n: int = 20                 # candidates before budget cut

    # ── Redundancy Filtering ──────────────────────────────────
    redundancy_threshold: float = 0.85     # cosine similarity cutoff

    # ── Context Compression ───────────────────────────────────
    compression_ratio: float = 0.6        # keep top 60% sentences
    min_sentence_tokens: int = 10

    # ── Datasets ──────────────────────────────────────────────
    datasets_to_load: list = field(default_factory=lambda: [
        "natural_questions", "trivia_qa", "squad_v2"
    ])
    max_docs_per_dataset: int = 50_000     # limit for indexing

    # ── API ───────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list = field(default_factory=lambda: ["http://localhost:3000"])

    # ── HuggingFace ───────────────────────────────────────────
    hf_token: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_TOKEN")
    )


# Singleton config instance
config = TEARConfig()
