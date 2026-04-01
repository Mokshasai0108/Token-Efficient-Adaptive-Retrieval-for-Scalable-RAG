"""
TEAR — Module 2 + 3: Adaptive Retrieval Engine
Supports FAISS and ChromaDB with optional BM25 hybrid retrieval.
Dynamic k is fed from ComplexityResult.
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

import faiss
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from loguru import logger

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


@dataclass
class RetrievedDocument:
    doc_id: str
    text: str
    score: float                # dense similarity score
    sparse_score: float = 0.0  # BM25 score
    hybrid_score: float = 0.0  # combined
    token_count: int = 0
    source: str = ""            # dataset origin


class AdaptiveRetrievalEngine:
    """
    Module 2: Adaptive k is received from ComplexityEstimator.
    Module 3: Retrieval using FAISS (dense) + BM25 (sparse) hybrid.

    Hybrid score formula:
        hybrid = alpha * dense_norm + (1-alpha) * sparse_norm
    """

    def __init__(self, config):
        self.config = config
        self.alpha = config.hybrid_alpha
        self.use_hybrid = config.use_hybrid

        # Embedding model
        logger.info(f"Loading SentenceTransformer: {config.embed_model_id}")
        self.encoder = SentenceTransformer(config.embed_model_id)

        # State
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_sources: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.chroma_collection = None

        # Initialize vector store
        if config.vector_store == "chroma" and CHROMA_AVAILABLE:
            self._init_chroma()
        else:
            if config.vector_store == "chroma" and not CHROMA_AVAILABLE:
                logger.warning("ChromaDB not available, falling back to FAISS")
            self._init_faiss_placeholder()

    # ── Initialization ─────────────────────────────────────────

    def _init_chroma(self):
        logger.info("Initializing ChromaDB...")
        client = chromadb.PersistentClient(path=self.config.chroma_persist_dir)
        self.chroma_collection = client.get_or_create_collection(
            name=self.config.chroma_collection,
            metadata={"hnsw:space": "cosine"}
        )

    def _init_faiss_placeholder(self):
        """FAISS index created after documents are indexed."""
        pass

    # ── Indexing ───────────────────────────────────────────────

    def index_documents(
        self,
        texts: List[str],
        doc_ids: List[str],
        sources: List[str],
        batch_size: int = 256,
    ):
        """
        Encode and index all documents.
        Builds FAISS index + BM25 index in memory.
        """
        logger.info(f"Indexing {len(texts)} documents...")
        self.documents = texts
        self.doc_ids = doc_ids
        self.doc_sources = sources

        # ── Dense Embeddings ──────────────────────────────────
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embs = self.encoder.encode(
                batch, convert_to_numpy=True,
                show_progress_bar=False, normalize_embeddings=True
            )
            all_embeddings.append(embs)
            if i % 5000 == 0:
                logger.info(f"  Encoded {i}/{len(texts)}")

        self.embeddings = np.vstack(all_embeddings).astype("float32")

        # ── Build FAISS Index ─────────────────────────────────
        dim = self.embeddings.shape[1]
        if len(texts) > 100_000:
            # IVF index for large collections
            quantizer = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, 256)
            self.faiss_index.train(self.embeddings)
        else:
            # Flat exact search for smaller collections
            self.faiss_index = faiss.IndexFlatIP(dim)

        self.faiss_index.add(self.embeddings)
        logger.info(f"FAISS index built: {self.faiss_index.ntotal} vectors")

        # Save FAISS index
        os.makedirs(os.path.dirname(self.config.faiss_index_path) or ".", exist_ok=True)
        faiss.write_index(self.faiss_index, self.config.faiss_index_path + ".bin")

        # Save metadata
        with open(self.config.faiss_index_path + "_meta.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "doc_ids": self.doc_ids,
                "doc_sources": self.doc_sources,
            }, f)

        # ── Build BM25 Index ──────────────────────────────────
        if self.use_hybrid:
            logger.info("Building BM25 index...")
            tokenized = [doc.lower().split() for doc in texts]
            self.bm25 = BM25Okapi(tokenized)
            logger.info("BM25 index ready")

        # ── ChromaDB Indexing ─────────────────────────────────
        if self.chroma_collection is not None:
            logger.info("Indexing into ChromaDB...")
            batch_size = 1000
            for i in range(0, len(texts), batch_size):
                self.chroma_collection.add(
                    documents=texts[i:i + batch_size],
                    ids=doc_ids[i:i + batch_size],
                    embeddings=self.embeddings[i:i + batch_size].tolist(),
                    metadatas=[{"source": s} for s in sources[i:i + batch_size]],
                )

        logger.info("Indexing complete.")

    def load_index(self):
        """Load pre-built FAISS index from disk."""
        index_path = self.config.faiss_index_path + ".bin"
        meta_path = self.config.faiss_index_path + "_meta.pkl"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        self.faiss_index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.documents = meta["documents"]
        self.doc_ids = meta["doc_ids"]
        self.doc_sources = meta["doc_sources"]

        if self.use_hybrid:
            tokenized = [doc.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized)

        logger.info(f"Index loaded: {self.faiss_index.ntotal} vectors")

    # ── Retrieval ─────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int,
        similarity_threshold: float = 0.0,
    ) -> List[RetrievedDocument]:
        """
        Retrieve top-k documents for a query.
        Uses hybrid scoring if BM25 is available.

        Args:
            query:                input query string
            k:                    number of documents to retrieve (from complexity estimator)
            similarity_threshold: minimum score to include a document

        Returns:
            List of RetrievedDocument sorted by hybrid_score desc
        """
        if self.faiss_index is None:
            raise RuntimeError("Index not built. Call index_documents() first.")

        # Encode query
        query_emb = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        # ── Dense Retrieval ───────────────────────────────────
        # Retrieve more candidates than k, filter after scoring
        candidate_k = min(k * 3, len(self.documents))
        dense_scores, dense_indices = self.faiss_index.search(query_emb, candidate_k)
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]

        # Normalize dense scores to [0,1]
        d_min, d_max = dense_scores.min(), dense_scores.max()
        dense_norm = (dense_scores - d_min) / (d_max - d_min + 1e-9)

        # ── Sparse Retrieval (BM25) ───────────────────────────
        sparse_map = {}
        if self.use_hybrid and self.bm25 is not None:
            bm25_scores = np.array(
                self.bm25.get_scores(query.lower().split())
            )
            # Only for the candidate indices
            cand_bm25 = bm25_scores[dense_indices]
            b_min, b_max = cand_bm25.min(), cand_bm25.max()
            bm25_norm = (cand_bm25 - b_min) / (b_max - b_min + 1e-9)
            sparse_map = dict(zip(dense_indices.tolist(), bm25_norm.tolist()))

        # ── Hybrid Scoring ────────────────────────────────────
        results = []
        for rank, idx in enumerate(dense_indices):
            if idx < 0 or idx >= len(self.documents):
                continue

            d_score = float(dense_norm[rank])
            s_score = sparse_map.get(int(idx), 0.0)
            h_score = self.alpha * d_score + (1 - self.alpha) * s_score

            if h_score < similarity_threshold:
                continue

            text = self.documents[idx]
            results.append(RetrievedDocument(
                doc_id=self.doc_ids[idx],
                text=text,
                score=d_score,
                sparse_score=s_score,
                hybrid_score=h_score,
                token_count=len(text.split()),
                source=self.doc_sources[idx],
            ))

        # Sort by hybrid score, return top-k
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return results[:k]
