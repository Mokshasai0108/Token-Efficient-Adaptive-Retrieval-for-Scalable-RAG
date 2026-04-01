"""
TEAR — Modules 4–8:
  4. Re-ranking Module
  5. Token Utility Scoring
  6. Budget-Constrained Selection (greedy knapsack)
  7. Redundancy Filtering
  8. Context Compression
"""

import re
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from loguru import logger

from modules.retrieval_engine import RetrievedDocument


# ═══════════════════════════════════════════════════════════════
# MODULE 4 — Re-ranking
# ═══════════════════════════════════════════════════════════════

class Reranker:
    """
    Re-ranks retrieved documents using:
    - Bi-encoder (fast, cosine similarity, default)
    - Cross-encoder (slow but accurate, optional)

    Cross-encoder models the query-document pair jointly,
    giving much better relevance scores at higher latency.
    """

    def __init__(self, config, use_cross_encoder: bool = False):
        self.use_cross_encoder = use_cross_encoder
        self.config = config

        if use_cross_encoder:
            logger.info("Loading cross-encoder for reranking...")
            self.cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512
            )
        else:
            logger.info("Using bi-encoder for reranking (fast mode)")
            self.encoder = SentenceTransformer(config.embed_model_id)

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_n: int = None,
    ) -> List[RetrievedDocument]:
        """
        Re-rank documents by relevance to query.
        Returns documents sorted by rerank score descending.
        """
        if not documents:
            return documents

        top_n = top_n or len(documents)

        if self.use_cross_encoder:
            return self._cross_encoder_rerank(query, documents, top_n)
        else:
            return self._bi_encoder_rerank(query, documents, top_n)

    def _cross_encoder_rerank(self, query, docs, top_n):
        pairs = [(query, doc.text) for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        for doc, score in zip(docs, scores):
            doc.score = float(score)
        docs.sort(key=lambda x: x.score, reverse=True)
        return docs[:top_n]

    def _bi_encoder_rerank(self, query, docs, top_n):
        query_emb = self.encoder.encode([query], normalize_embeddings=True)
        doc_texts = [d.text for d in docs]
        doc_embs = self.encoder.encode(doc_texts, normalize_embeddings=True)
        scores = (query_emb @ doc_embs.T)[0]
        for doc, score in zip(docs, scores):
            doc.score = float(score)
        docs.sort(key=lambda x: x.score, reverse=True)
        return docs[:top_n]


# ═══════════════════════════════════════════════════════════════
# MODULE 5 — Token Utility Scoring
# ═══════════════════════════════════════════════════════════════

class TokenUtilityScorer:
    """
    Computes utility score for each document:
        U(d) = Rel(d, q) / Tokens(d)

    This implements the core TEAR objective:
    maximize relevance per token consumed.
    """

    def score(
        self,
        documents: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """
        Attaches utility score to each document in-place.
        Documents with zero tokens get utility = 0.
        """
        for doc in documents:
            tokens = max(doc.token_count, 1)
            doc.hybrid_score = doc.score / tokens * 100  # scale for readability
        return documents

    def get_utility_ranking(
        self,
        documents: List[RetrievedDocument],
    ) -> List[Tuple[RetrievedDocument, float]]:
        """Returns (doc, utility) pairs sorted by utility descending."""
        scored = [(doc, doc.score / max(doc.token_count, 1)) for doc in documents]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ═══════════════════════════════════════════════════════════════
# MODULE 6 — Budget-Constrained Selection (Greedy Knapsack)
# ═══════════════════════════════════════════════════════════════

class BudgetConstrainedSelector:
    """
    Selects documents to maximize total relevance under a token budget.

    Algorithm:
        Greedy 0/1 Knapsack approximation:
        1. Sort documents by U(d) = relevance / tokens (descending)
        2. Greedily add documents until budget B is exhausted

    This approximates the NP-hard 0/1 knapsack problem in O(n log n).
    """

    def __init__(self, token_budget: int):
        self.token_budget = token_budget

    def select(
        self,
        documents: List[RetrievedDocument],
    ) -> Tuple[List[RetrievedDocument], int]:
        """
        Select documents under token budget.

        Returns:
            selected_docs: list of chosen documents
            tokens_used:   total tokens consumed
        """
        if not documents:
            return [], 0

        # Sort by utility: relevance per token
        ranked = sorted(
            documents,
            key=lambda d: d.score / max(d.token_count, 1),
            reverse=True
        )

        selected = []
        tokens_used = 0

        for doc in ranked:
            if tokens_used + doc.token_count <= self.token_budget:
                selected.append(doc)
                tokens_used += doc.token_count
            else:
                # Try to fit a trimmed version
                remaining = self.token_budget - tokens_used
                if remaining > 50:  # only if meaningful space left
                    words = doc.text.split()[:remaining]
                    trimmed_text = " ".join(words)
                    trimmed_doc = RetrievedDocument(
                        doc_id=doc.doc_id + "_trimmed",
                        text=trimmed_text,
                        score=doc.score,
                        token_count=len(words),
                        source=doc.source,
                    )
                    selected.append(trimmed_doc)
                    tokens_used += len(words)
                break  # budget exhausted

        return selected, tokens_used


# ═══════════════════════════════════════════════════════════════
# MODULE 7 — Redundancy Filtering
# ═══════════════════════════════════════════════════════════════

class RedundancyFilter:
    """
    Removes near-duplicate documents using cosine similarity on embeddings.
    Two documents with sim > threshold are considered redundant;
    only the higher-scored one is kept.

    Implements greedy MMR-style (Maximal Marginal Relevance) filtering.
    """

    def __init__(self, config):
        self.threshold = config.redundancy_threshold
        self.encoder = SentenceTransformer(config.embed_model_id)

    def filter(
        self,
        documents: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """
        Remove redundant documents.
        Documents are already sorted by score (highest first).
        """
        if len(documents) <= 1:
            return documents

        texts = [doc.text for doc in documents]
        embeddings = self.encoder.encode(texts, normalize_embeddings=True)

        selected_indices = []
        selected_embeddings = []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            if not selected_indices:
                selected_indices.append(i)
                selected_embeddings.append(emb)
                continue

            # Check similarity against all selected docs
            sims = np.dot(selected_embeddings, emb)
            max_sim = float(np.max(sims))

            if max_sim < self.threshold:
                selected_indices.append(i)
                selected_embeddings.append(emb)

        filtered = [documents[i] for i in selected_indices]
        removed = len(documents) - len(filtered)
        if removed > 0:
            logger.debug(f"Redundancy filter removed {removed} duplicate docs")
        return filtered


# ═══════════════════════════════════════════════════════════════
# MODULE 8 — Context Compression
# ═══════════════════════════════════════════════════════════════

class ContextCompressor:
    """
    Compresses each document to retain only the most relevant sentences.

    Strategy: Extractive sentence scoring
    1. Split document into sentences
    2. Score each sentence by similarity to query embedding
    3. Keep top-p% sentences by score
    4. Reconstruct compressed document preserving order

    This reduces token count while preserving semantic density.
    """

    def __init__(self, config):
        self.compression_ratio = config.compression_ratio
        self.min_sentence_tokens = config.min_sentence_tokens
        self.encoder = SentenceTransformer(config.embed_model_id)

    def compress(
        self,
        query: str,
        documents: List[RetrievedDocument],
    ) -> Tuple[List[RetrievedDocument], int]:
        """
        Compress each document's text by extracting key sentences.

        Returns:
            compressed_docs: documents with shortened text
            tokens_saved:    total tokens removed
        """
        query_emb = self.encoder.encode([query], normalize_embeddings=True)[0]
        compressed = []
        tokens_before = 0
        tokens_after = 0

        for doc in documents:
            tokens_before += doc.token_count
            compressed_text, new_token_count = self._compress_doc(
                doc.text, query_emb
            )
            tokens_after += new_token_count

            new_doc = RetrievedDocument(
                doc_id=doc.doc_id,
                text=compressed_text,
                score=doc.score,
                sparse_score=doc.sparse_score,
                hybrid_score=doc.hybrid_score,
                token_count=new_token_count,
                source=doc.source,
            )
            compressed.append(new_doc)

        tokens_saved = tokens_before - tokens_after
        logger.debug(f"Compression: {tokens_before} → {tokens_after} tokens "
                     f"(saved {tokens_saved})")
        return compressed, tokens_saved

    def _compress_doc(
        self, text: str, query_emb: np.ndarray
    ) -> Tuple[str, int]:
        """Compress a single document's text."""
        sentences = self._split_sentences(text)

        if len(sentences) <= 2:
            return text, len(text.split())

        # Filter too-short sentences
        sentences = [s for s in sentences
                     if len(s.split()) >= self.min_sentence_tokens]
        if not sentences:
            return text, len(text.split())

        # Score sentences by similarity to query
        sent_embs = self.encoder.encode(sentences, normalize_embeddings=True)
        scores = np.dot(sent_embs, query_emb)

        # Keep top compression_ratio sentences
        n_keep = max(1, int(len(sentences) * self.compression_ratio))
        top_indices = sorted(
            np.argsort(scores)[-n_keep:].tolist()
        )  # preserve original order

        compressed_sentences = [sentences[i] for i in top_indices]
        compressed_text = " ".join(compressed_sentences)
        return compressed_text, len(compressed_text.split())

    def _split_sentences(self, text: str) -> List[str]:
        """Simple rule-based sentence splitter."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def build_context(self, documents: List[RetrievedDocument]) -> str:
        """
        Assemble final context string from compressed documents.
        Format: numbered passages with source labels.
        """
        parts = []
        for i, doc in enumerate(documents, 1):
            parts.append(
                f"[Passage {i} | Source: {doc.source}]\n{doc.text}"
            )
        return "\n\n".join(parts)
