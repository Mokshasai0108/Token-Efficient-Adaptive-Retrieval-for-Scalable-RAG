"""
TEAR — Core Pipeline Orchestrator
Chains all modules: Complexity → Retrieval → Rerank → Utility → Budget → Filter → Compress → Generate
"""

import time
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger

from config import TEARConfig, config as default_config
from modules.complexity_estimator import QueryComplexityEstimator, ComplexityResult
from modules.retrieval_engine import AdaptiveRetrievalEngine, RetrievedDocument
from modules.pipeline_modules import (
    Reranker, TokenUtilityScorer,
    BudgetConstrainedSelector, RedundancyFilter, ContextCompressor
)
from modules.llm_generator import LLMGenerator, GenerationResult


@dataclass
class TEARResult:
    """Complete pipeline output returned to the API."""
    query: str
    answer: str

    # Pipeline stats
    complexity_score: float
    complexity_label: str
    k_used: int

    docs_retrieved: int
    docs_after_filter: int
    docs_in_context: int

    tokens_retrieved: int
    tokens_in_context: int
    tokens_saved: int
    token_budget: int

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float

    # Retrieved passages for display
    passages: list = field(default_factory=list)

    # Feature breakdown
    complexity_features: dict = field(default_factory=dict)


class TEARPipeline:
    """
    Full TEAR Pipeline:
    Query
    → Complexity Estimation        (Module 1)
    → Adaptive Retrieval           (Module 2+3)
    → Re-ranking                   (Module 4)
    → Token Utility Scoring        (Module 5)
    → Budget-Constrained Selection (Module 6)
    → Redundancy Filtering         (Module 7)
    → Context Compression          (Module 8)
    → LLM Generation               (Module 9)
    """

    def __init__(self, cfg: TEARConfig = None, lazy_llm: bool = False):
        """
        Args:
            cfg:      config object (defaults to global config)
            lazy_llm: if True, skip loading LLM (for indexing-only use)
        """
        self.config = cfg or default_config

        logger.info("Initializing TEAR Pipeline...")

        # Module 1
        self.complexity_estimator = QueryComplexityEstimator(self.config)

        # Module 2+3
        self.retrieval_engine = AdaptiveRetrievalEngine(self.config)

        # Module 4
        self.reranker = Reranker(self.config, use_cross_encoder=False)

        # Module 5
        self.utility_scorer = TokenUtilityScorer()

        # Module 6
        self.budget_selector = BudgetConstrainedSelector(self.config.token_budget)

        # Module 7
        self.redundancy_filter = RedundancyFilter(self.config)

        # Module 8
        self.compressor = ContextCompressor(self.config)

        # Module 9
        if not lazy_llm:
            self.generator = LLMGenerator(self.config)
        else:
            self.generator = None
            logger.info("LLM skipped (lazy mode)")

        logger.info("TEAR Pipeline ready.")

    def run(self, query: str) -> TEARResult:
        """
        Execute the full TEAR pipeline on a query.

        Args:
            query: user question string

        Returns:
            TEARResult with answer and all pipeline telemetry
        """
        t_start = time.time()

        # ── Step 1: Complexity Estimation ─────────────────────
        complexity: ComplexityResult = self.complexity_estimator.estimate(query)
        k = complexity.suggested_k
        logger.info(
            f"[1] Complexity={complexity.complexity_label} "
            f"(score={complexity.complexity_score:.3f}), k={k}"
        )

        # ── Step 2+3: Adaptive Retrieval ──────────────────────
        retrieved = self.retrieval_engine.retrieve(query=query, k=k)
        tokens_retrieved = sum(d.token_count for d in retrieved)
        logger.info(f"[2+3] Retrieved {len(retrieved)} docs, {tokens_retrieved} tokens")

        # ── Step 4: Re-ranking ────────────────────────────────
        reranked = self.reranker.rerank(
            query, retrieved, top_n=self.config.rerank_top_n
        )
        logger.info(f"[4] Reranked {len(reranked)} docs")

        # ── Step 5: Token Utility Scoring ─────────────────────
        scored = self.utility_scorer.score(reranked)
        logger.info("[5] Utility scores computed")

        # ── Step 6: Budget-Constrained Selection ──────────────
        selected, tokens_selected = self.budget_selector.select(scored)
        logger.info(
            f"[6] Budget selection: {len(selected)} docs, "
            f"{tokens_selected}/{self.config.token_budget} tokens"
        )

        # ── Step 7: Redundancy Filtering ──────────────────────
        filtered = self.redundancy_filter.filter(selected)
        logger.info(f"[7] After redundancy filter: {len(filtered)} docs")

        # ── Step 8: Context Compression ───────────────────────
        compressed, tokens_saved = self.compressor.compress(query, filtered)
        context = self.compressor.build_context(compressed)
        tokens_in_context = sum(d.token_count for d in compressed)
        logger.info(
            f"[8] Compressed context: {tokens_in_context} tokens "
            f"(saved {tokens_saved})"
        )

        # ── Step 9: LLM Generation ────────────────────────────
        if self.generator:
            gen_result: GenerationResult = self.generator.generate(
                query=query, context=context
            )
            answer = gen_result.answer
            prompt_tokens = gen_result.prompt_tokens
            completion_tokens = gen_result.completion_tokens
            total_tokens = gen_result.total_tokens
        else:
            answer = "[LLM not loaded]"
            prompt_tokens = completion_tokens = total_tokens = 0

        latency = round(time.time() - t_start, 3)
        logger.info(f"Pipeline complete in {latency}s")

        # ── Assemble Result ───────────────────────────────────
        passages = [
            {
                "doc_id": d.doc_id,
                "text": d.text[:300] + "..." if len(d.text) > 300 else d.text,
                "score": round(d.score, 4),
                "source": d.source,
                "token_count": d.token_count,
            }
            for d in compressed
        ]

        return TEARResult(
            query=query,
            answer=answer,
            complexity_score=complexity.complexity_score,
            complexity_label=complexity.complexity_label,
            k_used=k,
            docs_retrieved=len(retrieved),
            docs_after_filter=len(filtered),
            docs_in_context=len(compressed),
            tokens_retrieved=tokens_retrieved,
            tokens_in_context=tokens_in_context,
            tokens_saved=tokens_saved,
            token_budget=self.config.token_budget,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency,
            passages=passages,
            complexity_features=complexity.features,
        )

    def run_stream(self, query: str):
        """
        Streaming variant: yields tokens from the LLM.
        Pipeline steps 1-8 run synchronously first.
        """
        if not self.generator:
            raise RuntimeError("LLM not loaded")

        complexity = self.complexity_estimator.estimate(query)
        k = complexity.suggested_k
        retrieved = self.retrieval_engine.retrieve(query=query, k=k)
        reranked = self.reranker.rerank(query, retrieved)
        scored = self.utility_scorer.score(reranked)
        selected, _ = self.budget_selector.select(scored)
        filtered = self.redundancy_filter.filter(selected)
        compressed, _ = self.compressor.compress(query, filtered)
        context = self.compressor.build_context(compressed)

        yield from self.generator.generate_stream(query=query, context=context)
