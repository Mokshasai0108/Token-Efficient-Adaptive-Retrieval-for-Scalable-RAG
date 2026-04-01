"""
TEAR — Module 10: Evaluation Pipeline
"""

import json
import string
import re
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from collections import Counter

import numpy as np
from loguru import logger
from tqdm import tqdm

from pipeline import TEARPipeline, TEARResult
from config import TEARConfig


@dataclass
class EvalMetrics:
    """Evaluation results for one system configuration."""
    system_name: str
    exact_match: float = 0.0
    f1_score: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    avg_tokens_used: float = 0.0
    avg_latency: float = 0.0
    token_efficiency: float = 0.0       # F1 / avg_tokens
    n_samples: int = 0
    results: List[Dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"System         : {self.system_name}\n"
            f"Samples        : {self.n_samples}\n"
            f"{'─'*55}\n"
            f"Exact Match    : {self.exact_match:.4f}\n"
            f"F1 Score       : {self.f1_score:.4f}\n"
            f"Precision@k    : {self.precision_at_k:.4f}\n"
            f"Recall@k       : {self.recall_at_k:.4f}\n"
            f"{'─'*55}\n"
            f"Avg Tokens     : {self.avg_tokens_used:.1f}\n"
            f"Avg Latency    : {self.avg_latency:.3f}s\n"
            f"Token Efficiency (F1/Tokens): {self.token_efficiency:.6f}\n"
            f"{'='*55}\n"
        )


class TEAREvaluator:
    """
    Evaluates TEAR against baselines on QA datasets.

    Baselines:
    1. No-RAG:         LLM only (no retrieval)
    2. Standard RAG:   Fixed top-k=5, no budget, no compression
    3. TEAR Full:      All modules active

    Ablations (remove one component at a time):
    4. No Adaptive:    Fixed k=5
    5. No Budget:      No token budget constraint
    6. No Filter:      No redundancy filter
    7. No Compression: No context compression
    """

    def __init__(self, pipeline: TEARPipeline):
        self.pipeline = pipeline
        self.config = pipeline.config

    def evaluate(
        self,
        qa_pairs: List[Dict],
        system: str = "tear",
        n_samples: int = 500,
        seed: int = 42,
    ) -> EvalMetrics:
        """
        Run evaluation on a QA dataset subset.

        Args:
            qa_pairs:   list of {question, answer, source}
            system:     one of: tear | no_rag | standard_rag
            n_samples:  number of samples to evaluate
            seed:       random seed for sampling

        Returns:
            EvalMetrics
        """
        np.random.seed(seed)
        sample = np.random.choice(qa_pairs, size=min(n_samples, len(qa_pairs)), replace=False)

        em_scores, f1_scores = [], []
        p_at_k_scores, r_at_k_scores = [], []
        token_counts, latencies = [], []
        results = []

        for item in tqdm(sample, desc=f"Evaluating [{system}]"):
            question = item["question"]
            ground_truth = item["answer"]

            if not question or not ground_truth:
                continue

            t0 = time.time()

            if system == "no_rag":
                result = self._no_rag(question)
            elif system == "standard_rag":
                result = self._standard_rag(question)
            else:
                result = self.pipeline.run(question)

            latency = time.time() - t0
            latencies.append(latency)
            token_counts.append(result.tokens_in_context)

            # Score answer
            em = self._exact_match(result.answer, ground_truth)
            f1 = self._f1_score(result.answer, ground_truth)
            em_scores.append(em)
            f1_scores.append(f1)

            # Retrieval quality (only for RAG systems)
            if system != "no_rag":
                p_k = self._precision_at_k(result.passages, ground_truth)
                r_k = self._recall_at_k(result.passages, ground_truth)
                p_at_k_scores.append(p_k)
                r_at_k_scores.append(r_k)

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": result.answer,
                "em": em, "f1": f1,
                "tokens": result.tokens_in_context,
                "latency": round(latency, 3),
            })

        avg_em = np.mean(em_scores) if em_scores else 0.0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_p = np.mean(p_at_k_scores) if p_at_k_scores else 0.0
        avg_r = np.mean(r_at_k_scores) if r_at_k_scores else 0.0
        avg_tok = np.mean(token_counts) if token_counts else 1.0
        avg_lat = np.mean(latencies) if latencies else 0.0

        return EvalMetrics(
            system_name=system,
            exact_match=round(float(avg_em), 4),
            f1_score=round(float(avg_f1), 4),
            precision_at_k=round(float(avg_p), 4),
            recall_at_k=round(float(avg_r), 4),
            avg_tokens_used=round(float(avg_tok), 2),
            avg_latency=round(float(avg_lat), 4),
            token_efficiency=round(float(avg_f1) / max(float(avg_tok), 1), 6),
            n_samples=len(results),
            results=results,
        )

    def run_full_comparison(self, qa_pairs: List[Dict], n_samples: int = 500):
        """
        Run all systems and print comparison table.
        Returns list of EvalMetrics.
        """
        systems = ["no_rag", "standard_rag", "tear"]
        all_metrics = []
        for sys in systems:
            m = self.evaluate(qa_pairs, system=sys, n_samples=n_samples)
            all_metrics.append(m)
            logger.info(m.summary())
            self._save_results(m)

        self._print_comparison_table(all_metrics)
        return all_metrics

    # ── Baseline Systems ──────────────────────────────────────

    def _no_rag(self, question: str) -> TEARResult:
        """LLM only — no retrieval."""
        if self.pipeline.generator:
            gen = self.pipeline.generator.generate(
                query=question,
                context="No context provided. Answer from your own knowledge."
            )
            answer = gen.answer
            tokens = gen.prompt_tokens
        else:
            answer = ""
            tokens = 0

        return TEARResult(
            query=question, answer=answer,
            complexity_score=0, complexity_label="n/a", k_used=0,
            docs_retrieved=0, docs_after_filter=0, docs_in_context=0,
            tokens_retrieved=0, tokens_in_context=tokens,
            tokens_saved=0, token_budget=0,
            prompt_tokens=tokens, completion_tokens=0, total_tokens=tokens,
            latency_seconds=0,
        )

    def _standard_rag(self, question: str) -> TEARResult:
        """Standard RAG: fixed top-k=5, no compression."""
        retrieved = self.pipeline.retrieval_engine.retrieve(question, k=5)
        context = "\n\n".join(d.text for d in retrieved)
        tokens = sum(d.token_count for d in retrieved)

        if self.pipeline.generator:
            gen = self.pipeline.generator.generate(question, context)
            answer = gen.answer
        else:
            answer = ""

        return TEARResult(
            query=question, answer=answer,
            complexity_score=0, complexity_label="standard", k_used=5,
            docs_retrieved=5, docs_after_filter=5, docs_in_context=5,
            tokens_retrieved=tokens, tokens_in_context=tokens,
            tokens_saved=0, token_budget=999999,
            prompt_tokens=tokens, completion_tokens=0, total_tokens=tokens,
            latency_seconds=0,
        )

    # ── Metrics ───────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        """Lowercase, strip punctuation and articles."""
        text = text.lower().strip()
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = ''.join(ch for ch in text if ch not in string.punctuation)
        return re.sub(r'\s+', ' ', text).strip()

    def _exact_match(self, pred: str, gold: str) -> float:
        return float(self._normalize(pred) == self._normalize(gold))

    def _f1_score(self, pred: str, gold: str) -> float:
        pred_tokens = self._normalize(pred).split()
        gold_tokens = self._normalize(gold).split()
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = Counter(pred_tokens) & Counter(gold_tokens)
        n_common = sum(common.values())
        if n_common == 0:
            return 0.0
        precision = n_common / len(pred_tokens)
        recall = n_common / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    def _precision_at_k(self, passages: list, gold: str) -> float:
        if not passages:
            return 0.0
        gold_norm = self._normalize(gold)
        hits = sum(1 for p in passages if gold_norm in self._normalize(p["text"]))
        return hits / len(passages)

    def _recall_at_k(self, passages: list, gold: str) -> float:
        gold_norm = self._normalize(gold)
        if not gold_norm:
            return 0.0
        for p in passages:
            if gold_norm in self._normalize(p["text"]):
                return 1.0
        return 0.0

    # ── Reporting ─────────────────────────────────────────────

    def _print_comparison_table(self, metrics: List[EvalMetrics]):
        header = f"\n{'System':<20} {'EM':>8} {'F1':>8} {'P@k':>8} {'R@k':>8} {'Tokens':>10} {'Eff':>12}"
        logger.info(header)
        logger.info("─" * 80)
        for m in metrics:
            row = (
                f"{m.system_name:<20} "
                f"{m.exact_match:>8.4f} "
                f"{m.f1_score:>8.4f} "
                f"{m.precision_at_k:>8.4f} "
                f"{m.recall_at_k:>8.4f} "
                f"{m.avg_tokens_used:>10.1f} "
                f"{m.token_efficiency:>12.6f}"
            )
            logger.info(row)

    def _save_results(self, metrics: EvalMetrics):
        path = f"results_{metrics.system_name}.json"
        with open(path, "w") as f:
            data = asdict(metrics)
            data.pop("results")  # too large for main file
            json.dump(data, f, indent=2)
        logger.info(f"Saved metrics to {path}")
