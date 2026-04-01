"""
TEAR — Ablation Study Runner
Systematically disables each component and measures performance drop.

Components tested:
  1. Full TEAR (baseline for comparison)
  2. No Adaptive Retrieval  → fixed k=5
  3. No Token Budget        → unlimited token selection
  4. No Redundancy Filter   → skip MMR filtering
  5. No Context Compression → raw passages used as context
  6. Standard RAG           → fixed k=5, no pipeline modules
  7. No RAG                 → LLM only

Usage:
    python run_ablation.py --n-samples 300
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from config import TEARConfig
from pipeline import TEARPipeline, TEARResult
from modules.evaluator import TEAREvaluator, EvalMetrics


# ─────────────────────────────────────────────────────────────
# Ablation Variant Configs
# Each entry disables one specific component
# ─────────────────────────────────────────────────────────────

def make_ablation_pipeline(variant: str) -> TEARPipeline:
    cfg = TEARConfig()
    pipeline = TEARPipeline(cfg, lazy_llm=False)
    pipeline.retrieval_engine.load_index()

    if variant == "no_adaptive":
        # Fix k=5 regardless of complexity
        original_estimate = pipeline.complexity_estimator.estimate
        def fixed_k_estimate(query):
            result = original_estimate(query)
            result.suggested_k = 5
            return result
        pipeline.complexity_estimator.estimate = fixed_k_estimate

    elif variant == "no_budget":
        # Remove token budget constraint (allow unlimited tokens)
        pipeline.budget_selector.token_budget = 999_999

    elif variant == "no_filter":
        # Skip redundancy filtering (identity function)
        pipeline.redundancy_filter.filter = lambda docs: docs

    elif variant == "no_compression":
        # Skip compression and return raw text
        original_compress = pipeline.compressor.compress
        def no_compress(query, docs):
            tokens_total = sum(d.token_count for d in docs)
            return docs, 0
        pipeline.compressor.compress = no_compress

    return pipeline


# ─────────────────────────────────────────────────────────────
# Run Ablation
# ─────────────────────────────────────────────────────────────

def run_ablation(n_samples: int):
    logger.info("=" * 65)
    logger.info("  TEAR — Ablation Study")
    logger.info("=" * 65)

    # Load QA pairs
    qa_path = "data/qa_pairs.json"
    if not os.path.exists(qa_path):
        raise FileNotFoundError(
            "QA pairs not found. Run build_index.py first."
        )
    with open(qa_path) as f:
        qa_pairs = json.load(f)

    # Filter to only answerable questions
    qa_pairs = [q for q in qa_pairs if q.get("answer", "").strip()]
    logger.info(f"Loaded {len(qa_pairs)} answerable QA pairs")

    variants = [
        ("full_tear",       "Full TEAR (all modules)"),
        ("no_adaptive",     "TEAR − Adaptive k"),
        ("no_budget",       "TEAR − Token Budget"),
        ("no_filter",       "TEAR − Redundancy Filter"),
        ("no_compression",  "TEAR − Compression"),
    ]

    all_metrics: list[EvalMetrics] = []

    for variant_id, variant_name in variants:
        logger.info(f"\nRunning variant: {variant_name}")
        pipeline = make_ablation_pipeline(variant_id)
        evaluator = TEAREvaluator(pipeline)
        metrics = evaluator.evaluate(
            qa_pairs=qa_pairs,
            system=variant_id,
            n_samples=n_samples,
        )
        metrics.system_name = variant_name
        all_metrics.append(metrics)
        logger.info(metrics.summary())

    # Also run standard_rag and no_rag baselines
    base_pipeline = TEARPipeline(TEARConfig(), lazy_llm=False)
    base_pipeline.retrieval_engine.load_index()
    base_evaluator = TEAREvaluator(base_pipeline)

    for system_id, label in [("standard_rag", "Standard RAG (fixed k=5)"),
                              ("no_rag", "No RAG (LLM only)")]:
        m = base_evaluator.evaluate(qa_pairs, system=system_id, n_samples=n_samples)
        m.system_name = label
        all_metrics.append(m)
        logger.info(m.summary())

    # Print comparison table
    _print_ablation_table(all_metrics)

    # Save to JSON
    results = [
        {
            "system": m.system_name,
            "exact_match": m.exact_match,
            "f1_score": m.f1_score,
            "avg_tokens": m.avg_tokens_used,
            "avg_latency": m.avg_latency,
            "token_efficiency": m.token_efficiency,
        }
        for m in all_metrics
    ]
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Ablation results saved to ablation_results.json")


def _print_ablation_table(metrics: list):
    header = (
        f"\n{'System':<35} {'EM':>7} {'F1':>7} "
        f"{'Tokens':>8} {'Eff':>10} {'Δ F1':>8}"
    )
    logger.info(header)
    logger.info("─" * 80)

    full_tear = next((m for m in metrics if "Full TEAR" in m.system_name), None)
    base_f1 = full_tear.f1_score if full_tear else 0

    for m in metrics:
        delta = m.f1_score - base_f1 if full_tear and m != full_tear else 0
        delta_str = f"{delta:+.4f}" if m != full_tear else "—"
        row = (
            f"{m.system_name:<35} "
            f"{m.exact_match:>7.4f} "
            f"{m.f1_score:>7.4f} "
            f"{m.avg_tokens_used:>8.1f} "
            f"{m.token_efficiency:>10.6f} "
            f"{delta_str:>8}"
        )
        logger.info(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TEAR Ablation Study")
    parser.add_argument(
        "--n-samples", type=int, default=300,
        help="Number of QA samples per variant (default: 300)"
    )
    args = parser.parse_args()
    run_ablation(args.n_samples)
