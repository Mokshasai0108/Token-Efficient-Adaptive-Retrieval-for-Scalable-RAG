"""
TEAR — Index Builder Script
Run this ONCE before starting the API to download datasets and build the FAISS index.

Usage:
    python build_index.py
    python build_index.py --max-docs 10000   # smaller for quick testing
    python build_index.py --datasets squad_v2 trivia_qa
"""

import argparse
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from config import TEARConfig
from modules.dataset_loader import DatasetLoader
from modules.retrieval_engine import AdaptiveRetrievalEngine


def build_index(max_docs: int, datasets: list):
    logger.info("=" * 60)
    logger.info("  TEAR — Index Builder")
    logger.info("=" * 60)

    # Config
    cfg = TEARConfig()
    cfg.max_docs_per_dataset = max_docs
    if datasets:
        cfg.datasets_to_load = datasets

    os.makedirs("data", exist_ok=True)

    # Step 1: Load datasets
    logger.info(f"Datasets: {cfg.datasets_to_load}")
    logger.info(f"Max docs per dataset: {cfg.max_docs_per_dataset}")

    loader = DatasetLoader(cfg)
    documents, qa_pairs = loader.load_all()

    logger.info(f"Total documents to index: {len(documents)}")
    logger.info(f"Total QA pairs for evaluation: {len(qa_pairs)}")

    # Save QA pairs for later evaluation
    import json
    with open("data/qa_pairs.json", "w") as f:
        json.dump(qa_pairs, f)
    logger.info("QA pairs saved to data/qa_pairs.json")

    # Step 2: Build retrieval index
    engine = AdaptiveRetrievalEngine(cfg)
    engine.index_documents(
        texts=[d.text for d in documents],
        doc_ids=[d.doc_id for d in documents],
        sources=[d.source for d in documents],
    )

    logger.info("=" * 60)
    logger.info("Index build complete!")
    logger.info(f"  FAISS index → {cfg.faiss_index_path}.bin")
    logger.info(f"  Metadata    → {cfg.faiss_index_path}_meta.pkl")
    logger.info(f"  QA pairs    → data/qa_pairs.json")
    logger.info("You can now run: python api.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TEAR search index")
    parser.add_argument(
        "--max-docs", type=int, default=50000,
        help="Max documents per dataset (default: 50000)"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["natural_questions", "trivia_qa", "squad_v2"],
        default=None,
        help="Datasets to load (default: all three)"
    )
    args = parser.parse_args()
    build_index(args.max_docs, args.datasets)
