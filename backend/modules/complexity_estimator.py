"""
TEAR — Module 1: Query Complexity Estimator
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

import spacy
import torch
from transformers import AutoTokenizer, AutoModel
from loguru import logger


@dataclass
class ComplexityResult:
    query: str
    complexity_score: float
    complexity_label: str       # simple|moderate|complex
    suggested_k: int
    features: Dict[str, float] = field(default_factory=dict)


class QueryComplexityEstimator:
    """
    Estimates query complexity using 5 linguistic + semantic features.
    C(q) = w1*f1 + w2*f2 + w3*f3 + w4*f4 + w5*f5
    """

    WEIGHTS = {
        "token_length":     0.20,
        "entity_density":   0.25,
        "pos_diversity":    0.15,
        "question_type":    0.25,
        "semantic_entropy": 0.15,
    }

    QUESTION_WEIGHTS = {
        "why": 1.0, "how": 0.9, "explain": 0.9,
        "compare": 1.0, "analyze": 1.0, "describe": 0.8,
        "what": 0.5, "which": 0.5,
        "where": 0.3, "when": 0.3, "who": 0.3,
        "is": 0.1, "are": 0.1, "does": 0.1,
    }

    def __init__(self, config):
        self.config = config
        self.k_map = {
            "simple":   config.k_simple,
            "moderate": config.k_moderate,
            "complex":  config.k_complex,
        }
        self.thresholds = config.complexity_thresholds

        logger.info("Loading spaCy model...")
        self.nlp = spacy.load(config.spacy_model)

        logger.info(f"Loading embedding model: {config.embed_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.embed_model_id)
        self.embed_model = AutoModel.from_pretrained(config.embed_model_id)
        self.embed_model.eval()

    def estimate(self, query: str) -> ComplexityResult:
        query = query.strip()
        doc = self.nlp(query)

        f1 = self._token_length_score(doc)
        f2 = self._entity_density_score(doc)
        f3 = self._pos_diversity_score(doc)
        f4 = self._question_type_score(query)
        f5 = self._semantic_entropy_score(query)

        features = {
            "token_length":     round(f1, 4),
            "entity_density":   round(f2, 4),
            "pos_diversity":    round(f3, 4),
            "question_type":    round(f4, 4),
            "semantic_entropy": round(f5, 4),
        }

        score = sum(self.WEIGHTS[k] * v for k, v in features.items())
        score = float(np.clip(score, 0.0, 1.0))
        label = self._to_label(score)

        return ComplexityResult(
            query=query,
            complexity_score=round(score, 4),
            complexity_label=label,
            suggested_k=self.k_map[label],
            features=features,
        )

    def estimate_batch(self, queries: List[str]) -> List[ComplexityResult]:
        return [self.estimate(q) for q in queries]

    # ── Feature Extractors ────────────────────────────────────

    def _token_length_score(self, doc) -> float:
        length = len([t for t in doc if not t.is_space])
        return min(length, 50) / 50

    def _entity_density_score(self, doc) -> float:
        tokens = [t for t in doc if not t.is_space and not t.is_punct]
        if not tokens:
            return 0.0
        return min(len(doc.ents) / len(tokens), 1.0)

    def _pos_diversity_score(self, doc) -> float:
        tags = [t.pos_ for t in doc if not t.is_space]
        if not tags:
            return 0.0
        return float(np.clip(len(set(tags)) / len(tags), 0.0, 1.0))

    def _question_type_score(self, query: str) -> float:
        words = query.lower().split()
        for word in words[:4]:
            clean = re.sub(r'[^a-z]', '', word)
            if clean in self.QUESTION_WEIGHTS:
                return self.QUESTION_WEIGHTS[clean]
        return 0.5

    def _semantic_entropy_score(self, query: str) -> float:
        inputs = self.tokenizer(
            query, return_tensors="pt",
            truncation=True, max_length=64, padding=True
        )
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
        variance = outputs.last_hidden_state.squeeze(0).var(dim=0).mean().item()
        return float(np.clip(variance * 20, 0.0, 1.0))

    def _to_label(self, score: float) -> str:
        low, high = self.thresholds
        if score < low:
            return "simple"
        elif score < high:
            return "moderate"
        return "complex"
