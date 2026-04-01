"""
TEAR — Dataset Loader
"""

import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm


@dataclass
class TEARDocument:
    """Unified document format across all datasets."""
    doc_id: str
    text: str
    source: str             # dataset name
    question: str = ""      # associated question (for eval)
    answer: str = ""        # ground truth answer (for eval)


class DatasetLoader:
    """
    Loads NaturalQuestions, TriviaQA, SQuAD v2 and returns:
    1. A flat list of TEARDocuments for indexing
    2. A QA pair list for evaluation
    """

    DATASET_CONFIGS = {
        "natural_questions": {
            "path": "google-research-datasets/natural_questions",
            "name": "default",
            "split": "train",
        },
        "trivia_qa": {
            "path": "mandarjoshi/trivia_qa",
            "name": "rc.wikipedia",
            "split": "train",
        },
        "squad_v2": {
            "path": "rajpurkar/squad_v2",
            "name": None,
            "split": "train",
        },
    }

    def __init__(self, config):
        self.config = config
        self.max_docs = config.max_docs_per_dataset

    def load_all(self) -> Tuple[List[TEARDocument], List[Dict]]:
        """
        Load all configured datasets.

        Returns:
            documents: flat list of TEARDocument for indexing
            qa_pairs:  list of {question, answer, source} for evaluation
        """
        all_documents = []
        all_qa_pairs = []

        for dataset_name in self.config.datasets_to_load:
            logger.info(f"Loading dataset: {dataset_name}")
            docs, qa = self._load_dataset(dataset_name)
            all_documents.extend(docs)
            all_qa_pairs.extend(qa)
            logger.info(
                f"  {dataset_name}: {len(docs)} docs, {len(qa)} QA pairs"
            )

        # Deduplicate documents by text hash
        all_documents = self._deduplicate(all_documents)
        logger.info(f"Total unique documents: {len(all_documents)}")
        return all_documents, all_qa_pairs

    def _load_dataset(
        self, name: str
    ) -> Tuple[List[TEARDocument], List[Dict]]:
        cfg = self.DATASET_CONFIGS[name]

        dataset = load_dataset(
            cfg["path"],
            cfg["name"],
            split=cfg["split"],
            trust_remote_code=True,
            streaming=False,
        )

        # Limit for indexing
        if len(dataset) > self.max_docs:
            dataset = dataset.select(range(self.max_docs))

        if name == "natural_questions":
            return self._parse_nq(dataset)
        elif name == "trivia_qa":
            return self._parse_triviaqa(dataset)
        elif name == "squad_v2":
            return self._parse_squad(dataset)
        else:
            raise ValueError(f"Unknown dataset: {name}")

    # ── NaturalQuestions ──────────────────────────────────────

    def _parse_nq(
        self, dataset
    ) -> Tuple[List[TEARDocument], List[Dict]]:
        documents = []
        qa_pairs = []

        for item in tqdm(dataset, desc="NaturalQuestions"):
            question = item.get("question", {})
            if isinstance(question, dict):
                q_text = question.get("text", "")
            else:
                q_text = str(question)

            # Extract long answer context
            annotations = item.get("annotations", {})
            long_answers = annotations.get("long_answer", [{}])
            context = ""

            document_html = item.get("document", {}).get("html", "")
            if document_html and long_answers:
                la = long_answers[0] if isinstance(long_answers, list) else long_answers
                start = la.get("start_byte", 0)
                end = la.get("end_byte", 0)
                if end > start:
                    # Strip HTML tags
                    import re
                    raw = document_html[start:end]
                    context = re.sub(r'<[^>]+>', ' ', raw).strip()
                    context = re.sub(r'\s+', ' ', context)[:1000]  # cap length

            # Short answers
            short_answers = annotations.get("short_answers", [[]])
            if isinstance(short_answers, list) and short_answers:
                sa = short_answers[0]
                if isinstance(sa, dict):
                    answer = sa.get("text", [""])[0] if sa.get("text") else ""
                else:
                    answer = str(sa)
            else:
                answer = ""

            if context and q_text:
                doc_id = self._make_id("nq", q_text)
                documents.append(TEARDocument(
                    doc_id=doc_id,
                    text=context,
                    source="natural_questions",
                    question=q_text,
                    answer=answer,
                ))
                qa_pairs.append({
                    "question": q_text,
                    "answer": answer,
                    "context_doc_id": doc_id,
                    "source": "natural_questions",
                })

        return documents, qa_pairs

    # ── TriviaQA ──────────────────────────────────────────────

    def _parse_triviaqa(
        self, dataset
    ) -> Tuple[List[TEARDocument], List[Dict]]:
        documents = []
        qa_pairs = []

        for item in tqdm(dataset, desc="TriviaQA"):
            question = item.get("question", "")
            answer_data = item.get("answer", {})
            answer = answer_data.get("value", "") if isinstance(answer_data, dict) else ""

            # Extract Wikipedia context passages
            entity_pages = item.get("entity_pages", {})
            wiki_contexts = entity_pages.get("wiki_context", [])
            if isinstance(wiki_contexts, str):
                wiki_contexts = [wiki_contexts]

            for ctx in wiki_contexts[:2]:  # max 2 passages per question
                if ctx and len(ctx.split()) > 20:
                    doc_id = self._make_id("tqa", ctx[:50])
                    documents.append(TEARDocument(
                        doc_id=doc_id,
                        text=ctx[:1000],
                        source="trivia_qa",
                        question=question,
                        answer=answer,
                    ))

            if question and answer:
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "source": "trivia_qa",
                })

        return documents, qa_pairs

    # ── SQuAD v2 ─────────────────────────────────────────────

    def _parse_squad(
        self, dataset
    ) -> Tuple[List[TEARDocument], List[Dict]]:
        documents = []
        qa_pairs = []
        seen_contexts = set()

        for item in tqdm(dataset, desc="SQuAD v2"):
            context = item.get("context", "")
            question = item.get("question", "")
            answers_data = item.get("answers", {})
            answer_texts = answers_data.get("text", [])
            answer = answer_texts[0] if answer_texts else ""

            if context and context not in seen_contexts:
                seen_contexts.add(context)
                doc_id = self._make_id("squad", context[:50])
                documents.append(TEARDocument(
                    doc_id=doc_id,
                    text=context[:1000],
                    source="squad_v2",
                    question=question,
                    answer=answer,
                ))

            if question:
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "source": "squad_v2",
                    "is_impossible": not bool(answer_texts),
                })

        return documents, qa_pairs

    # ── Utilities ─────────────────────────────────────────────

    def _make_id(self, prefix: str, text: str) -> str:
        h = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{prefix}_{h}"

    def _deduplicate(self, docs: List[TEARDocument]) -> List[TEARDocument]:
        seen = set()
        unique = []
        for doc in docs:
            h = hashlib.md5(doc.text.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(doc)
        return unique
