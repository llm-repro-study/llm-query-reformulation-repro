"""Dataset loading utilities for TREC DL and BEIR benchmarks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from .methods.base import Query

# ── Dataset metadata ────────────────────────────────────────────────────────

DATASETS: Dict[str, Dict] = {
    # TREC Deep Learning (MS MARCO V1 passage)
    "dl19": {
        "topics": "dl19-passage",
        "qrels":  "dl19-passage",
        "index_bm25":   "msmarco-v1-passage",
        "index_splade": "msmarco-v1-passage-splade-pp-ed",
        "index_bge":    "msmarco-v1-passage.bge-base-en-v1.5",
        "bm25_k1": 0.9,
        "bm25_b":  0.4,
        "eval_metrics": ["ndcg_cut_10", "recall_1000"],
        "eval_depth":   1000,
        "trec_args":    ["-c", "-l", "2"],
        "group": "trec",
    },
    "dl20": {
        "topics": "dl20-passage",
        "qrels":  "dl20-passage",
        "index_bm25":   "msmarco-v1-passage",
        "index_splade": "msmarco-v1-passage-splade-pp-ed",
        "index_bge":    "msmarco-v1-passage.bge-base-en-v1.5",
        "bm25_k1": 0.9,
        "bm25_b":  0.4,
        "eval_metrics": ["ndcg_cut_10", "recall_1000"],
        "eval_depth":   1000,
        "trec_args":    ["-c", "-l", "2"],
        "group": "trec",
    },
    "dlhard": {
        "topics": "dl19-passage",   # queries loaded from file
        "qrels":  None,              # user supplies path
        "index_bm25":   "msmarco-v1-passage",
        "index_splade": "msmarco-v1-passage-splade-pp-ed",
        "index_bge":    "msmarco-v1-passage.bge-base-en-v1.5",
        "bm25_k1": 0.9,
        "bm25_b":  0.4,
        "eval_metrics": ["ndcg_cut_10", "recall_1000"],
        "eval_depth":   1000,
        "trec_args":    ["-c", "-l", "2"],
        "group": "trec",
    },
    # BEIR
    "scifact": {
        "topics": "beir-v1.0.0-scifact-test",
        "qrels":  "beir-v1.0.0-scifact-test",
        "index_bm25":   "beir-v1.0.0-scifact.flat",
        "index_splade": "beir-v1.0.0-scifact-splade-pp-ed",
        "index_bge":    "beir-v1.0.0-scifact.bge-base-en-v1.5",
        "bm25_k1": 0.9,
        "bm25_b":  0.4,
        "eval_metrics": ["ndcg_cut_10", "recall_100"],
        "eval_depth":   1000,
        "trec_args":    ["-c"],
        "group": "beir",
    },
    "arguana": {
        "topics": "beir-v1.0.0-arguana-test",
        "qrels":  "beir-v1.0.0-arguana-test",
        "index_bm25":   "beir-v1.0.0-arguana.flat",
        "index_splade": "beir-v1.0.0-arguana-splade-pp-ed",
        "index_bge":    "beir-v1.0.0-arguana.bge-base-en-v1.5",
        "bm25_k1": 0.9,
        "bm25_b":  0.4,
        "eval_metrics": ["ndcg_cut_10", "recall_100"],
        "eval_depth":   1000,
        "trec_args":    ["-c"],
        "group": "beir",
    },
    "covid": {
        "topics": "beir-v1.0.0-trec-covid-test",
        "qrels":  "beir-v1.0.0-trec-covid-test",
        "index_bm25":   "beir-v1.0.0-trec-covid.flat",
        "index_splade": "beir-v1.0.0-trec-covid-splade-pp-ed",
        "index_bge":    "beir-v1.0.0-trec-covid.bge-base-en-v1.5",
        "bm25_k1": 0.9,
        "bm25_b":  0.4,
        "eval_metrics": ["ndcg_cut_10", "recall_100"],
        "eval_depth":   1000,
        "trec_args":    ["-c"],
        "group": "beir",
    },
    "fiqa": {
        "topics": "beir-v1.0.0-fiqa-test",
        "qrels":  "beir-v1.0.0-fiqa-test",
        "index_bm25":   "beir-v1.0.0-fiqa.flat",
        "index_splade": "beir-v1.0.0-fiqa-splade-pp-ed",
        "index_bge":    "beir-v1.0.0-fiqa.bge-base-en-v1.5",
        "bm25_k1": 0.9,
        "bm25_b":  0.4,
        "eval_metrics": ["ndcg_cut_10", "recall_100"],
        "eval_depth":   1000,
        "trec_args":    ["-c"],
        "group": "beir",
    },
    "dbpedia": {
        "topics": "beir-v1.0.0-dbpedia-entity-test",
        "qrels":  "beir-v1.0.0-dbpedia-entity-test",
        "index_bm25":   "beir-v1.0.0-dbpedia-entity.flat",
        "index_splade": "beir-v1.0.0-dbpedia-entity-splade-pp-ed",
        "index_bge":    "beir-v1.0.0-dbpedia-entity.bge-base-en-v1.5",
        "bm25_k1": 0.9,
        "bm25_b":  0.4,
        "eval_metrics": ["ndcg_cut_10", "recall_100"],
        "eval_depth":   1000,
        "trec_args":    ["-c"],
        "group": "beir",
    },
    "news": {
        "topics": "beir-v1.0.0-trec-news-test",
        "qrels":  "beir-v1.0.0-trec-news-test",
        "index_bm25":   "beir-v1.0.0-trec-news.flat",
        "index_splade": "beir-v1.0.0-trec-news-splade-pp-ed",
        "index_bge":    "beir-v1.0.0-trec-news.bge-base-en-v1.5",
        "bm25_k1": 0.9,
        "bm25_b":  0.4,
        "eval_metrics": ["ndcg_cut_10", "recall_100"],
        "eval_depth":   1000,
        "trec_args":    ["-c"],
        "group": "beir",
    },
}


def load_queries_tsv(path: str | Path) -> List[Query]:
    """Load queries from a two-column TSV file (qid \\t text)."""
    queries: List[Query] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                queries.append(Query(qid=parts[0].strip(), text=parts[1].strip()))
    return queries


def save_queries_tsv(queries: list, path: str | Path) -> None:
    """Save reformulated queries to a TSV file.

    ``queries`` can be a list of :class:`ReformulatedQuery` or plain
    ``(qid, text)`` tuples.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for q in queries:
            if hasattr(q, "reformulated"):
                writer.writerow([q.qid, q.reformulated])
            else:
                writer.writerow(q)


def get_dataset_config(name: str) -> Dict:
    """Return dataset metadata by name, raising on unknown datasets."""
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}"
        )
    return DATASETS[name]

