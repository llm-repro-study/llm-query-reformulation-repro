"""Retrieval module — BM25, SPLADE, and BGE via Pyserini CLI.

This module handles two distinct retrieval needs:

1. **Full retrieval** (``run_retrieval``): Runs Pyserini CLI to produce
   TREC-format run files for evaluation (top-1000).
2. **Context retrieval** (``retrieve_contexts_for_queries``): Uses Pyserini's
   ``LuceneSearcher`` Python API to fetch top-k passage texts at reformulation
   time.  Corpus-grounded methods (CSQE, LameR) call this to condition
   their prompts on initially-retrieved evidence.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .data import DATASETS

logger = logging.getLogger(__name__)

# ── Retriever configuration ──────────────────────────────────────────────────

RETRIEVER_CONFIGS = {
    "bm25": {
        "searcher": "lucene",
        "index_key": "index_bm25",
        "extra_args": [],
    },
    "splade": {
        "searcher": "lucene",
        "index_key": "index_splade",
        "extra_args": ["--impact", "--pretokenized"],
    },
    "bge": {
        "searcher": "faiss",
        "index_key": "index_bge",
        "extra_args": [
            "--encoder-class", "auto",
            "--encoder", "BAAI/bge-base-en-v1.5",
            "--l2-norm",
        ],
    },
}


def run_retrieval(
    queries_tsv: str | Path,
    dataset: str,
    retriever: str,
    output_run: str | Path,
    hits: int = 1000,
    threads: int = 16,
    batch_size: int = 512,
    remove_query: bool = False,
    query_prefix: str = "",
) -> Path:
    """Run a retrieval experiment using Pyserini's CLI.

    Parameters
    ----------
    queries_tsv : path
        TSV file with ``qid \\t query`` rows.
    dataset : str
        Dataset name (key in :data:`DATASETS`).
    retriever : str
        One of ``"bm25"``, ``"splade"``, ``"bge"``.
    output_run : path
        Where to write the TREC-format run file.
    hits : int
        Number of documents to retrieve per query.
    threads / batch_size : int
        Parallelism settings.
    remove_query : bool
        If True, pass ``--remove-query`` (used for BEIR + dense).
    query_prefix : str
        Prefix appended before each query (used for BGE).

    Returns
    -------
    Path
        The ``output_run`` path.
    """
    ds_cfg = DATASETS[dataset]
    ret_cfg = RETRIEVER_CONFIGS[retriever]
    index_name = ds_cfg[ret_cfg["index_key"]]
    output_run = Path(output_run)
    output_run.parent.mkdir(parents=True, exist_ok=True)

    if ret_cfg["searcher"] == "faiss":
        module = "pyserini.search.faiss"
    else:
        module = "pyserini.search.lucene"

    cmd = [
        "python", "-m", module,
        "--threads", str(threads),
        "--batch-size", str(batch_size),
        "--index", index_name,
        "--topics", str(queries_tsv),
        "--output", str(output_run),
        "--hits", str(hits),
    ] + ret_cfg["extra_args"]

    if query_prefix:
        cmd.extend(["--query-prefix", query_prefix])
    if remove_query:
        cmd.append("--remove-query")

    print(f"  [{retriever.upper()}] Running retrieval on {dataset} …")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        raise RuntimeError(
            f"Retrieval failed (exit {result.returncode}):\n{result.stderr[:1000]}"
        )
    print(f"  [{retriever.upper()}] Done → {output_run}")
    return output_run


# ── Context retrieval for corpus-grounded methods ─────────────────────────────


def _extract_passage_text(hit) -> str:
    """Extract readable passage text from a Pyserini Lucene hit.

    Pyserini indexes store documents in different formats:
    - MS MARCO passages have a ``raw`` JSON field with ``"contents"`` key.
    - BEIR indexes typically store a ``contents`` field directly.
    This helper handles both transparently.
    """
    doc = hit.lucene_document

    # Try the "raw" field first (MS MARCO-style indexes)
    raw = doc.get("raw")
    if raw:
        try:
            parsed = json.loads(raw)
            # MS MARCO: {"id": "...", "contents": "passage text"}
            if "contents" in parsed:
                return parsed["contents"].strip()
            # Some indexes use "body" or "text"
            for key in ("body", "text", "passage"):
                if key in parsed:
                    return parsed[key].strip()
            # Fall back to the full raw string if no known key
            return raw.strip()
        except (json.JSONDecodeError, TypeError):
            return raw.strip()

    # Try "contents" field directly (BEIR-style flat indexes)
    contents = doc.get("contents")
    if contents:
        return contents.strip()

    return ""


def retrieve_contexts_for_queries(
    queries_tsv: str | Path,
    dataset: str,
    retriever: str = "bm25",
    k: int = 10,
    threads: int = 16,
) -> Dict[str, List[str]]:
    """Retrieve top-*k* passage texts for each query via BM25.

    Used by corpus-grounded methods (CSQE, LameR) that condition their
    prompts on initially-retrieved documents.

    The searcher is configured with the dataset-specific BM25 weights
    (k1, b) from the dataset registry so context retrieval is consistent
    with the paper's experimental setup.

    Parameters
    ----------
    queries_tsv : path
        TSV file with ``qid \\t query`` rows.
    dataset : str
        Dataset name (key in :data:`DATASETS`).
    retriever : str
        Currently only ``"bm25"`` is supported for context retrieval.
    k : int
        Number of top passages to retrieve per query (default 10).
    threads : int
        Thread count for batch search.

    Returns
    -------
    dict[str, list[str]]
        Mapping ``qid → [passage_text_1, …, passage_text_k]``.
    """
    from pyserini.search.lucene import LuceneSearcher

    ds_cfg = DATASETS[dataset]
    index_name = ds_cfg["index_bm25"]
    bm25_k1 = ds_cfg.get("bm25_k1", 0.9)
    bm25_b = ds_cfg.get("bm25_b", 0.4)

    logger.info(
        "Context retrieval: index=%s  k1=%.2f  b=%.2f  top_k=%d",
        index_name, bm25_k1, bm25_b, k,
    )
    print(
        f"  [CTX] Fetching top-{k} BM25 passages "
        f"(index={index_name}, k1={bm25_k1}, b={bm25_b}) …"
    )

    # Initialise searcher with dataset-specific BM25 weights
    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    searcher.set_bm25(k1=bm25_k1, b=bm25_b)

    queries = _read_tsv(queries_tsv)

    # Batch search for efficiency (Pyserini's Java backend parallelises this)
    query_texts = [text for _, text in queries]
    query_ids = [qid for qid, _ in queries]

    results = searcher.batch_search(
        queries=query_texts,
        qids=query_ids,
        k=k,
        threads=threads,
    )

    ctx_map: Dict[str, List[str]] = {}
    for qid in query_ids:
        hits = results.get(qid, [])
        ctx_map[qid] = [_extract_passage_text(h) for h in hits]

    print(f"  [CTX] Retrieved contexts for {len(ctx_map)} queries")
    return ctx_map


def _read_tsv(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs

