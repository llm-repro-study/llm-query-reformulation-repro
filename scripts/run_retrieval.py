#!/usr/bin/env python3
"""Step 2 — Run retrieval with reformulated queries across retrievers.

Usage
-----
python scripts/run_retrieval.py \
    --queries outputs/gpt-4.1/genqr/dl19.tsv \
    --dataset dl19 \
    --retrievers bm25 splade bge \
    --output-dir outputs/gpt-4.1/genqr/runs/
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import run_retrieval


def main():
    parser = argparse.ArgumentParser(description="Run retrieval with Pyserini.")
    parser.add_argument("--queries", required=True, help="Reformulated queries TSV.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "splade", "bge"],
                        help="Retriever(s) to use.")
    parser.add_argument("--output-dir", required=True, help="Directory for run files.")
    parser.add_argument("--hits", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ret in args.retrievers:
        run_file = out / f"{args.dataset}.{ret}.run"

        # BGE needs a query prefix for MS MARCO-based datasets
        query_prefix = ""
        remove_query = False
        if ret == "bge":
            query_prefix = "Represent this sentence for searching relevant passages:"
            # BEIR datasets use --remove-query
            from src.data import DATASETS
            if DATASETS.get(args.dataset, {}).get("group") == "beir":
                remove_query = True

        run_retrieval(
            queries_tsv=args.queries,
            dataset=args.dataset,
            retriever=ret,
            output_run=run_file,
            hits=args.hits,
            threads=args.threads,
            batch_size=args.batch_size,
            remove_query=remove_query,
            query_prefix=query_prefix,
        )


if __name__ == "__main__":
    main()

