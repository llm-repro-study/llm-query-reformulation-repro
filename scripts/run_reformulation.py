#!/usr/bin/env python3
"""Step 1 — Generate reformulated queries for a given method + LLM + dataset.

Usage
-----
python scripts/run_reformulation.py \
    --method genqr \
    --llm gpt-4.1 \
    --dataset dl19 \
    --queries data/dl19.tsv \
    --output outputs/gpt-4.1/genqr/dl19.tsv

For corpus-grounded methods (csqe, lamer) that need retrieved contexts,
pass ``--contexts-from bm25`` to first retrieve top-k passages via BM25.
"""

import argparse
import sys
from pathlib import Path

import yaml

# allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import LLMClient
from src.prompts import PromptBank
from src.methods import get_method
from src.data import load_queries_tsv, save_queries_tsv
from src.retrieval import retrieve_contexts_for_queries


def main():
    parser = argparse.ArgumentParser(description="Run LLM-based query reformulation.")
    parser.add_argument("--method", required=True, help="Reformulation method name.")
    parser.add_argument("--llm", default="gpt-4.1", help="LLM identifier.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--queries", required=True, help="Input queries TSV (qid\\ttext).")
    parser.add_argument("--output", required=True, help="Output TSV for reformulated queries.")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML file.")
    parser.add_argument("--prompts", default=None, help="Override prompt bank path.")
    parser.add_argument("--contexts-from", default=None,
                        help="Retriever to use for fetching contexts (bm25). "
                             "Required for csqe and lamer.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max output tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature.")
    args = parser.parse_args()

    # ── Load config ─────────────────────────────────────────────────────
    cfg = yaml.safe_load(open(args.config))
    method_params = cfg.get("methods", {}).get(args.method, {})

    # ── Initialise LLM client ───────────────────────────────────────────
    llm_cfg = cfg.get("llm", {})
    llm = LLMClient(
        model_name=args.llm,
        max_tokens=args.max_tokens or llm_cfg.get("max_tokens", 256),
        temperature=args.temperature if args.temperature is not None else llm_cfg.get("temperature", 0.0),
    )

    # ── Load prompts ────────────────────────────────────────────────────
    prompts_path = args.prompts or cfg.get("paths", {}).get("prompts", "prompts/prompts.json")
    prompts = PromptBank(prompts_path)

    # ── Instantiate method ──────────────────────────────────────────────
    # Inject the dataset name so methods with dataset-specific prompts
    # (e.g. LameR) select the correct template automatically.
    if "dataset" in method_params:
        method_params["dataset"] = args.dataset
    method_cls = get_method(args.method)
    method = method_cls(llm=llm, prompts=prompts, params=method_params)

    # ── Load queries ────────────────────────────────────────────────────
    queries = load_queries_tsv(args.queries)
    print(f"Loaded {len(queries)} queries from {args.queries}")

    # ── Retrieve contexts if needed ─────────────────────────────────────
    ctx_map = None
    if method.REQUIRES_CONTEXTS:
        ctx_cfg = cfg.get("context_retrieval", {})
        retriever = args.contexts_from or ctx_cfg.get("retriever", "bm25")
        ctx_k = method_params.get("context_k", ctx_cfg.get("k", 10))
        ctx_threads = ctx_cfg.get("threads", 16)
        print(f"Retrieving top-{ctx_k} contexts via {retriever} for {args.dataset} …")
        ctx_map = retrieve_contexts_for_queries(
            args.queries, args.dataset, retriever=retriever,
            k=ctx_k, threads=ctx_threads,
        )

    # ── Reformulate ─────────────────────────────────────────────────────
    results = method.reformulate_batch(queries, ctx_map=ctx_map)

    # ── Save ────────────────────────────────────────────────────────────
    save_queries_tsv(results, args.output)
    print(f"Saved {len(results)} reformulated queries → {args.output}")


if __name__ == "__main__":
    main()

