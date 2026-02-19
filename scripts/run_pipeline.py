#!/usr/bin/env python3
"""End-to-end pipeline: reformulation → retrieval → evaluation.

Runs all configured methods × LLMs × datasets × retrievers and produces
a consolidated results table.

Usage
-----
# Full experiment with default config
python scripts/run_pipeline.py --config configs/default.yaml

# Single method / LLM / dataset
python scripts/run_pipeline.py \
    --methods genqr mugi \
    --llms gpt-4.1 \
    --datasets dl19 dl20 \
    --retrievers bm25
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import LLMClient
from src.prompts import PromptBank
from src.methods import get_method
from src.data import load_queries_tsv, save_queries_tsv, DATASETS
from src.retrieval import run_retrieval, retrieve_contexts_for_queries
from src.evaluation import evaluate, results_to_table


def main():
    parser = argparse.ArgumentParser(description="Full reproducibility pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--methods", nargs="+", default=None, help="Override methods list.")
    parser.add_argument("--llms", nargs="+", default=None, help="Override LLMs list.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Override datasets list.")
    parser.add_argument("--retrievers", nargs="+", default=None, help="Override retrievers list.")
    parser.add_argument("--queries-dir", default="data/queries",
                        help="Directory with query TSV files named {dataset}.tsv")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--dlhard-qrels", default=None, help="Path to DL-HARD qrels.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    methods = args.methods or list(cfg.get("methods", {}).keys())
    llms = args.llms or [cfg.get("llm", {}).get("model", "gpt-4.1")]
    datasets = args.datasets or cfg.get("datasets", list(DATASETS.keys()))
    retrievers = args.retrievers or cfg.get("retrievers", ["bm25", "splade", "bge"])
    out_root = Path(args.output_dir or cfg.get("paths", {}).get("output", "outputs"))
    prompts_path = cfg.get("paths", {}).get("prompts", "prompts/prompts.json")
    ret_cfg = cfg.get("retrieval", {})
    ctx_cfg = cfg.get("context_retrieval", {})

    all_results = {}

    for llm_name in llms:
        print(f"\n{'='*70}")
        print(f"  LLM: {llm_name}")
        print(f"{'='*70}")

        llm = LLMClient(
            model_name=llm_name,
            max_tokens=cfg.get("llm", {}).get("max_tokens", 256),
            temperature=cfg.get("llm", {}).get("temperature", 0.0),
        )
        prompts = PromptBank(prompts_path)

        for method_name in methods:
            print(f"\n  Method: {method_name}")
            method_params = cfg.get("methods", {}).get(method_name, {})
            method_cls = get_method(method_name)
            method = method_cls(llm=llm, prompts=prompts, params=method_params)

            for ds in datasets:
                print(f"    Dataset: {ds}")

                queries_file = Path(args.queries_dir) / f"{ds}.tsv"
                if not queries_file.exists():
                    print(f"      [SKIP] {queries_file} not found")
                    continue

                queries = load_queries_tsv(queries_file)

                # Update dataset tag for methods that use dataset-specific
                # prompts (e.g. LameR selects lamer_{dataset} templates).
                if "dataset" in method_params:
                    method_params["dataset"] = ds
                    method = method_cls(llm=llm, prompts=prompts, params=method_params)

                # ── Step 1: Reformulate ─────────────────────────────────
                reform_dir = out_root / llm_name / method_name
                reform_file = reform_dir / f"{ds}.tsv"

                if not reform_file.exists():
                    ctx_map = None
                    if method.REQUIRES_CONTEXTS:
                        ctx_k = method_params.get("context_k", ctx_cfg.get("k", 10))
                        ctx_threads = ctx_cfg.get("threads", 16)
                        ctx_map = retrieve_contexts_for_queries(
                            queries_file, ds,
                            k=ctx_k, threads=ctx_threads,
                        )
                    results = method.reformulate_batch(queries, ctx_map=ctx_map)
                    save_queries_tsv(results, reform_file)
                else:
                    print(f"      [CACHED] {reform_file}")

                # ── Step 2: Retrieve ────────────────────────────────────
                run_dir = reform_dir / "runs"
                for ret in retrievers:
                    run_file = run_dir / f"{ds}.{ret}.run"
                    if run_file.exists():
                        print(f"      [CACHED] {run_file}")
                        continue

                    query_prefix = ""
                    remove_query = False
                    if ret == "bge":
                        query_prefix = "Represent this sentence for searching relevant passages:"
                        if DATASETS.get(ds, {}).get("group") == "beir":
                            remove_query = True

                    try:
                        run_retrieval(
                            queries_tsv=reform_file,
                            dataset=ds,
                            retriever=ret,
                            output_run=run_file,
                            hits=ret_cfg.get("hits", 1000),
                            threads=ret_cfg.get("threads", 16),
                            batch_size=ret_cfg.get("batch_size", 512),
                            remove_query=remove_query,
                            query_prefix=query_prefix,
                        )
                    except Exception as e:
                        print(f"      [ERROR] {ret}: {e}")

                # ── Step 3: Evaluate ────────────────────────────────────
                for ret in retrievers:
                    run_file = run_dir / f"{ds}.{ret}.run"
                    if not run_file.exists():
                        continue

                    qrels = None
                    if ds == "dlhard" and args.dlhard_qrels:
                        qrels = args.dlhard_qrels

                    try:
                        metrics = evaluate(run_file, ds, qrels_path=qrels)
                        key = f"{llm_name}/{method_name}/{ds}/{ret}"
                        all_results[key] = metrics
                        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                        print(f"      [{ret.upper()}] {metric_str}")
                    except Exception as e:
                        print(f"      [ERROR] eval {ret}: {e}")

    # ── Save consolidated results ───────────────────────────────────────
    results_file = out_root / "all_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_file}")


if __name__ == "__main__":
    main()

