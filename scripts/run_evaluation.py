#!/usr/bin/env python3
"""Step 3 — Evaluate retrieval runs and produce result tables.

Usage
-----
python scripts/run_evaluation.py \
    --run-dir outputs/gpt-4.1/genqr/runs/ \
    --datasets dl19 dl20 dlhard scifact arguana covid fiqa dbpedia news \
    --retrievers bm25 splade bge \
    --output results/gpt-4.1_genqr.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import evaluate_all, results_to_table


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval runs.")
    parser.add_argument("--run-dir", required=True, help="Directory containing .run files.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to evaluate (default: all).")
    parser.add_argument("--retrievers", nargs="+", default=None,
                        help="Retrievers to evaluate (default: all).")
    parser.add_argument("--dlhard-qrels", default=None,
                        help="Path to DL-HARD qrels file.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    args = parser.parse_args()

    qrels_overrides = {}
    if args.dlhard_qrels:
        qrels_overrides["dlhard"] = args.dlhard_qrels

    results = evaluate_all(
        run_dir=args.run_dir,
        datasets=args.datasets,
        retrievers=args.retrievers,
        qrels_overrides=qrels_overrides,
    )

    table = results_to_table(results, output_path=args.output)
    print(table)

    # Also save raw JSON
    if args.output:
        json_path = Path(args.output).with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output} and {json_path}")


if __name__ == "__main__":
    main()

