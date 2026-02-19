"""Evaluation module — compute nDCG@10, Recall@100/1000 via pytrec_eval."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .data import DATASETS


def evaluate(
    run_file: str | Path,
    dataset: str,
    qrels_path: Optional[str | Path] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate a TREC-format run file against qrels.

    Parameters
    ----------
    run_file : path
        TREC-format run file.
    dataset : str
        Dataset name (to look up default qrels and metric list).
    qrels_path : path, optional
        Override qrels path (e.g. for DL-HARD with a custom file).
    metrics : list[str], optional
        Override which metrics to compute (pytrec_eval names, e.g.
        ``["ndcg_cut_10", "recall_100"]``).

    Returns
    -------
    dict[str, float]
        Mapping from metric name to its aggregate value.
    """
    ds_cfg = DATASETS[dataset]

    if qrels_path is None:
        qrels = ds_cfg["qrels"]
    else:
        qrels = str(qrels_path)

    if metrics is None:
        metrics = ds_cfg["eval_metrics"]

    trec_args = ds_cfg.get("trec_args", ["-c"])

    results: Dict[str, float] = {}
    for metric in metrics:
        # pytrec_eval uses dots: ndcg_cut.10 / recall.100
        trec_metric = metric.replace("_", ".", 1) if "_" in metric else metric

        cmd = [
            "python", "-m", "pyserini.eval.trec_eval",
        ] + trec_args + [
            "-m", trec_metric,
            qrels,
            str(run_file),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"  [WARN] evaluation for {trec_metric} failed: {proc.stderr[:300]}")
            results[metric] = float("nan")
            continue

        # Parse the last numeric value from trec_eval output
        for line in proc.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "all":
                try:
                    results[metric] = float(parts[2])
                except ValueError:
                    pass

    return results


def evaluate_all(
    run_dir: str | Path,
    datasets: Optional[List[str]] = None,
    retrievers: Optional[List[str]] = None,
    qrels_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Batch-evaluate all run files in a directory.

    Expects naming convention: ``{dataset}.{retriever}.run``

    Returns
    -------
    dict
        Nested: ``results[dataset][retriever][metric] = value``.
    """
    run_dir = Path(run_dir)
    datasets = datasets or list(DATASETS.keys())
    retrievers = retrievers or ["bm25", "splade", "bge"]
    qrels_overrides = qrels_overrides or {}

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for ds in datasets:
        all_results[ds] = {}
        for ret in retrievers:
            run_file = run_dir / f"{ds}.{ret}.run"
            if not run_file.exists():
                continue
            qrels = qrels_overrides.get(ds)
            all_results[ds][ret] = evaluate(run_file, ds, qrels_path=qrels)

    return all_results


def results_to_table(results: Dict, output_path: Optional[str | Path] = None) -> str:
    """Format nested evaluation results as a readable table and optionally
    save as CSV.

    Parameters
    ----------
    results : dict
        Output of :func:`evaluate_all`.
    output_path : path, optional
        If given, write the table as CSV.

    Returns
    -------
    str
        Human-readable table string.
    """
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)

    # collect all metrics
    all_metrics = set()
    for ds_dict in results.values():
        for ret_dict in ds_dict.values():
            all_metrics.update(ret_dict.keys())
    all_metrics = sorted(all_metrics)

    # header
    header = ["Dataset", "Retriever"] + all_metrics
    writer.writerow(header)

    for ds in sorted(results.keys()):
        for ret in sorted(results[ds].keys()):
            row = [ds, ret] + [
                f"{results[ds][ret].get(m, float('nan')):.4f}" for m in all_metrics
            ]
            writer.writerow(row)

    table = buf.getvalue()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(table)

    return table

