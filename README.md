# A Reproducibility Study of LLM-Based Query Reformulation

This repository provides the code, prompt templates, configurations, and evaluation scripts for reproducing the experiments described in:

> **A Reproducibility Study of LLM-Based Query Reformulation**  

## Overview

We evaluate **10 representative LLM-based query reformulation methods** under a unified and strictly controlled experimental framework:

| Family | Method | Key Idea |
|---|---|---|
| **Keyword-Level** | GenQR | N independent keyword-generation calls |
| | GenQREnsemble | 10 paraphrased instruction variants, merged keywords |
| | Q2K (Query2Keyword) | Single-call keyword list generation |
| **Document-Level** | Q2D (ZS/FS/CoT) | Pseudo-document generation (three prompting variants) |
| | QA-Expand | Sub-question decomposition → answers → filtering |
| | MUGI | Multiple pseudo-documents with adaptive query weighting |
| **Corpus-Grounded** | CSQE | Knowledge-based + context-based sentence extraction |
| | LameR | Evidence-conditioned rewriting from retrieved documents |

All methods are evaluated across:
- **4 LLMs**: GPT-4.1, GPT-4.1-nano, Qwen2.5-72B, Qwen2.5-7B
- **3 Retrievers**: BM25 (lexical), SPLADE (learned sparse), BGE (dense)
- **9 Datasets**: TREC DL 2019, DL 2020, DL-HARD, SciFact, ArguAna, COVID, FiQA, DBPedia, News

## Repository Structure

```
.
├── configs/
│   ├── default.yaml              # Experiment configuration
│   └── dataset_registry.yaml     # Pyserini index names & BM25 weights reference
├── prompts/
│   └── prompts.json              # All prompt templates (OpenAI-style messages)
├── src/
│   ├── llm_client.py             # Unified LLM client (OpenAI + OpenRouter)
│   ├── prompts.py                # Prompt bank loader
│   ├── data.py                   # Dataset metadata and query I/O
│   ├── retrieval.py              # BM25 / SPLADE / BGE via Pyserini
│   ├── evaluation.py             # nDCG@10, Recall metrics via trec_eval
│   └── methods/
│       ├── base.py               # Base class and helpers
│       ├── genqr.py              # GenQR
│       ├── genqr_ensemble.py     # GenQREnsemble
│       ├── q2k.py                # Query2Keyword
│       ├── q2d.py                # Query2Doc (ZS / FS / CoT)
│       ├── qa_expand.py          # QA-Expand
│       ├── mugi.py               # MUGI
│       ├── csqe.py               # CSQE
│       └── lamer.py              # LameR
├── scripts/
│   ├── run_reformulation.py      # Step 1: generate reformulated queries
│   ├── run_retrieval.py          # Step 2: run retrieval
│   ├── run_evaluation.py         # Step 3: evaluate
│   └── run_pipeline.py           # End-to-end pipeline
├── data/
│   └── queries/                  # Input query TSV files (qid\tquery_text)
├── artifacts/
│   ├── reformulated_queries/
│   │   └── queries.zip           # All reformulated query TSV files (zipped)
│   └── retrieval_results/        # Aggregated retrieval result tables (CSV)
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

### 1. Install dependencies

```bash
conda create -n qr-repro python=3.10 -y
conda activate qr-repro
pip install -r requirements.txt
```

### 2. Set API keys

```bash
# For GPT-4.1 / GPT-4.1-nano
export OPENAI_API_KEY="your-openai-key"

# For Qwen2.5-72B / Qwen2.5-7B (via OpenRouter)
export OPENROUTER_API_KEY="your-openrouter-key"
```

### 3. Prepare query files

Place query TSV files (format: `qid\tquery_text`) in `data/queries/`:
```
data/queries/dl19.tsv
data/queries/dl20.tsv
data/queries/dlhard.tsv
data/queries/scifact.tsv
...
```

Pyserini can also load topics from its built-in collections; see `src/data.py` for dataset identifiers.

## Running Experiments

### Full Pipeline (Recommended)

Run all methods × LLMs × datasets × retrievers:

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

Run a subset:

```bash
python scripts/run_pipeline.py \
    --methods genqr mugi q2d_zs \
    --llms gpt-4.1 qwen-72b \
    --datasets dl19 dl20 scifact \
    --retrievers bm25 bge
```

### Step-by-Step

**Step 1 — Reformulate queries:**

```bash
python scripts/run_reformulation.py \
    --method genqr \
    --llm gpt-4.1 \
    --dataset dl19 \
    --queries data/queries/dl19.tsv \
    --output outputs/gpt-4.1/genqr/dl19.tsv
```

For corpus-grounded methods (CSQE, LameR), add `--contexts-from bm25`:

```bash
python scripts/run_reformulation.py \
    --method csqe \
    --llm gpt-4.1 \
    --dataset dl19 \
    --queries data/queries/dl19.tsv \
    --output outputs/gpt-4.1/csqe/dl19.tsv \
    --contexts-from bm25
```

**Step 2 — Run retrieval:**

```bash
python scripts/run_retrieval.py \
    --queries outputs/gpt-4.1/genqr/dl19.tsv \
    --dataset dl19 \
    --retrievers bm25 splade bge \
    --output-dir outputs/gpt-4.1/genqr/runs/
```

**Step 3 — Evaluate:**

```bash
python scripts/run_evaluation.py \
    --run-dir outputs/gpt-4.1/genqr/runs/ \
    --datasets dl19 \
    --output results/gpt-4.1_genqr.csv
```

## Configuration

All hyper-parameters are in `configs/default.yaml`:

- **LLM**: model name, max tokens (256), temperature
- **Methods**: per-method parameters (number of calls, query repeats, etc.)
- **Context retrieval**: BM25 top-k settings for corpus-grounded methods (CSQE, LameR)
- **Retrieval**: hits (1000), threads, batch size
- **Datasets**: list of evaluation benchmarks

Dataset-specific Pyserini index names, topic/qrels identifiers, and BM25 weights (k1, b) are documented in `configs/dataset_registry.yaml` and coded in `src/data.py`.

## Supported LLMs

| Key | Model | Provider |
|---|---|---|
| `gpt-4.1` | GPT-4.1 | OpenAI API |
| `gpt-4.1-nano` | GPT-4.1-nano | OpenAI API |
| `qwen-72b` | Qwen2.5-72B-Instruct | OpenRouter |
| `qwen-7b` | Qwen2.5-7B-Instruct | OpenRouter |

## Evaluation Metrics

- **TREC DL** (DL 2019, DL 2020, DL-HARD): nDCG@10, Recall@1000
- **BEIR** (SciFact, ArguAna, COVID, FiQA, DBPedia, News): nDCG@10, Recall@100

## Artifacts

Pre-computed outputs are provided under `artifacts/` so that results can be verified without re-running LLM inference.

### Reformulated Queries

`artifacts/reformulated_queries/` contains `queries.zip`, an archive of the reformulated query TSV files generated by all 10 methods × 4 LLMs × 9 datasets.

Each file inside the archive follows the naming convention `<llm>_<method>_<dataset>.tsv` (e.g., `gpt-4.1_q2d_zs_dl19.tsv`) and uses the two-column TSV format:

```
qid	reformulated_query
```

### Retrieval Results

`artifacts/retrieval_results/` contains aggregated evaluation tables (CSV) — one per LLM backbone:

| File | LLM |
|---|---|
| `gpt-4.1_results.csv` | GPT-4.1 |
| `gpt-4.1-nano_results.csv` | GPT-4.1-nano |
| `qwen-72b_results.csv` | Qwen2.5-72B-Instruct |
| `qwen-7b_results.csv` | Qwen2.5-7B-Instruct |

Each CSV reports nDCG@10 and Recall@{100,1000} for every method × dataset × retriever combination.

## Prompt Templates

All prompts are in `prompts/prompts.json`. Each entry specifies an OpenAI-style message list with `{variable}` placeholders that are filled at runtime. See the file for the full set of templates used for each method.

## Output Format

Every reformulation script produces a file:

```
<qid>\t<reformulated_query>
```

These files are consumed directly by Pyserini for retrieval. The framework guarantees that tab characters within reformulated text are replaced with spaces to preserve the TSV structure.

