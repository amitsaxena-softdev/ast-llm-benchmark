# Benchmarking the Evaluators: AST vs. LLM-as-a-Judge

A research pipeline that compares **deterministic AST parsing** against **LLM-based code evaluation** as methods for detecting algorithmic techniques in competitive programming solutions.

The core question: when an LLM judges whether a solution uses a particular technique (e.g. recursion, list comprehension), how accurate is it compared to a ground-truth structural analysis of the code?

## Overview

The pipeline runs over the [NeoCoder](https://github.com/JHU-CLSP/NeoCoder) dataset (198 problems, 4725 Python solutions from Codeforces) and proceeds in five phases:

| Phase | Description |
|-------|-------------|
| 1 | Load dataset (downloads from NeoCoder GitHub if not cached) |
| 2 | Parse every solution with Python's `ast` module — produces 100%-accurate binary labels |
| 3 | Send solutions to an LLM judge (Llama-3-8B via Groq, or use pre-computed GPT-4 labels) |
| 4 | *(Optional)* DeBERTa embedding analysis — measures semantic similarity vs. structural divergence |
| 5 | Compute precision/recall/F1/FPR/FNR per technique; generate CSV, heatmaps, and a Markdown report |

## Techniques Evaluated

`for_loop` · `while_loop` · `recursion` · `list_comprehension` · `lambda` · `sorting`

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

`.env` is loaded automatically at startup via `python-dotenv`. The only required value is:

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Only for live Llama judge | Free API key from [console.groq.com](https://console.groq.com). Not needed when using `--skip-llm`. |

> **Never commit `.env`** — it contains secrets. Only commit `.env.example`.

## Usage

```bash
# Full pipeline — AST + Llama-3-8B judge + DeBERTa embeddings
python main.py

# Skip Llama; use pre-computed GPT-4 labels from the dataset instead
python main.py --skip-llm

# Skip both Llama and the heavy DeBERTa embedding pass
python main.py --skip-llm --skip-embeddings

# Re-run only the reporting step using cached labels from a prior run
python main.py --skip-llm --skip-embeddings --skip-ast

# Write outputs to a custom directory
python main.py --output-dir my_results
```

## Outputs

All files land in `results/` (or `--output-dir`):

| File | Contents |
|------|----------|
| `ast_labels.json` | Ground-truth binary labels per solution |
| `llm_labels.json` | LLM judge binary labels per solution |
| `gpt4_binary_labels.json` | GPT-4 labels converted to binary format |
| `llm_metrics.csv` | Per-technique accuracy/precision/recall/F1/FPR/FNR for Llama |
| `gpt4_metrics.csv` | Same metrics for GPT-4 |
| `llm_heatmap.png` | Heatmap of LLM judge error rates |
| `gpt4_heatmap.png` | Heatmap of GPT-4 error rates |
| `report.md` | Full Markdown report with all tables and conclusions |

## Key Results (Llama-3-8B vs. AST Ground Truth)

| Technique | Accuracy | F1 | FPR (hallucination) | FNR (miss rate) |
|-----------|----------|----|---------------------|-----------------|
| for_loop | 0.976 | 0.988 | 0.468 | 0.002 |
| while_loop | 0.987 | 0.932 | 0.000 | 0.125 |
| recursion | 0.998 | 0.286 | 0.002 | 0.333 |
| list_comprehension | 0.891 | 0.000 | 0.000 | 1.000 |
| lambda | 0.971 | 0.000 | 0.000 | 1.000 |
| sorting | 0.962 | 0.803 | 0.001 | 0.321 |
| **Macro avg** | **0.964** | **0.502** | **0.079** | **0.464** |

High accuracy is misleading due to class imbalance. The macro-average F1 of 0.50 reveals substantial miss rates for rare techniques (`lambda`, `list_comprehension`), while `for_loop` inflates the false-positive rate due to frequency bias.

## Troubleshooting

**`NumPy 2.x detected` error at startup** — torch, scikit-learn, and transformers require NumPy < 2. Downgrade it:
```bash
pip install "numpy<2"
```

**`AttributeError: 'NoneType' object has no attribute 'endswith'` in DeBERTa tokenizer** — a bug in certain transformers versions with the fast tokenizer. Already worked around in code via `use_fast=False`; if you still see it, upgrade transformers:
```bash
pip install --upgrade transformers
```

**Groq rate-limit errors** — the free tier allows ~30 RPM. Reduce `llm_max_solutions` in [config.py](config.py) or use `--skip-llm` to avoid API calls altogether.

## Project Structure

```
ast_analysis/      # AST visitor — deterministic technique detection
llm_judge/         # LLM evaluator (Groq/Llama) + GPT-4 label converter
embeddings/        # DeBERTa embedding analyzer
data/              # Dataset loader (downloads + caches NeoCoder)
reporting/         # Metrics computation and report generation
datasets/          # Cached dataset JSON files
results/           # Generated outputs
config.py          # Central configuration (models, techniques, thresholds)
main.py            # Pipeline orchestrator
```
