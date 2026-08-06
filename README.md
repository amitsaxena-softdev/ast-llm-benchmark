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

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python version and dependencies, so setup is a single command with no manual venv/activate steps.

Install uv (skip if already installed):
```powershell
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the pinned Python version (3.11) and all locked dependencies into a project-local `.venv`:
```bash
uv sync
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

> Prefer plain `pip`? `requirements.txt` is kept in sync as a fallback: `pip install -r requirements.txt` inside your own virtual environment works too, but won't get the exact locked versions in `uv.lock`.

`.env` is loaded automatically at startup via `python-dotenv`. The only required value is:

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Only for live Llama judge | Free API key from [console.groq.com](https://console.groq.com). Not needed when using `--skip-llm`. |

> **Never commit `.env`** — it contains secrets. Only commit `.env.example`.

## Usage

Run everything through `uv run`, which uses the `.venv` created by `uv sync` automatically:

```bash
# Full pipeline — AST + Llama-3-8B judge + DeBERTa embeddings
uv run python main.py

# Skip Llama; use pre-computed GPT-4 labels from the dataset instead
uv run python main.py --skip-llm

# Skip both Llama and the heavy DeBERTa embedding pass
uv run python main.py --skip-llm --skip-embeddings

# Re-run only the reporting step using cached labels from a prior run
uv run python main.py --skip-llm --skip-embeddings --skip-ast

# Write outputs to a custom directory
uv run python main.py --output-dir my_results

# Launch the Streamlit dashboard
uv run streamlit run streamlit_app.py
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

**`NumPy 2.x detected` error at startup** — shouldn't happen via `uv sync`, since `uv.lock` pins NumPy < 2 for you. If you installed with plain `pip` instead and still hit this:
```bash
pip install "numpy<2"
```

**`AttributeError: 'NoneType' object has no attribute 'endswith'` in DeBERTa tokenizer** — a bug in certain transformers versions with the fast tokenizer. Already worked around in code via `use_fast=False`. If you still see it on a non-uv install, upgrade transformers:
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
pyproject.toml     # Dependency source of truth (used by uv)
uv.lock            # Locked, reproducible dependency versions
.python-version    # Pinned interpreter version (3.11) for uv
requirements.txt   # Fallback dependency list for non-uv/pip installs
```
