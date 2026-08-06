#!/usr/bin/env python3
"""
Benchmarking the Evaluators: AST vs. LLM-as-a-Judge
=====================================================
Top-level pipeline orchestrator.

Usage
-----
# Full pipeline (requires GROQ_API_KEY for the Llama judge)
python main.py

# Skip Llama — compare GPT-4 labels from the dataset against AST ground truth
python main.py --skip-llm

# Skip both Llama and the heavy DeBERTa embedding pass
python main.py --skip-llm --skip-embeddings

# Use a previously cached run (re-run only the reporting step)
python main.py --skip-llm --skip-embeddings --skip-ast

# Change output directory
python main.py --output-dir my_results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
if tuple(int(x) for x in np.__version__.split(".")[:2]) >= (2, 0):
    sys.exit(
        f"NumPy {np.__version__} detected. torch/sklearn/transformers require NumPy <2.\n"
        "Fix: pip install 'numpy<2'"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args():
    p = argparse.ArgumentParser(description="AST vs LLM-as-a-Judge benchmark")
    p.add_argument("--skip-llm", action="store_true",
                   help="Skip Llama-3-8B evaluation; use GPT-4 labels from dataset as the LLM judge")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="Skip DeBERTa embedding analysis (saves time/RAM)")
    p.add_argument("--skip-ast", action="store_true",
                   help="Reload AST labels from a previous run (results/ast_labels.json must exist)")
    p.add_argument("--output-dir", default="results",
                   help="Directory for all output files (default: results/)")
    return p.parse_args()


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1 — Data ingestion
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 1 — Loading NeoCoder dataset")
    logger.info("=" * 60)

    from config import Config
    from data.loader import load_dataset

    cfg = Config()
    solutions, gpt4_labels = load_dataset(cfg)

    # ------------------------------------------------------------------
    # Phase 2 — Ground truth AST parsing
    # ------------------------------------------------------------------
    ast_labels_path = output_dir / "ast_labels.json"

    if args.skip_ast and ast_labels_path.exists():
        logger.info("PHASE 2 — Reloading cached AST labels")
        from ast_analysis.visitor import ASTAnalyzer
        ast_labels = ASTAnalyzer.load(ast_labels_path)
    else:
        logger.info("=" * 60)
        logger.info("PHASE 2 — Deterministic AST parsing (ground truth)")
        logger.info("=" * 60)
        from ast_analysis.visitor import ASTAnalyzer
        analyzer = ASTAnalyzer(cfg.techniques)
        ast_labels = analyzer.analyze_all(solutions)
        analyzer.save(ast_labels, ast_labels_path)

    # ------------------------------------------------------------------
    # Phase 3 — LLM-as-a-Judge evaluation
    # ------------------------------------------------------------------
    llm_labels_path = output_dir / "llm_labels.json"

    if args.skip_llm:
        logger.info("=" * 60)
        logger.info("PHASE 3 — Converting GPT-4 labels (from dataset) as LLM judge")
        logger.info("=" * 60)
        from llm_judge.evaluator import convert_gpt4_labels
        llm_labels = convert_gpt4_labels(
            gpt4_labels, cfg.techniques, cfg.technique_label_map
        )
        # Save so the reporter can use it
        llm_labels_path.write_text(json.dumps(llm_labels, indent=2), encoding="utf-8")
    else:
        if llm_labels_path.exists():
            logger.info("PHASE 3 — Reloading cached Llama labels")
            from llm_judge.evaluator import LLMJudge
            llm_labels = LLMJudge.load(llm_labels_path)
        else:
            logger.info("=" * 60)
            logger.info("PHASE 3 — Llama-3-8B-Instruct judge (via Groq API)")
            logger.info("=" * 60)
            from llm_judge.evaluator import LLMJudge
            judge = LLMJudge(cfg)
            llm_labels = judge.evaluate_all(solutions, ast_labels=ast_labels)
            judge.save(llm_labels, llm_labels_path)

    # Also convert GPT-4 labels to binary format for comparison in the report
    from llm_judge.evaluator import convert_gpt4_labels
    gpt4_binary = convert_gpt4_labels(
        gpt4_labels, cfg.techniques, cfg.technique_label_map
    )
    (output_dir / "gpt4_binary_labels.json").write_text(
        json.dumps(gpt4_binary, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Phase 4 — Embedding vs. structure analysis
    # ------------------------------------------------------------------
    emb_path = output_dir / "embedding_results.json"
    embedding_results = None

    if not args.skip_embeddings:
        if emb_path.exists():
            logger.info("PHASE 4 — Reloading cached embedding results")
            from embeddings.analyzer import EmbeddingAnalyzer
            embedding_results = EmbeddingAnalyzer.load(emb_path)
        else:
            logger.info("=" * 60)
            logger.info("PHASE 4 — DeBERTa embedding vs. structure analysis")
            logger.info("=" * 60)
            from embeddings.analyzer import EmbeddingAnalyzer
            emb_analyzer = EmbeddingAnalyzer(cfg)
            results_obj = emb_analyzer.analyze(solutions, ast_labels)
            emb_analyzer.save(results_obj, emb_path)
            embedding_results = results_obj.to_dict()
    else:
        logger.info("PHASE 4 — Skipped (--skip-embeddings)")

    # ------------------------------------------------------------------
    # Phase 5 — Reporting
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 5 — Computing metrics and generating report")
    logger.info("=" * 60)

    from reporting.metrics import MetricsReporter
    reporter = MetricsReporter(cfg)
    reporter.generate(
        solutions=solutions,
        ast_labels=ast_labels,
        llm_labels=llm_labels,
        gpt4_labels=gpt4_binary,
        embedding_results=embedding_results,
        output_dir=output_dir,
    )

    logger.info("=" * 60)
    logger.info(f"Done.  All outputs written to: {output_dir.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
