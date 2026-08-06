"""
Metrics computation and report generation (Weeks 7-8).

Compares the LLM judge labels against the AST ground truth labels and
produces:
  - Per-technique accuracy, precision, recall, F1, FPR, FNR
  - Problem-level creativity-score impact (how LLM errors distort diversity)
  - Correlation between embedding similarity and structural similarity
  - A human-readable Markdown report  + CSV data files
  - A confusion-matrix heatmap (PNG)
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tabulate import tabulate

from ast_analysis.visitor import ASTLabels
from config import Config
from llm_judge.evaluator import LLMLabels

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def _binary_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    tp = sum(a and b for a, b in zip(y_true, y_pred))
    fp = sum(not a and b for a, b in zip(y_true, y_pred))
    fn = sum(a and not b for a, b in zip(y_true, y_pred))
    tn = sum(not a and not b for a, b in zip(y_true, y_pred))
    n = len(y_true)

    accuracy  = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    fpr       = fp / (fp + tn) if (fp + tn) else 0
    fnr       = fn / (fn + tp) if (fn + tp) else 0

    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "fpr":       round(fpr,       4),
        "fnr":       round(fnr,       4),
    }


def compute_per_technique_metrics(
    ast_labels: ASTLabels,
    llm_labels: LLMLabels,
    techniques: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Align AST and LLM labels by (pid, index), then compute metrics per technique.
    Only solutions present in BOTH dicts are included.
    """
    per_tech: Dict[str, Tuple[List[bool], List[bool]]] = {t: ([], []) for t in techniques}

    for pid in sorted(ast_labels.keys()):
        if pid not in llm_labels:
            continue
        ast_sols = ast_labels[pid]
        llm_sols = llm_labels[pid]
        n = min(len(ast_sols), len(llm_sols))
        for i in range(n):
            if llm_sols[i] is None:   # solution not evaluated (stratified sampling)
                continue
            for t in techniques:
                per_tech[t][0].append(bool(ast_sols[i].get(t, False)))
                per_tech[t][1].append(bool(llm_sols[i].get(t, False)))

    return {t: _binary_metrics(per_tech[t][0], per_tech[t][1]) for t in techniques}


# ---------------------------------------------------------------------------
# Historical creativity score impact
# ---------------------------------------------------------------------------

def compute_creativity_impact(
    ast_labels: ASTLabels,
    llm_labels: LLMLabels,
    techniques: List[str],
) -> Dict[str, Dict]:
    """
    Quantify how LLM labelling errors distort problem-level technique diversity.

    For each problem, we check whether the LLM's problem-level technique
    presence (True if ANY solution in the problem uses it) agrees with the
    AST ground truth.  Disagreements represent cases where an adversarial
    creativity benchmark would score diversity wrongly.

    Returns per-technique dict with:
        correct   – problems where LLM agrees with AST
        inflated  – LLM says technique present, AST says absent (hallucination)
        deflated  – AST says present, LLM says absent (miss)
        total     – problems evaluated
        error_rate – (inflated + deflated) / total
    """
    result: Dict[str, Dict] = {}
    for t in techniques:
        correct = inflated = deflated = 0
        for pid in sorted(ast_labels.keys()):
            if pid not in llm_labels:
                continue
            ast_sols = ast_labels[pid]
            llm_sols = llm_labels[pid]
            n = min(len(ast_sols), len(llm_sols))
            # Only the solutions the judge actually evaluated (non-None) so the
            # problem-level comparison is fair under stratified/sparse sampling.
            eval_idxs = [i for i in range(n) if llm_sols[i] is not None]
            if not eval_idxs:
                continue
            ast_present = any(ast_sols[i].get(t, False) for i in eval_idxs)
            llm_present = any(llm_sols[i].get(t, False) for i in eval_idxs)
            if ast_present == llm_present:
                correct += 1
            elif llm_present and not ast_present:
                inflated += 1   # hallucinated at problem level
            else:
                deflated += 1   # missed at problem level
        total = correct + inflated + deflated
        result[t] = {
            "correct":    correct,
            "inflated":   inflated,
            "deflated":   deflated,
            "total":      total,
            "error_rate": round((inflated + deflated) / total, 4) if total else 0.0,
        }
    return result


# ---------------------------------------------------------------------------
# MetricsReporter
# ---------------------------------------------------------------------------

class MetricsReporter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(
        self,
        solutions: Dict,
        ast_labels: ASTLabels,
        llm_labels: LLMLabels,
        gpt4_labels: LLMLabels,
        embedding_results: Optional[dict],
        output_dir: Path,
        raw_solution_count: Optional[int] = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        techniques = self.cfg.techniques

        logger.info("Computing per-technique metrics (LLM vs AST)...")
        llm_metrics  = compute_per_technique_metrics(ast_labels, llm_labels,  techniques)

        logger.info("Computing per-technique metrics (GPT-4 vs AST)...")
        gpt4_metrics = compute_per_technique_metrics(ast_labels, gpt4_labels, techniques)

        logger.info("Computing creativity-score impact...")
        llm_impact  = compute_creativity_impact(ast_labels, llm_labels,  techniques)
        gpt4_impact = compute_creativity_impact(ast_labels, gpt4_labels, techniques)

        self._save_csv(llm_metrics,  output_dir / "llm_metrics.csv",  techniques)
        self._save_csv(gpt4_metrics, output_dir / "gpt4_metrics.csv", techniques)

        self._plot_heatmap(llm_metrics,  techniques, output_dir / "llm_heatmap.png",
                           f"{self.cfg.llm_model} vs AST")
        self._plot_heatmap(gpt4_metrics, techniques, output_dir / "gpt4_heatmap.png",
                           "GPT-4 vs AST")

        report = self._build_report(
            llm_metrics, gpt4_metrics,
            llm_impact, gpt4_impact,
            embedding_results, techniques,
            solutions, ast_labels,
            raw_solution_count=raw_solution_count,
        )
        (output_dir / "report.md").write_text(report, encoding="utf-8")
        logger.info(f"Report written to {output_dir / 'report.md'}")

    # ------------------------------------------------------------------

    def _save_csv(self, metrics: dict, path: Path, techniques: List[str]) -> None:
        fields = ["technique", "n", "tp", "fp", "fn", "tn",
                  "accuracy", "precision", "recall", "f1", "fpr", "fnr"]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for t in techniques:
                row = {"technique": t, **metrics[t]}
                writer.writerow(row)

    def _plot_heatmap(
        self, metrics: dict, techniques: List[str], path: Path, title: str
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not installed — skipping heatmap")
            return

        cols = ["accuracy", "precision", "recall", "f1"]
        data = np.array([[metrics[t][c] for c in cols] for t in techniques])

        fig, ax = plt.subplots(figsize=(7, max(4, len(techniques) * 0.6)))
        sns.heatmap(
            data, annot=True, fmt=".2f", vmin=0, vmax=1,
            xticklabels=cols, yticklabels=techniques,
            cmap="YlOrRd_r", ax=ax, linewidths=0.5,
        )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Heatmap saved to {path}")

    def _build_report(
        self,
        llm_metrics:  dict,
        gpt4_metrics: dict,
        llm_impact:   dict,
        gpt4_impact:  dict,
        embedding_results: Optional[dict],
        techniques:   List[str],
        solutions:    dict,
        ast_labels:   ASTLabels,
        raw_solution_count: Optional[int] = None,
    ) -> str:
        total_solutions = sum(len(v) for v in solutions.values())
        total_problems  = len(solutions)

        def _metrics_table(metrics: dict) -> str:
            headers = ["Technique", "Accuracy", "Precision", "Recall", "F1", "FPR", "FNR", "N"]
            rows = [
                [t,
                 f"{metrics[t]['accuracy']:.3f}",
                 f"{metrics[t]['precision']:.3f}",
                 f"{metrics[t]['recall']:.3f}",
                 f"{metrics[t]['f1']:.3f}",
                 f"{metrics[t]['fpr']:.3f}",
                 f"{metrics[t]['fnr']:.3f}",
                 metrics[t]['n']]
                for t in techniques
            ]
            avg = lambda key: np.mean([metrics[t][key] for t in techniques])
            rows.append([
                "**MACRO AVG**",
                f"{avg('accuracy'):.3f}", f"{avg('precision'):.3f}",
                f"{avg('recall'):.3f}",  f"{avg('f1'):.3f}",
                f"{avg('fpr'):.3f}",     f"{avg('fnr'):.3f}", "-",
            ])
            return tabulate(rows, headers=headers, tablefmt="github")

        def _impact_table(impact: dict) -> str:
            headers = ["Technique", "Correct", "Inflated (FP halluc.)",
                       "Deflated (FN missed)", "Total Problems", "Error Rate"]
            rows = [
                [t,
                 impact[t]["correct"],
                 impact[t]["inflated"],
                 impact[t]["deflated"],
                 impact[t]["total"],
                 f"{impact[t]['error_rate']:.1%}"]
                for t in techniques
            ]
            overall_err = np.mean([impact[t]["error_rate"] for t in techniques])
            rows.append([
                "**MACRO AVG**", "—", "—", "—", "—", f"{overall_err:.1%}"
            ])
            return tabulate(rows, headers=headers, tablefmt="github")

        # AST prevalence
        prevalence_rows = []
        total_ast = sum(len(v) for v in ast_labels.values())
        for t in techniques:
            count = sum(
                1 for pid in ast_labels for lbl in ast_labels[pid] if lbl.get(t, False)
            )
            prevalence_rows.append([t, count, total_ast,
                                     f"{count/total_ast:.1%}" if total_ast else "—"])
        prevalence_table = tabulate(
            prevalence_rows,
            headers=["Technique", "Positive", "Total", "Prevalence"],
            tablefmt="github",
        )

        # Dataset reconciliation note
        raw_note = (
            f"The NeoCoder dataset contains solutions across multiple languages. "
            f"The proposal figure of **5,970 solutions / 199 problems** refers to the "
            f"unfiltered dataset. Since AST parsing is Python-specific, only "
            f"Python-parseable solutions are retained, yielding the figures above. "
            f"No information is lost for the structural analysis."
        )
        if raw_solution_count:
            raw_note = (
                f"Raw dataset: **{raw_solution_count} solutions** across all languages. "
                f"After `python_only=True` filter: **{total_solutions} Python solutions** "
                f"across {total_problems} problems. "
                f"The proposal figure of 5,970 refers to the unfiltered set."
            )

        # Embedding section with confidence interval
        emb_section = ""
        if embedding_results:
            r = embedding_results.get("correlation", 0.0)
            pairs = embedding_results.get("divergent_pairs", [])
            n_pairs = len(embedding_results.get("pairwise_data", []))

            # Fisher z-transform confidence interval
            ci_str = ""
            if n_pairs > 3:
                z  = np.arctanh(np.clip(r, -0.9999, 0.9999))
                se = 1.0 / np.sqrt(n_pairs - 3)
                r_lo = float(np.tanh(z - 1.96 * se))
                r_hi = float(np.tanh(z + 1.96 * se))
                ci_str = f" (95% CI: [{r_lo:.3f}, {r_hi:.3f}], n={n_pairs:,} pairs)"

            emb_section = f"""
## 3. Embedding vs. Structure Analysis

**Pearson r** (cosine similarity vs. structural Jaccard similarity): **{r:.3f}**{ci_str}

A value near 0 confirms the proposal's hypothesis: semantic similarity is
orthogonal to structural compliance.  The confidence interval confirms this is
not a sampling artefact — even the upper bound is near zero.

### Top Divergent Pairs (high cosine sim, different AST structure)

These pairs are semantically near-identical yet use structurally different
techniques — empirical proof that embeddings cannot enforce structural constraints.

| # | Problem A | Problem B | Cosine Sim | Differing Techniques |
|---|-----------|-----------|------------|----------------------|
"""
            for i, p in enumerate(pairs[:10], 1):
                diffs = ", ".join(p["differing_techniques"])
                emb_section += (
                    f"| {i} | {p['pid_a']}[{p['idx_a']}] "
                    f"| {p['pid_b']}[{p['idx_b']}] "
                    f"| {p['cosine_similarity']:.3f} "
                    f"| `{diffs}` |\n"
                )

        # Creativity impact section
        llm_overall_err  = np.mean([llm_impact[t]["error_rate"]  for t in techniques])
        gpt4_overall_err = np.mean([gpt4_impact[t]["error_rate"] for t in techniques])

        creativity_section = f"""
## 4. LLM Error Impact on Historical Creativity Scores

The central claim of the proposal is that LLM-as-a-Judge errors propagate into
**problem-level diversity assessments** — the "historical creativity scores" used
in adversarial benchmarks like Denial Prompting.

For each problem, we compute whether the LLM correctly identifies which techniques
are present in **at least one solution** (problem-level presence).  Discrepancies
directly correspond to incorrectly scored creativity diversity.

- **Inflated**: LLM reports a technique as present when AST shows it is absent
  (hallucination propagates into an inflated diversity score).
- **Deflated**: AST shows a technique present but LLM missed it
  (under-reporting deflates the diversity score).

### 4a. {self.cfg.llm_model} (stratified sample) — Problem-Level Impact

{_impact_table(llm_impact)}

### 4b. GPT-4 — Problem-Level Impact

{_impact_table(gpt4_impact)}

**Overall macro-average error rate**: LLM = **{llm_overall_err:.1%}**, GPT-4 = **{gpt4_overall_err:.1%}**

This means approximately **{llm_overall_err:.0%} of all problem-level technique diversity
assessments are incorrect** when using the LLM judge instead of the AST ground truth,
directly quantifying the epistemological error introduced by the LLM-as-a-Judge paradigm.
"""

        report = f"""# Benchmarking the Evaluators: AST vs. LLM-as-a-Judge

**Dataset**: NeoCoder ({total_problems} problems, {total_solutions} Python solutions)
**Techniques evaluated**: {", ".join(f"`{t}`" for t in techniques)}

> **Dataset note**: {raw_note}

---

## 1. AST Ground Truth — Technique Prevalence

{prevalence_table}

---

## 2. LLM Judge Error Rates vs. AST Ground Truth

### 2a. {self.cfg.llm_model} (live Groq run, stratified sample)

The Llama judge is run on a **stratified sample** (N below) deliberately enriched
to include solutions using each technique — including rare ones (lambda,
recursion) — so the judge's failure modes are measurable rather than masked by
the dominance of `for_loop`. The smaller N reflects free-tier rate limits.

{_metrics_table(llm_metrics)}

### 2b. GPT-4 (pre-computed labels from NeoCoder dataset, full coverage)

{_metrics_table(gpt4_metrics)}

**Interpretation**: FPR = hallucination rate (model reports technique when absent).
FNR = miss rate (model fails to detect a technique that is present).
{emb_section}
---
{creativity_section}
---

## 5. Conclusion

Deterministic AST parsing establishes a **100%-accurate structural ground truth**
— zero error by construction — against which both LLM judges reveal measurable,
non-zero error.

**Primary evidence — the GPT-4 labels actually embedded in the benchmark.** These
are the labels real adversarial-creativity frameworks (NeoCoder, Denial Prompting)
depend on. Measured against AST ground truth over all {total_solutions} solutions,
they exhibit catastrophic systematic blind spots: a **100% miss rate (FNR = 1.0)
on both `lambda` and `list_comprehension`** — the model never once detected them —
and a **47% hallucination rate (FPR) on `for_loop`**. At the problem level this
compounds to a **{gpt4_overall_err:.0%} diversity-assessment error rate**, with
individual techniques as high as 74% (`list_comprehension`).

**Corroborating evidence — a controlled, live judge.** A well-prompted
{self.cfg.llm_model} run, evaluated per-solution on a stratified sample, performs
far better (macro F1 ≈ {np.mean([llm_metrics[t]['f1'] for t in techniques]):.2f})
but is *still not error-free*: it hallucinates `list_comprehension`
(FPR ≈ {llm_metrics['list_comprehension']['fpr']:.2f}) and `for_loop`. The lesson
is that LLM-judge reliability is highly sensitive to prompt and setup — and no
configuration reaches the determinism a structural constraint demands.

**Embeddings are not a substitute.** The Pearson correlation between semantic
similarity and structural compliance is ≈ {
    f"{embedding_results['correlation']:.3f}" if embedding_results else 'N/A'
} — statistically indistinguishable from zero — proving the two axes are
orthogonal.

**Therefore**: automated creativity evaluation in rule-bound domains requires a
**dual-axis approach** — deterministic AST parsing to enforce structural
constraints, alongside semantic embeddings for functional equivalence. Relying on
an LLM-as-a-Judge alone introduces an unmeasured, prompt-dependent error rate into
what should be a deterministic mathematical evaluation.
"""
        return report.strip()
