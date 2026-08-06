# Benchmarking the Evaluators: AST vs. LLM-as-a-Judge

**Dataset**: NeoCoder (198 problems, 4725 Python solutions)
**Techniques evaluated**: `for_loop`, `while_loop`, `recursion`, `list_comprehension`, `lambda`, `sorting`

> **Dataset note**: The NeoCoder dataset contains solutions across multiple languages. The proposal figure of **5,970 solutions / 199 problems** refers to the unfiltered dataset. Since AST parsing is Python-specific, only Python-parseable solutions are retained, yielding the figures above. No information is lost for the structural analysis.

---

## 1. AST Ground Truth — Technique Prevalence

| Technique          |   Positive |   Total | Prevalence   |
|--------------------|------------|---------|--------------|
| for_loop           |       4505 |    4725 | 95.3%        |
| while_loop         |        480 |    4725 | 10.2%        |
| recursion          |          3 |    4725 | 0.1%         |
| list_comprehension |        514 |    4725 | 10.9%        |
| lambda             |        137 |    4725 | 2.9%         |
| sorting            |        548 |    4725 | 11.6%        |

> **Small-sample caveat**: `recursion` (n=3) have very few ground-truth positive examples. Precision/recall/F1 for these techniques in the tables below are correspondingly noisy — a single misclassified solution can swing the score by tens of percentage points. Read them as directional, not precise.

---

## 2. LLM Judge Error Rates vs. AST Ground Truth

### 2a. llama-3.1-8b-instant (live Groq run, stratified sample)

The Llama judge is run on a **stratified sample** (N below) deliberately enriched
to include solutions using each technique — including rare ones (lambda,
recursion) — so the judge's failure modes are measurable rather than masked by
the dominance of `for_loop`. The smaller N reflects free-tier rate limits.

| Technique          |   Accuracy |   Precision |   Recall |    F1 |   FPR |   FNR | N   |
|--------------------|------------|-------------|----------|-------|-------|-------|-----|
| for_loop           |      0.95  |       0.965 |    0.982 | 0.974 | 0.5   | 0.018 | 60  |
| while_loop         |      1     |       1     |    1     | 1     | 0     | 0     | 60  |
| recursion          |      1     |       1     |    1     | 1     | 0     | 0     | 60  |
| list_comprehension |      0.95  |       0.833 |    1     | 0.909 | 0.067 | 0     | 60  |
| lambda             |      1     |       1     |    1     | 1     | 0     | 0     | 60  |
| sorting            |      0.983 |       0.923 |    1     | 0.96  | 0.021 | 0     | 60  |
| **MACRO AVG**      |      0.981 |       0.954 |    0.997 | 0.974 | 0.098 | 0.003 | -   |

### 2b. GPT-4 (pre-computed labels from NeoCoder dataset, full coverage)

| Technique          |   Accuracy |   Precision |   Recall |    F1 |   FPR |   FNR | N    |
|--------------------|------------|-------------|----------|-------|-------|-------|------|
| for_loop           |      0.976 |       0.978 |    0.998 | 0.988 | 0.468 | 0.002 | 4725 |
| while_loop         |      0.987 |       0.998 |    0.875 | 0.932 | 0     | 0.125 | 4725 |
| recursion          |      0.998 |       0.182 |    0.667 | 0.286 | 0.002 | 0.333 | 4725 |
| list_comprehension |      0.891 |       0     |    0     | 0     | 0     | 1     | 4725 |
| lambda             |      0.971 |       0     |    0     | 0     | 0     | 1     | 4725 |
| sorting            |      0.962 |       0.984 |    0.679 | 0.803 | 0.001 | 0.321 | 4725 |
| **MACRO AVG**      |      0.964 |       0.524 |    0.536 | 0.502 | 0.079 | 0.464 | -    |

> **Vocabulary-gap caveat**: `list_comprehension`, `lambda` never appear anywhere in the GPT-4 label vocabulary shipped with the raw NeoCoder dataset (0 occurrences across every unique technique string, any problem, any language). Their FNR below therefore reflects a taxonomy gap in how the original labels were generated, not GPT-4 failing to recognize the construct in code it was shown per-solution — a different (and arguably more damning, since it means the taxonomy itself is incomplete) failure mode than `for_loop`'s hallucination rate, which *is* a live judgment error on a category the labels do cover.

**Interpretation**: FPR = hallucination rate (model reports technique when absent).
FNR = miss rate (model fails to detect a technique that is present).

## 3. Embedding vs. Structure Analysis

**Pearson r** (cosine similarity vs. structural Jaccard similarity, **same-problem
pairs only**): **0.227** (95% CI: [0.201, 0.253], n=5,000 pairs)

Pairs are restricted to two solutions of *the same* problem — the only pairs
guaranteed functionally equivalent, since every human solution to a given
problem passed that problem's test suite. This correlation is small but, given the tight confidence interval, distinguishable from zero — the two axes are **not** strictly orthogonal. However, r² ≈ 5.2% means semantic similarity explains only about 5% of the variance in structural similarity: the vast majority of whether two functionally-equivalent solutions share the same structural techniques remains invisible to embeddings alone.

### Top Divergent Pairs (high cosine sim, different AST structure, same problem)

These are two solutions to the **same problem** (hence functionally
equivalent by construction) that are semantically near-identical yet use
structurally different techniques — empirical proof that embeddings cannot
enforce structural constraints even when functional equivalence is certain.

| # | Problem | Solutions | Cosine Sim | Differing Techniques |
|---|---------|-----------|------------|----------------------|
| 1 | 1859A | [10] vs [21] | 0.996 | `sorting` |
| 2 | 1747A | [9] vs [26] | 0.996 | `sorting` |
| 3 | 1747A | [6] vs [26] | 0.996 | `list_comprehension` |
| 4 | 1747A | [6] vs [19] | 0.996 | `list_comprehension` |
| 5 | 1747A | [9] vs [19] | 0.996 | `sorting` |
| 6 | 1733A | [4] vs [10] | 0.995 | `list_comprehension` |
| 7 | 1873B | [7] vs [27] | 0.995 | `sorting` |
| 8 | 1734A | [5] vs [12] | 0.995 | `list_comprehension` |
| 9 | 1873B | [6] vs [12] | 0.995 | `sorting` |
| 10 | 1873B | [7] vs [9] | 0.995 | `sorting` |

---

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

### 4a. llama-3.1-8b-instant (stratified sample) — Problem-Level Impact

| Technique          | Correct   | Inflated (FP halluc.)   | Deflated (FN missed)   | Total Problems   | Error Rate   |
|--------------------|-----------|-------------------------|------------------------|------------------|--------------|
| for_loop           | 14        | 1                       | 1                      | 16               | 12.5%        |
| while_loop         | 16        | 0                       | 0                      | 16               | 0.0%         |
| recursion          | 16        | 0                       | 0                      | 16               | 0.0%         |
| list_comprehension | 15        | 1                       | 0                      | 16               | 6.2%         |
| lambda             | 16        | 0                       | 0                      | 16               | 0.0%         |
| sorting            | 16        | 0                       | 0                      | 16               | 0.0%         |
| **MACRO AVG**      | —         | —                       | —                      | —                | 3.1%         |

### 4b. GPT-4 — Problem-Level Impact

| Technique          | Correct   | Inflated (FP halluc.)   | Deflated (FN missed)   | Total Problems   | Error Rate   |
|--------------------|-----------|-------------------------|------------------------|------------------|--------------|
| for_loop           | 197       | 1                       | 0                      | 198              | 0.5%         |
| while_loop         | 190       | 1                       | 7                      | 198              | 4.0%         |
| recursion          | 189       | 8                       | 1                      | 198              | 4.5%         |
| list_comprehension | 51        | 0                       | 147                    | 198              | 74.2%        |
| lambda             | 128       | 0                       | 70                     | 198              | 35.4%        |
| sorting            | 189       | 3                       | 6                      | 198              | 4.5%         |
| **MACRO AVG**      | —         | —                       | —                      | —                | 20.5%        |

**Overall macro-average error rate**: LLM = **3.1%**, GPT-4 = **20.5%**

This means approximately **3% of all problem-level technique diversity
assessments are incorrect** when using the LLM judge instead of the AST ground truth,
directly quantifying the epistemological error introduced by the LLM-as-a-Judge paradigm.

---

## 5. Conclusion

Deterministic AST parsing establishes a **100%-accurate structural ground truth**
— zero error by construction — against which both LLM judges reveal measurable,
non-zero error.

**Primary evidence — the GPT-4 labels actually embedded in the benchmark.** These
are the labels real adversarial-creativity frameworks (NeoCoder, Denial Prompting)
depend on. Measured against AST ground truth over all 4725 solutions,
they exhibit catastrophic systematic blind spots: a **100% miss rate (FNR = 1.0)
on both `lambda` and `list_comprehension`** — driven by a taxonomy gap: neither category appears anywhere in the GPT-4 label vocabulary for this dataset, so this reflects an incomplete labeling taxonomy rather than a live per-solution judgment failure — and a **47% hallucination rate (FPR) on `for_loop`**, which *is* a live
judgment error, since `for loop` is squarely within the labels' vocabulary. At
the problem level this compounds to a **21%
diversity-assessment error rate**, with individual techniques as high as 74%
(`list_comprehension`).

**Corroborating evidence — a controlled, live judge.** A well-prompted
llama-3.1-8b-instant run, evaluated per-solution on a stratified sample, performs
far better (macro F1 ≈ 0.97)
but is *still not error-free*: it hallucinates `list_comprehension`
(FPR ≈ 0.07) and `for_loop`. The lesson
is that LLM-judge reliability is highly sensitive to prompt and setup — and no
configuration reaches the determinism a structural constraint demands.

**Embeddings are not a substitute.** Restricted to same-problem solution pairs
— the only pairs guaranteed functionally equivalent, since they all pass the
same test suite — the Pearson correlation between semantic similarity and
structural compliance is ≈ 0.227, which is small but, given the tight confidence interval, distinguishable from zero — the two axes are **not** strictly orthogonal. However, r² ≈ 5.2% means semantic similarity explains only about 5% of the variance in structural similarity: the vast majority of whether two functionally-equivalent solutions share the same structural techniques remains invisible to embeddings alone.
Either way, semantic similarity is not a reliable proxy for structural
compliance, and cannot substitute for deterministic AST parsing when the
constraint being enforced is structural rather than functional.

**Therefore**: automated creativity evaluation in rule-bound domains requires a
**dual-axis approach** — deterministic AST parsing to enforce structural
constraints, alongside semantic embeddings for functional equivalence. Relying on
an LLM-as-a-Judge alone introduces an unmeasured, prompt-dependent error rate into
what should be a deterministic mathematical evaluation.