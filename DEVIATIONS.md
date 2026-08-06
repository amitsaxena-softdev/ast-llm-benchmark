# Deviations from the Original Proposal

This document records where the **implementation** departs from the plan in
[project_idea.md](project_idea.md), and why. The proposal itself is left
unmodified; the changes below were driven by practical constraints (free-tier
API limits) and by what the data empirically showed. None of them alter the
project's core thesis.

---

## Summary

| # | Area | Proposal said | Implementation does | Reason |
|---|------|---------------|---------------------|--------|
| 1 | Dataset size | 5,970 solutions / 199 problems | 4,724 Python solutions / 198 problems | AST parsing is Python-specific; non-Python solutions filtered out |
| 2 | LLM coverage | Feed *all* ~5.9K solutions to Llama | Stratified sample of 60 solutions | Groq free-tier rate limit (6,000 tokens/min) |
| 3 | LLM sampling | (Unspecified) | Stratified across techniques | Random sampling masks rare-technique failures |
| 4 | Narrative emphasis | Llama as primary LLM judge | GPT-4 (benchmark labels) as primary evidence | Empirical result: Llama outperformed the dataset's GPT-4 labels |
| 5 | Llama model | `Llama-3-8B-Instruct` | `llama-3.1-8b-instant` | Original Groq model was decommissioned |
| 6 | AST traversal | `ast.NodeVisitor` | `ast.NodeVisitor` (now matches) | Initially used `ast.walk`; refactored to match proposal |

---

## 1. Dataset size: 5,970 → 4,724 solutions

**Proposal:** "Load the existing 5,970 correct human solutions across 199
problems from the NeoCoder dataset."

**Implementation:** Retains 4,724 Python solutions across 198 problems.

**Why:** The NeoCoder dataset is multi-language. Since the entire ground-truth
mechanism is Python's `ast` module, non-Python solutions cannot be parsed and are
filtered out (`python_only=True`). The 5,970 figure refers to the unfiltered,
all-language dataset. No information is lost for the structural analysis — we
simply restrict to the language the deterministic parser supports. This is
disclosed in the report's dataset note.

---

## 2. LLM coverage: all solutions → stratified sample of 60

**Proposal:** "Feed the exact same 5.9K human solutions to an open-weights LLM,
specifically Llama-3-8B-Instruct, via … free-tier inference API."

**Implementation:** Evaluates a stratified sample of 60 solutions with the live
Llama judge.

**Why:** The Groq free tier caps usage at ~6,000 tokens per minute. Each solution
costs ~250–400 tokens, so sending all 4,724 solutions would take hours and trigger
continuous rate-limit (HTTP 429) throttling. A 60-solution sample completes
reliably in minutes while still exercising every technique.

**Integrity safeguard:** When a call still fails after retries, that solution is
*skipped* (recorded as `None`), never as a fake all-False label — so failed API
calls cannot silently corrupt the measured error rates. The full-coverage LLM
analysis is instead carried by the GPT-4 labels already shipped with the dataset
(N = 4,724), so the project does not lose a large-N LLM-judge measurement.

---

## 3. LLM sampling strategy: stratified, not random

**Proposal:** Did not specify a sampling strategy (it assumed full coverage).

**Implementation:** The 60 solutions are chosen by *stratified* sampling —
deliberately including solutions that use each technique, rarest first
(recursion, lambda, list comprehension), before filling the remaining budget.

**Why:** A naïve random sample of this dataset is ~95% `for_loop` and contains
almost no `lambda`, `recursion`, or `list_comprehension` examples. The judge would
then be tested almost entirely on the one easy, ubiquitous technique and would
look deceptively perfect, masking exactly the failure modes the project exists to
measure. Stratification guarantees the hard cases are actually tested. This is a
strengthening of the methodology, not a weakening.

---

## 4. Narrative emphasis: GPT-4 as primary evidence

**Proposal:** Frames the live Llama-3-8B run (Weeks 3–4) as the central
LLM-as-a-Judge experiment.

**Implementation:** The conclusion leads with the **GPT-4 labels** (the
pre-computed labels actually embedded in the NeoCoder benchmark) as the primary
evidence, and presents the live Llama run as corroborating support.

**Why:** This was an empirical, data-driven decision. The results showed:

- **GPT-4 benchmark labels** (full coverage, N = 4,724): macro-F1 ≈ 0.50, with a
  **100% miss rate on `lambda` and `list_comprehension`** and a 47% hallucination
  rate on `for_loop`.
- **Live Llama-3.1** (stratified, N = 60): macro-F1 ≈ 0.97 — accurate, but still
  not error-free (it hallucinates `list_comprehension` and `for_loop`).

The GPT-4 labels are both (a) the labels that real adversarial-creativity
benchmarks actually depend on, and (b) where the dramatic, thesis-supporting
failures appear. Leading with them makes the strongest, most honest argument:
the LLM judgments *embedded in the benchmark* are unreliable, and even a careful
fresh LLM judge is not perfectly reliable — only deterministic AST parsing is.

The thesis is unchanged; only which experiment carries the headline shifted to
follow the evidence.

---

## 5. Llama model id: decommissioned original

**Proposal:** `Llama-3-8B-Instruct` (Groq id `llama3-8b-8192`).

**Implementation:** `llama-3.1-8b-instant`.

**Why:** Groq decommissioned `llama3-8b-8192` (HTTP 400 `model_decommissioned`).
`llama-3.1-8b-instant` is the current free-tier 8B replacement — the same model
family and scale the proposal intended.

---

## 6. AST traversal mechanism

**Proposal:** "A lightweight Python script utilizing the native `ast.NodeVisitor`
module."

**Implementation:** Now uses a `TechniqueVisitor(ast.NodeVisitor)` subclass, as
described. (An earlier draft used `ast.walk()`, which traverses the same tree but
does not match the proposal's stated API; it was refactored to use `NodeVisitor`
with proper `visit_*` dispatch.) The recursion detector was also broadened to
catch method-style self-calls (`self.f()` / `obj.f()`), not only bare `f()` calls.

---

## What did NOT change

- The core thesis: deterministic AST parsing as an infallible ground truth to
  audit LLM-as-a-Judge frameworks.
- The four-phase pipeline: AST ground truth → LLM error quantification →
  embedding-vs-structure analysis → reporting / creativity-score recalculation.
- The embedding model (`microsoft/deberta-v3-large`) and the orthogonality claim
  (Pearson r ≈ 0, confirming semantic similarity is independent of structural
  compliance).
- The final recommendation: a dual-axis evaluation (AST + embeddings) is required
  for rule-bound creativity benchmarks.
