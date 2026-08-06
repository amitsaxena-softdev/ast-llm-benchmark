"""
LLM-as-a-Judge evaluator.

Two modes
---------
1. gpt4  – reads the pre-computed GPT-4 labels already in the NeoCoder dataset
            (human_solution_techniques.json) and converts them into our binary
            technique format.  No API calls needed.

2. llama – sends each solution to Llama-3-8B-Instruct via the Groq free-tier API
            and asks it to identify which techniques are present.  This is the
            "new" LLM judge run described in Weeks 3-4 of the proposal.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)

# Type alias matching ASTLabels: { pid: [ {technique: bool, ...}, ... ] }
LLMLabels = Dict[str, List[Dict[str, bool]]]


# ---------------------------------------------------------------------------
# GPT-4 label converter (no API call — reads existing dataset labels)
# ---------------------------------------------------------------------------

def convert_gpt4_labels(
    gpt4_raw: Dict[str, List[List[str]]],
    techniques: List[str],
    label_map: Dict[str, str],
) -> LLMLabels:
    """
    Convert raw GPT-4 technique strings from the dataset into our binary dict format.

    gpt4_raw  : { pid: [ ["for loop", "tuple", ...], ... ] }
    techniques: ["for_loop", "while_loop", ...]
    label_map : { "for_loop": "for loop", ... }
    """
    result: LLMLabels = {}
    for pid, sol_technique_lists in gpt4_raw.items():
        result[pid] = []
        for raw_labels in sol_technique_lists:
            raw_lower = {l.strip().lower() for l in raw_labels}
            binary = {
                t: (label_map[t].lower() in raw_lower)
                for t in techniques
            }
            result[pid].append(binary)
    logger.debug(f"Converted GPT-4 labels for {len(result)} problems")
    return result


# ---------------------------------------------------------------------------
# Llama-3-8B judge via Groq API
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise code analysis assistant. "
    "Respond ONLY with valid JSON. No explanation, no markdown fences."
)

_USER_TEMPLATE = """\
Analyze the following Python code and determine which of these techniques are used.
Respond with a JSON object where each key is a technique name and the value is true or false.

Techniques:
- for_loop: uses a for loop (for ... in ...)
- while_loop: uses a while loop
- recursion: a function calls itself
- list_comprehension: uses list comprehension [expr for x in ...]
- lambda: uses a lambda expression
- sorting: calls sorted() or .sort()

Code:
```python
{code}
```

JSON response:"""


def _retry_after_seconds(error_msg: str, default: float = 5.0) -> float:
    """Extract the 'try again in Xs' hint from a Groq 429 message."""
    m = re.search(r"try again in ([\d.]+)\s*s", error_msg)
    if m:
        # add a small buffer over the suggested wait
        return float(m.group(1)) + 1.0
    return default


def _parse_llm_response(text: str, techniques: List[str]) -> Dict[str, bool]:
    """Extract the JSON object from the LLM response, falling back to all-False."""
    # Strip markdown fences if the model disobeys instructions
    text = re.sub(r"```[a-z]*", "", text).strip()
    try:
        parsed = json.loads(text)
        return {t: bool(parsed.get(t, False)) for t in techniques}
    except (json.JSONDecodeError, AttributeError):
        logger.warning(f"Failed to parse LLM response: {text[:120]!r}")
        return {t: False for t in techniques}


class LLMJudge:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client: Optional[object] = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError:
                raise ImportError("Install groq: pip install groq")
            if not self.cfg.llm_api_key:
                raise ValueError(
                    "Set GROQ_API_KEY environment variable to use the Llama judge. "
                    "Get a free key at console.groq.com"
                )
            self._client = Groq(api_key=self.cfg.llm_api_key)
        return self._client

    def judge_one(self, code: str) -> Optional[Dict[str, bool]]:
        """
        Send one solution to the LLM and return binary technique labels.

        Retries on rate-limit (429) errors, honouring the server's retry-after
        hint.  Returns None if the call ultimately fails — the caller must skip
        that solution rather than record fake all-False labels, which would
        corrupt the error-rate analysis.
        """
        client = self._get_client()
        prompt = _USER_TEMPLATE.format(code=code[:3000])  # guard against very long solutions

        for attempt in range(self.cfg.llm_max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.cfg.llm_model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=256,
                )
                text = response.choices[0].message.content or ""
                return _parse_llm_response(text, self.cfg.techniques)
            except Exception as exc:
                msg = str(exc)
                is_rate_limit = "429" in msg or "rate_limit" in msg
                wait = _retry_after_seconds(msg) if is_rate_limit else 1.0
                if attempt < self.cfg.llm_max_retries - 1:
                    if is_rate_limit:
                        logger.info(f"Rate limited; waiting {wait:.1f}s (attempt {attempt+1})")
                    time.sleep(wait)
                    continue
                logger.warning(f"LLM API error (giving up after {attempt+1} attempts): {exc}")
                return None

    def _stratified_targets(self, solutions, ast_labels, budget):
        """
        Pick up to `budget` (pid, idx) solution targets so that every technique
        is represented by positive examples — including rare ones.

        Without this, a random sample is dominated by the most common technique
        (for_loop) and never tests the judge on lambda / recursion / list comp,
        making the judge look deceptively accurate.  Returns a sorted list of
        (pid, idx) pairs.
        """
        # Bucket every solution by the techniques AST says it uses.
        by_tech = {t: [] for t in self.cfg.techniques}
        for pid in sorted(ast_labels.keys()):
            if pid not in solutions:
                continue
            for idx, lbl in enumerate(ast_labels[pid]):
                if idx >= len(solutions[pid]):
                    continue
                for t in self.cfg.techniques:
                    if lbl.get(t, False):
                        by_tech[t].append((pid, idx))

        # Round-robin across techniques, rarest first, so scarce techniques are
        # guaranteed representation before the budget is exhausted.
        order = sorted(self.cfg.techniques, key=lambda t: len(by_tech[t]))
        chosen, seen = [], set()
        cursor = {t: 0 for t in order}
        while len(chosen) < budget:
            progressed = False
            for t in order:
                if len(chosen) >= budget:
                    break
                bucket = by_tech[t]
                while cursor[t] < len(bucket):
                    cand = bucket[cursor[t]]
                    cursor[t] += 1
                    if cand not in seen:
                        seen.add(cand)
                        chosen.append(cand)
                        progressed = True
                        break
            if not progressed:
                break
        return sorted(chosen)

    def evaluate_all(
        self,
        solutions: Dict[str, List[str]],
        ast_labels: Optional[Dict] = None,
        progress_callback=None,
    ) -> LLMLabels:
        """
        Evaluate up to cfg.llm_max_solutions solutions with the LLM judge.

        If ast_labels is provided, solutions are chosen by *stratified* sampling
        so every technique (including rare lambda / recursion) is represented —
        this is what makes the judge's failure modes measurable.  Otherwise it
        falls back to one solution per problem.

        Results are stored positionally: result[pid][idx] aligns with
        ast_labels[pid][idx]; unevaluated slots are None and skipped by the
        metrics.  Failed API calls (after retries) are left as None rather than
        recorded as fake all-False, which would corrupt the error-rate analysis.

        progress_callback(done, total) is invoked after each solution.
        """
        budget = self.cfg.llm_max_solutions
        if ast_labels:
            targets = self._stratified_targets(solutions, ast_labels, budget)
        else:
            targets = [(p, 0) for p in sorted(solutions) if solutions[p]][:budget]

        result: LLMLabels = {}
        total = len(targets)
        skipped = 0

        with tqdm(total=total, desc="LLM judging",
                  disable=progress_callback is not None) as pbar:
            for i, (pid, idx) in enumerate(targets, 1):
                labels = self.judge_one(solutions[pid][idx])
                # Ensure result[pid] is long enough, padding with None.
                slot = result.setdefault(pid, [])
                while len(slot) <= idx:
                    slot.append(None)
                if labels is not None:
                    slot[idx] = labels
                else:
                    skipped += 1
                pbar.update(1)
                if progress_callback:
                    progress_callback(i, total)
                time.sleep(self.cfg.llm_request_interval)

        if skipped:
            logger.warning(f"Skipped {skipped} solutions due to API failures")

        evaluated = sum(1 for v in result.values() for x in v if x is not None)
        logger.info(f"LLM judge evaluated {evaluated} solutions across {len(result)} problems")
        return result

    def save(self, labels: LLMLabels, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        logger.info(f"LLM labels saved to {path}")

    @staticmethod
    def load(path: Path) -> LLMLabels:
        return json.loads(path.read_text(encoding="utf-8"))
