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
    logger.info(f"Converted GPT-4 labels for {len(result)} problems")
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

    def judge_one(self, code: str) -> Dict[str, bool]:
        """Send one solution to the LLM and return binary technique labels."""
        client = self._get_client()
        prompt = _USER_TEMPLATE.format(code=code[:3000])  # guard against very long solutions
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
            logger.warning(f"LLM API error: {exc}")
            return {t: False for t in self.cfg.techniques}

    def evaluate_all(
        self,
        solutions: Dict[str, List[str]],
    ) -> LLMLabels:
        """
        Evaluate up to cfg.llm_max_solutions solutions with the LLM judge.

        Solutions are sampled uniformly across problems to respect the free-tier
        rate limit while covering as many problems as possible.
        """
        result: LLMLabels = {}
        evaluated = 0
        max_sol = self.cfg.llm_max_solutions

        for pid in tqdm(sorted(solutions.keys()), desc="LLM judging"):
            if evaluated >= max_sol:
                break
            codes = solutions[pid]
            result[pid] = []
            for code in codes:
                if evaluated >= max_sol:
                    break
                labels = self.judge_one(code)
                result[pid].append(labels)
                evaluated += 1
                # Small sleep to stay within Groq free-tier rate limits
                time.sleep(0.05)

        logger.info(f"LLM judge evaluated {evaluated} solutions across {len(result)} problems")
        return result

    def save(self, labels: LLMLabels, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(labels, indent=2))
        logger.info(f"LLM labels saved to {path}")

    @staticmethod
    def load(path: Path) -> LLMLabels:
        return json.loads(path.read_text())
