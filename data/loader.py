"""
Download and parse the NeoCoder dataset.

human_solutions.json   : { problem_id: [code_str, ...] }
human_solution_techniques.json : { problem_id: [[label, ...], ...] }

Both files are aligned: solutions[pid][i] corresponds to techniques[pid][i].
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)

# Internal type aliases
Solutions = Dict[str, List[str]]          # pid -> list of code strings
GPT4Labels = Dict[str, List[List[str]]]   # pid -> list of technique lists


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        logger.info(f"Using cached {dest.name}")
        return
    logger.info(f"Downloading {url} -> {dest}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def _is_python(code: str) -> bool:
    """Heuristic: try parsing with the ast module; C++ will always fail."""
    import ast
    try:
        # Legacy Codeforces solutions trip SyntaxWarning (e.g. old octal literals);
        # that's not a parse failure, so silence it instead of spamming the console.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(code)
        return True
    except SyntaxError:
        return False


def load_dataset(cfg: Config) -> Tuple[Solutions, GPT4Labels]:
    """
    Returns:
        solutions  – { pid: [code, ...] }  (Python-only if cfg.python_only)
        gpt4_labels – { pid: [[technique, ...], ...] }  aligned to solutions
    """
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(exist_ok=True)

    sol_path = data_dir / "human_solutions.json"
    tec_path = data_dir / "human_solution_techniques.json"

    _download(cfg.human_solutions_url, sol_path)
    _download(cfg.gpt4_techniques_url, tec_path)

    raw_solutions: Dict[str, List[str]] = json.loads(sol_path.read_text(encoding="utf-8"))
    raw_techniques: Dict[str, List[List[str]]] = json.loads(tec_path.read_text(encoding="utf-8"))

    solutions: Solutions = {}
    gpt4_labels: GPT4Labels = {}

    problem_ids = sorted(raw_solutions.keys())
    logger.info(f"Dataset has {len(problem_ids)} problems")

    for pid in tqdm(problem_ids, desc="Filtering solutions"):
        codes = raw_solutions.get(pid, [])
        techs = raw_techniques.get(pid, [])

        # Align: both lists must be same length
        n = min(len(codes), len(techs))
        codes, techs = codes[:n], techs[:n]

        if cfg.python_only:
            pairs = [(c, t) for c, t in zip(codes, techs) if _is_python(c)]
        else:
            pairs = list(zip(codes, techs))

        if pairs:
            solutions[pid] = [p[0] for p in pairs]
            gpt4_labels[pid] = [p[1] for p in pairs]

    total_solutions = sum(len(v) for v in solutions.values())
    logger.info(
        f"Retained {len(solutions)} problems, {total_solutions} Python solutions"
    )
    total_input = sum(min(len(raw_solutions.get(pid, [])), len(raw_techniques.get(pid, []))) for pid in problem_ids)
    dropped = total_input - total_solutions
    if dropped:
        logger.warning(f"Dropped {dropped} non-Python / misaligned solutions during filtering")
    return solutions, gpt4_labels


def flatten_dataset(
    solutions: Solutions, labels: GPT4Labels
) -> Tuple[List[str], List[List[str]], List[str]]:
    """
    Flatten nested dicts into parallel lists for easier iteration.

    Returns:
        codes     – flat list of code strings
        label_sets – flat list of technique label lists
        pids      – flat list of problem IDs (one per solution)
    """
    codes, label_sets, pids = [], [], []
    for pid in sorted(solutions.keys()):
        for code, lbls in zip(solutions[pid], labels.get(pid, [])):
            codes.append(code)
            label_sets.append(lbls)
            pids.append(pid)
    return codes, label_sets, pids
