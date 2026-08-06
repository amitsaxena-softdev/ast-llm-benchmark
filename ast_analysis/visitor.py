"""
Deterministic AST-based technique detector.

Uses Python's built-in `ast` module to walk the syntax tree of each solution
and produce a binary label vector over the configured technique set.

Techniques supported
--------------------
for_loop           : any `for` statement (ast.For)
while_loop         : any `while` statement (ast.While)
recursion          : a function that calls itself by name
list_comprehension : [x for x in ...]  (ast.ListComp)
lambda             : lambda expressions (ast.Lambda)
sorting            : calls to sorted() or .sort()
"""

import ast
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Type: { pid: [ {technique: bool, ...}, ... ] }
ASTLabels = Dict[str, List[Dict[str, bool]]]


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def _detect_for_loop(tree: ast.AST) -> bool:
    return any(isinstance(n, ast.For) for n in ast.walk(tree))


def _detect_while_loop(tree: ast.AST) -> bool:
    return any(isinstance(n, ast.While) for n in ast.walk(tree))


def _detect_list_comprehension(tree: ast.AST) -> bool:
    return any(isinstance(n, ast.ListComp) for n in ast.walk(tree))


def _detect_lambda(tree: ast.AST) -> bool:
    return any(isinstance(n, ast.Lambda) for n in ast.walk(tree))


def _detect_recursion(tree: ast.AST) -> bool:
    """A function definition that directly calls itself by name."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        func_name = node.name
        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(child, ast.Call):
                callee = child.func
                if isinstance(callee, ast.Name) and callee.id == func_name:
                    return True
    return False


def _detect_sorting(tree: ast.AST) -> bool:
    """Calls to sorted() builtin or .sort() method."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "sorted":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "sort":
            return True
    return False


_DETECTORS = {
    "for_loop":           _detect_for_loop,
    "while_loop":         _detect_while_loop,
    "recursion":          _detect_recursion,
    "list_comprehension": _detect_list_comprehension,
    "lambda":             _detect_lambda,
    "sorting":            _detect_sorting,
}


# ---------------------------------------------------------------------------
# ASTAnalyzer
# ---------------------------------------------------------------------------

class ASTAnalyzer:
    def __init__(self, techniques: List[str]):
        unknown = [t for t in techniques if t not in _DETECTORS]
        if unknown:
            raise ValueError(f"Unknown techniques: {unknown}. Available: {list(_DETECTORS)}")
        self.techniques = techniques
        self.detectors = {t: _DETECTORS[t] for t in techniques}

    def analyze_one(self, code: str) -> Dict[str, bool]:
        """
        Parse a single Python code string and return a dict of technique -> bool.
        Returns all-False on SyntaxError (unparseable code is skipped, not crashed).
        """
        try:
            # Legacy Codeforces solutions trip SyntaxWarning (e.g. old octal literals);
            # that's not a parse failure, so silence it instead of spamming the console.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(code)
        except SyntaxError:
            return {t: False for t in self.techniques}
        return {t: self.detectors[t](tree) for t in self.techniques}

    def analyze_all(self, solutions: Dict[str, List[str]]) -> ASTLabels:
        """
        Run analysis over the full solutions dict.

        Returns { pid: [ {technique: bool, ...}, ... ] }
        aligned with the input solutions dict.
        """
        result: ASTLabels = {}
        unparseable = 0
        for pid in tqdm(sorted(solutions.keys()), desc="AST parsing"):
            labels = []
            for code in solutions[pid]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    try:
                        tree = ast.parse(code)
                    except SyntaxError:
                        unparseable += 1
                        labels.append({t: False for t in self.techniques})
                        continue
                labels.append({t: self.detectors[t](tree) for t in self.techniques})
            result[pid] = labels

        total = sum(len(v) for v in result.values())
        logger.info(f"AST analysis complete: {total} solutions parsed")
        if unparseable:
            logger.warning(f"{unparseable}/{total} solutions had syntax errors and were labeled all-False")
        return result

    def save(self, labels: ASTLabels, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        logger.info(f"AST labels saved to {path}")

    @staticmethod
    def load(path: Path) -> ASTLabels:
        return json.loads(path.read_text(encoding="utf-8"))
