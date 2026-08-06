"""
Deterministic AST-based technique detector.

Uses Python's built-in ast.NodeVisitor to traverse the syntax tree of each
solution and produce a binary label vector over the configured technique set.

Techniques supported
--------------------
for_loop           : any `for` statement (ast.For)
while_loop         : any `while` statement (ast.While)
recursion          : a function that calls itself — bare call f() OR method
                     call self.f() / obj.f() where the attribute name matches
                     the enclosing function definition
list_comprehension : [x for x in ...]  (ast.ListComp)
lambda             : lambda expressions (ast.Lambda)
sorting            : calls to sorted() builtin or .sort() method
"""

import ast
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Type: { pid: [ {technique: bool, ...}, ... ] }
ASTLabels = Dict[str, List[Dict[str, bool]]]

_ALL_TECHNIQUES = frozenset([
    "for_loop", "while_loop", "recursion",
    "list_comprehension", "lambda", "sorting",
])


# ---------------------------------------------------------------------------
# Single-pass NodeVisitor
# ---------------------------------------------------------------------------

class TechniqueVisitor(ast.NodeVisitor):
    """
    Traverses an AST once and sets a detected flag for each technique found.

    Recursion detection covers both direct self-calls (f()) and method-style
    self-calls (self.f() / obj.f()) where the attribute name matches the
    enclosing function definition name.  Mutual recursion is not detected.
    """

    def __init__(self) -> None:
        self.detected: Dict[str, bool] = {t: False for t in _ALL_TECHNIQUES}
        # Stack of enclosing function names for recursion detection
        self._func_stack: List[str] = []

    def visit_For(self, node: ast.For) -> None:
        self.detected["for_loop"] = True
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.detected["while_loop"] = True
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.detected["list_comprehension"] = True
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.detected["lambda"] = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        # --- Sorting ---
        if isinstance(func, ast.Name) and func.id == "sorted":
            self.detected["sorting"] = True
        elif isinstance(func, ast.Attribute) and func.attr == "sort":
            self.detected["sorting"] = True

        # --- Recursion ---
        # Check the innermost enclosing function only (direct recursion)
        if self._func_stack:
            current = self._func_stack[-1]
            if isinstance(func, ast.Name) and func.id == current:
                # bare call: f()
                self.detected["recursion"] = True
            elif isinstance(func, ast.Attribute) and func.attr == current:
                # method-style call: self.f() or obj.f()
                self.detected["recursion"] = True

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# ASTAnalyzer
# ---------------------------------------------------------------------------

class ASTAnalyzer:
    def __init__(self, techniques: List[str]):
        unknown = [t for t in techniques if t not in _ALL_TECHNIQUES]
        if unknown:
            raise ValueError(
                f"Unknown techniques: {unknown}. Available: {sorted(_ALL_TECHNIQUES)}"
            )
        self.techniques = techniques

    def _analyze_one(self, code: str) -> Tuple[Dict[str, bool], bool]:
        """
        Parse a single Python code string.

        Returns (labels, parsed) where labels is all-False and parsed is False
        on SyntaxError (unparseable code is skipped, not crashed).
        """
        try:
            # Legacy Codeforces solutions trip SyntaxWarning (e.g. old octal literals);
            # that's not a parse failure, so silence it instead of spamming the console.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(code)
        except SyntaxError:
            return {t: False for t in self.techniques}, False
        visitor = TechniqueVisitor()
        visitor.visit(tree)
        return {t: visitor.detected[t] for t in self.techniques}, True

    def analyze_one(self, code: str) -> Dict[str, bool]:
        """Parse a single Python code string and return a dict of technique -> bool."""
        labels, _ = self._analyze_one(code)
        return labels

    def analyze_all(self, solutions: Dict[str, List[str]], progress_callback=None) -> ASTLabels:
        """
        Run analysis over the full solutions dict.

        Returns { pid: [ {technique: bool, ...}, ... ] }
        aligned with the input solutions dict.

        progress_callback(fraction) is invoked per problem so a UI can render
        a live progress bar.
        """
        result: ASTLabels = {}
        unparseable = 0
        pids = sorted(solutions.keys())
        for i, pid in enumerate(tqdm(pids, desc="AST parsing",
                                     disable=progress_callback is not None), 1):
            labels_list = []
            for code in solutions[pid]:
                labels, parsed = self._analyze_one(code)
                labels_list.append(labels)
                if not parsed:
                    unparseable += 1
            result[pid] = labels_list
            if progress_callback:
                progress_callback(i / len(pids))

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
