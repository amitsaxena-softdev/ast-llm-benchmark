"""
Embedding-vs-structure analysis (Weeks 5-6).

Encodes solutions with DeBERTa-v3-large (mean-pool last hidden state),
then finds pairs that are semantically close (high cosine similarity) but
structurally divergent (different AST technique vectors).

These "Semantic Collision with Structural Divergence" pairs are the
empirical proof that embeddings cannot enforce structural constraints.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ast_analysis.visitor import ASTLabels
from config import Config

logger = logging.getLogger(__name__)


@dataclass
class DivergentPair:
    pid_a: str
    pid_b: str
    idx_a: int
    idx_b: int
    cosine_similarity: float
    techniques_a: Dict[str, bool]
    techniques_b: Dict[str, bool]
    differing_techniques: List[str]

    def to_dict(self) -> dict:
        return {
            "pid_a": self.pid_a,
            "pid_b": self.pid_b,
            "idx_a": self.idx_a,
            "idx_b": self.idx_b,
            "cosine_similarity": round(self.cosine_similarity, 4),
            "techniques_a": self.techniques_a,
            "techniques_b": self.techniques_b,
            "differing_techniques": self.differing_techniques,
        }


@dataclass
class EmbeddingResults:
    # Pearson correlation between cosine sim and structural similarity
    correlation: float
    # All pairwise (cosine_sim, structural_sim) data points (sampled)
    pairwise_data: List[Tuple[float, float]]
    # Top divergent pairs (high cosine sim, different AST)
    divergent_pairs: List[DivergentPair]

    def to_dict(self) -> dict:
        return {
            "correlation": self.correlation,
            "pairwise_data": [[round(c, 4), round(s, 4)] for c, s in self.pairwise_data],
            "divergent_pairs": [p.to_dict() for p in self.divergent_pairs],
        }


def _load_model(model_name: str):
    """Load tokenizer + model from HuggingFace. Returns (tokenizer, model)."""
    import transformers
    import huggingface_hub
    from transformers import AutoModel, DebertaV2Tokenizer
    import torch

    # transformers/huggingface_hub manage their own logger verbosity, separate
    # from the root logger — this silences the "unauthenticated requests"
    # notice and the per-load "UNEXPECTED weights" report table, both benign
    # and expected every run (we load AutoModel from a checkpoint that also
    # ships pretraining-head weights we don't use).
    transformers.logging.set_verbosity_error()
    huggingface_hub.utils.logging.set_verbosity_error()

    logger.info(f"Loading embedding model: {model_name}")
    # AutoTokenizer ignores use_fast=False for DeBERTa-v2 in some transformers versions,
    # causing a vocab_file=None crash. Import the slow tokenizer class directly.
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    # use_safetensors=True avoids the torch.load CVE-2025-32434 block (requires torch>=2.6)
    model = AutoModel.from_pretrained(model_name, use_safetensors=True)
    model.eval()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device)

    if device == "mps":
        # Not every op has an MPS kernel in every PyTorch version — probe with
        # a trivial forward pass now rather than fail partway through a long
        # encoding run.
        try:
            _encode_batch(["print(1)"], tokenizer, model)
        except Exception as exc:
            logger.warning(f"MPS probe failed ({exc}); falling back to CPU")
            device = "cpu"
            model = model.to(device)

    logger.info(f"Model loaded on {device}")
    return tokenizer, model


def _mean_pool(token_embeddings, attention_mask) -> "np.ndarray":
    """Mean pool last hidden state, masked for padding."""
    import torch
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)).cpu().numpy()


def _encode_batch(codes: List[str], tokenizer, model, max_length: int = 512) -> np.ndarray:
    import torch
    inputs = tokenizer(
        codes,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _structural_sim(labels_a: Dict[str, bool], labels_b: Dict[str, bool]) -> float:
    """
    Jaccard similarity over the set of techniques active in either solution.

    intersection / union where both sets contain the techniques each solution uses.
    Returns 1.0 when both solutions use zero techniques (empty ∩ empty = identical).
    This avoids the sparse-vector inflation problem of the simple matching coefficient,
    which would count every shared absence as a match and inflate similarity.
    """
    keys = set(labels_a) | set(labels_b)
    intersection = sum(labels_a.get(k, False) and labels_b.get(k, False) for k in keys)
    union = sum(labels_a.get(k, False) or labels_b.get(k, False) for k in keys)
    return intersection / union if union > 0 else 1.0


def _techniques_differ(a: Dict[str, bool], b: Dict[str, bool]) -> List[str]:
    return [k for k in a if a[k] != b.get(k, False)]


class EmbeddingAnalyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._tokenizer = None
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._tokenizer, self._model = _load_model(self.cfg.embedding_model)

    def encode_all(
        self, solutions: Dict[str, List[str]], progress_callback=None
    ) -> Tuple[List[np.ndarray], List[str], List[int]]:
        """
        Encode solutions in batches.

        Solutions are subsampled by *whole problem*, not individually — this
        keeps every sampled problem's full solution set together, so
        same-problem pairs (the only pairs guaranteed functionally equivalent,
        since they all pass the same test suite) remain available to
        analyze().

        Returns:
            embeddings – list of 1-D numpy vectors
            pids       – parallel list of problem IDs
            idxs       – parallel list of within-problem indices
        """
        self._ensure_model()

        cap = self.cfg.embedding_max_solutions
        pids_sorted = sorted(solutions.keys())
        total_available = sum(len(solutions[p]) for p in pids_sorted)

        if cap and total_available > cap:
            rng = np.random.default_rng(self.cfg.random_seed)
            order = list(rng.permutation(pids_sorted))
            selected_pids, total = [], 0
            for pid in order:
                n = len(solutions[pid])
                if total and total + n > cap:
                    continue
                selected_pids.append(pid)
                total += n
                if total >= cap:
                    break
            selected_pids = sorted(selected_pids)
            logger.info(
                f"Embedding subsample: {total}/{total_available} solutions "
                f"across {len(selected_pids)} problems (whole-problem "
                f"sampling, so same-problem pairs remain available)"
            )
        else:
            selected_pids = pids_sorted

        flat_codes, flat_pids, flat_idxs = [], [], []
        for pid in selected_pids:
            for i, code in enumerate(solutions[pid]):
                flat_codes.append(code[:2000])
                flat_pids.append(pid)
                flat_idxs.append(i)

        embeddings = []
        bs = self.cfg.embedding_batch_size
        total_batches = range(0, len(flat_codes), bs)
        for start in tqdm(total_batches, desc="Encoding", disable=progress_callback is not None):
            batch = flat_codes[start : start + bs]
            vecs = _encode_batch(batch, self._tokenizer, self._model)
            embeddings.extend(vecs)
            if progress_callback:
                progress_callback(min(1.0, (start + bs) / max(len(flat_codes), 1)))

        return embeddings, flat_pids, flat_idxs

    def analyze(
        self,
        solutions: Dict[str, List[str]],
        ast_labels: ASTLabels,
        max_pairs: int = 5000,
        progress_callback=None,
    ) -> EmbeddingResults:
        """
        Main analysis:
        1. Encode solutions (whole-problem subsample).
        2. Form pairs *within the same problem only* — these are guaranteed
           functionally equivalent, since every human solution to a given
           Codeforces problem passed that problem's test suite. Cross-problem
           pairs are deliberately excluded: two solutions to different
           problems have no reason to be functionally equivalent, so their
           cosine similarity wouldn't test the proposal's claim (see project
           proposal section 1.2 — the comparison is about two solutions to
           *the same* problem).
        3. Find the top divergent pairs (high cosine sim, diff structure).
        4. Compute Pearson correlation between the two similarity axes.
        """
        embeddings, pids, idxs = self.encode_all(solutions, progress_callback=progress_callback)
        n = len(embeddings)
        logger.info(f"Encoded {n} solutions")

        by_pid: Dict[str, List[int]] = {}
        for i, pid in enumerate(pids):
            by_pid.setdefault(pid, []).append(i)

        candidate_pairs: List[Tuple[int, int]] = []
        for members in by_pid.values():
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    candidate_pairs.append((members[a], members[b]))

        logger.info(f"{len(candidate_pairs)} same-problem solution pairs available")

        rng = np.random.default_rng(self.cfg.random_seed)
        if len(candidate_pairs) > max_pairs:
            chosen = rng.choice(len(candidate_pairs), size=max_pairs, replace=False)
            candidate_pairs = [candidate_pairs[k] for k in chosen]

        pairwise_data: List[Tuple[float, float]] = []
        candidate_divergent: List[Tuple[float, DivergentPair]] = []

        for i, j in tqdm(candidate_pairs, desc="Computing pair similarities"):
            csim = _cosine_sim(embeddings[i], embeddings[j])

            labels_i = ast_labels.get(pids[i], [{}])[idxs[i]] if idxs[i] < len(ast_labels.get(pids[i], [])) else {}
            labels_j = ast_labels.get(pids[j], [{}])[idxs[j]] if idxs[j] < len(ast_labels.get(pids[j], [])) else {}

            ssim = _structural_sim(labels_i, labels_j)
            pairwise_data.append((csim, ssim))

            # Divergent: semantically similar but structurally different
            if csim >= self.cfg.similarity_threshold and ssim < 1.0:
                diffs = _techniques_differ(labels_i, labels_j)
                if diffs:
                    pair = DivergentPair(
                        pid_a=pids[i], pid_b=pids[j],
                        idx_a=idxs[i], idx_b=idxs[j],
                        cosine_similarity=csim,
                        techniques_a=labels_i, techniques_b=labels_j,
                        differing_techniques=diffs,
                    )
                    candidate_divergent.append((csim, pair))

        # Sort by cosine similarity descending and take top-N
        candidate_divergent.sort(key=lambda x: x[0], reverse=True)
        top_pairs = [p for _, p in candidate_divergent[: self.cfg.top_divergent_pairs]]

        # Pearson correlation
        if pairwise_data:
            cos_vals = [c for c, _ in pairwise_data]
            str_vals = [s for _, s in pairwise_data]
            correlation = float(np.corrcoef(cos_vals, str_vals)[0, 1])
        else:
            correlation = 0.0

        logger.info(
            f"Embedding analysis done. Pearson r={correlation:.3f}, "
            f"{len(top_pairs)} divergent pairs found"
        )
        return EmbeddingResults(
            correlation=correlation,
            pairwise_data=pairwise_data,
            divergent_pairs=top_pairs,
        )

    def save(self, results: EmbeddingResults, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"Embedding results saved to {path}")

    @staticmethod
    def load(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))
