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
    from transformers import AutoModel, DebertaV2Tokenizer
    import torch

    logger.info(f"Loading embedding model: {model_name}")
    # AutoTokenizer ignores use_fast=False for DeBERTa-v2 in some transformers versions,
    # causing a vocab_file=None crash. Import the slow tokenizer class directly.
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    # use_safetensors=True avoids the torch.load CVE-2025-32434 block (requires torch>=2.6)
    model = AutoModel.from_pretrained(model_name, use_safetensors=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    Jaccard similarity over the union of active techniques.
    Returns 1.0 if both have zero active techniques (both empty → identical).
    """
    keys = set(labels_a) | set(labels_b)
    if not keys:
        return 1.0
    matches = sum(labels_a.get(k, False) == labels_b.get(k, False) for k in keys)
    return matches / len(keys)


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
        Encode all solutions in batches.

        Returns:
            embeddings – list of 1-D numpy vectors
            pids       – parallel list of problem IDs
            idxs       – parallel list of within-problem indices
        """
        self._ensure_model()

        flat_codes, flat_pids, flat_idxs = [], [], []
        for pid in sorted(solutions.keys()):
            for i, code in enumerate(solutions[pid]):
                flat_codes.append(code[:2000])
                flat_pids.append(pid)
                flat_idxs.append(i)

        cap = self.cfg.embedding_max_solutions
        if cap and len(flat_codes) > cap:
            total = len(flat_codes)
            rng = np.random.default_rng(self.cfg.random_seed)
            chosen = rng.choice(total, size=cap, replace=False)
            chosen.sort()
            flat_codes = [flat_codes[k] for k in chosen]
            flat_pids  = [flat_pids[k]  for k in chosen]
            flat_idxs  = [flat_idxs[k]  for k in chosen]
            logger.info(f"Embedding subsample: {cap} / {total} solutions")

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
        1. Encode all solutions.
        2. Sample random pairs and compute (cosine_sim, structural_sim).
        3. Find the top divergent pairs (high cosine sim, diff structure).
        4. Compute Pearson correlation between the two similarity axes.
        """
        embeddings, pids, idxs = self.encode_all(solutions, progress_callback=progress_callback)
        n = len(embeddings)
        logger.info(f"Encoded {n} solutions")

        rng = np.random.default_rng(self.cfg.random_seed)
        sample_size = min(max_pairs, n * (n - 1) // 2)
        pair_indices = rng.choice(n, size=(sample_size, 2), replace=True)
        # Ensure i < j
        pair_indices = np.sort(pair_indices, axis=1)
        pair_indices = pair_indices[pair_indices[:, 0] != pair_indices[:, 1]]

        pairwise_data: List[Tuple[float, float]] = []
        candidate_divergent: List[Tuple[float, DivergentPair]] = []

        for i, j in tqdm(pair_indices[:max_pairs], desc="Computing pair similarities"):
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
        path.write_text(json.dumps(results.to_dict(), indent=2))
        logger.info(f"Embedding results saved to {path}")

    @staticmethod
    def load(path: Path) -> dict:
        return json.loads(path.read_text())
