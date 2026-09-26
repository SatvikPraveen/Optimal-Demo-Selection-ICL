"""
Shared sentence-embedding helper.

Every embedding-based selector in this repository (Top-K/SBERT, IDS, RDES,
Se², TopK+CoNE, ...) needs the same two things: a sentence-transformers
model and L2-normalised embeddings for a fixed candidate pool that are
computed once and reused across queries. Centralising that here means the
pool is embedded exactly once per experiment even when several selectors
share it, and the test-suite has a single network boundary to mock.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalise a 1-D or 2-D array so that dot products are cosines."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


class Embedder:
    """
    Thin wrapper around ``SentenceTransformer`` with an in-memory cache.

    Args:
        model_name: sentence-transformers checkpoint to load.
        device: torch device string, or ``None`` for automatic selection.
        normalize: if True (default) every returned embedding is unit-norm,
            so cosine similarity reduces to a dot product.
        batch_size: encoding batch size.
    """

    _shared_models: dict[str, SentenceTransformer] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        normalize: bool = True,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        key = f"{model_name}@{device}"
        if key not in self._shared_models:
            self._shared_models[key] = SentenceTransformer(model_name, device=device)
        self.model = self._shared_models[key]
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def clear_shared_models(cls) -> None:
        """Drop cached model instances (mainly for tests)."""
        cls._shared_models.clear()

    def encode(self, texts: str | Sequence[str]) -> np.ndarray:
        """
        Embed one string (returns shape ``(d,)``) or a sequence of strings
        (returns shape ``(n, d)``). Results are memoised per text.
        """
        single = isinstance(texts, str)
        items: list[str] = [texts] if single else list(texts)

        missing = [t for t in items if t not in self._cache]
        if missing:
            # De-duplicate while preserving order so we never encode a
            # string twice within one call either.
            unique_missing = list(dict.fromkeys(missing))
            vecs = self.model.encode(
                unique_missing,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vecs = np.asarray(vecs, dtype=np.float32)
            if self.normalize:
                vecs = l2_normalize(vecs)
            for t, v in zip(unique_missing, vecs):
                self._cache[t] = v

        out = np.stack([self._cache[t] for t in items])
        return out[0] if single else out

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


def cosine_top_k(query_embedding: np.ndarray, pool_embeddings: np.ndarray, k: int) -> np.ndarray:
    """
    Indices of the ``k`` rows of ``pool_embeddings`` most similar to
    ``query_embedding`` (cosine), most similar first. Ties are broken by
    lower index so results are deterministic.
    """
    q = l2_normalize(np.asarray(query_embedding).reshape(1, -1))[0]
    p = l2_normalize(np.asarray(pool_embeddings))
    sims = p @ q
    k = min(k, len(sims))
    # argsort on (-sim, index) gives a stable, deterministic ordering
    order = np.lexsort((np.arange(len(sims)), -sims))
    return order[:k]
