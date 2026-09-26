"""
Baseline demonstration selectors.

These are the reference points every learned or LM-guided method is compared
against:

* :class:`RandomSelector` - uniform random ``k`` demonstrations (lower bound).
* :class:`TopKSelector` - ``k`` nearest neighbours in sentence-embedding space
  (a.k.a. "SBERT" / "kNN" retrieval; Liu et al., 2022).
* :class:`BM25Selector` - classical lexical retrieval (Robertson & Zaragoza,
  2009) using an in-house Okapi BM25 so no extra dependency is needed.

All three are query-dependent and stateless apart from the fitted pool. They
return demonstrations ordered most-relevant-first, which is the convention the
prompt builder expects; callers that want the "most similar closest to the
query" ordering can reverse the list.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..utils.embeddings import DEFAULT_EMBEDDING_MODEL, Embedder, cosine_top_k
from .base import BaseSelector

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def simple_tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenizer used by BM25."""
    return _TOKEN_RE.findall(text.lower())


class RandomSelector(BaseSelector):
    """Uniformly random demonstrations, drawn without replacement per query."""

    name = "random"

    def __init__(self, k: int = 5, seed: int | None = None):
        super().__init__(k=k)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        """Re-seed the internal generator (e.g. between benchmark runs)."""
        self.seed = seed if seed is not None else self.seed
        self._rng = np.random.default_rng(self.seed)

    def select(self, query: str) -> list[int]:
        self._check_fitted()
        n = len(self.candidates)
        k = min(self.k, n)
        return self._rng.choice(n, size=k, replace=False).tolist()

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "seed": self.seed}


class TopKSelector(BaseSelector):
    """
    Nearest-neighbour retrieval in sentence-embedding space.

    Args:
        k: number of demonstrations.
        embedding_model: sentence-transformers checkpoint.
        embedder: an existing :class:`Embedder` to share across selectors
            (takes precedence over ``embedding_model``).
        device: torch device for the embedding model.
        reverse: if True, return the most similar demonstration *last*
            (closest to the query in the prompt), which several papers report
            as the better ordering for retrieval-based ICL.
    """

    name = "topk"

    def __init__(
        self,
        k: int = 5,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedder: Embedder | None = None,
        device: str | None = None,
        reverse: bool = False,
    ):
        super().__init__(k=k)
        self.embedder = embedder or Embedder(embedding_model, device=device)
        self.embedding_model = self.embedder.model_name
        self.reverse = reverse
        self.candidate_embeddings: np.ndarray | None = None

    def fit(
        self,
        candidates: Sequence[str],
        candidate_labels: Sequence[Any] | None = None,
        candidate_embeddings: np.ndarray | None = None,
    ) -> TopKSelector:
        super().fit(candidates, candidate_labels)
        if candidate_embeddings is not None:
            if len(candidate_embeddings) != len(self.candidates):
                raise ValueError("candidate_embeddings must have one row per candidate")
            self.candidate_embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
        else:
            self.candidate_embeddings = self.embedder.encode(self.candidates)
        return self

    def select(self, query: str) -> list[int]:
        self._check_fitted()
        assert self.candidate_embeddings is not None
        q = self.embedder.encode(query)
        idx = cosine_top_k(q, self.candidate_embeddings, self.k).tolist()
        return idx[::-1] if self.reverse else idx

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "embedding_model": self.embedding_model,
            "reverse": self.reverse,
        }


class BM25Selector(BaseSelector):
    """
    Okapi BM25 lexical retrieval.

    Args:
        k: number of demonstrations.
        k1: term-frequency saturation parameter (default 1.5).
        b: document-length normalisation (default 0.75).
        reverse: see :class:`TopKSelector`.
    """

    name = "bm25"

    def __init__(self, k: int = 5, k1: float = 1.5, b: float = 0.75, reverse: bool = False):
        super().__init__(k=k)
        self.k1 = k1
        self.b = b
        self.reverse = reverse
        self._doc_tfs: list[Counter] = []
        self._doc_lens: np.ndarray = np.zeros(0)
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}

    def fit(
        self,
        candidates: Sequence[str],
        candidate_labels: Sequence[Any] | None = None,
    ) -> BM25Selector:
        super().fit(candidates, candidate_labels)
        tokenized = [simple_tokenize(t) for t in self.candidates]
        self._doc_tfs = [Counter(toks) for toks in tokenized]
        self._doc_lens = np.array([len(t) for t in tokenized], dtype=np.float32)
        self._avgdl = float(self._doc_lens.mean()) if len(tokenized) else 0.0

        df: Counter = Counter()
        for tf in self._doc_tfs:
            df.update(tf.keys())
        n = len(tokenized)
        # Standard BM25 idf with the +1 inside the log to keep it non-negative.
        self._idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}
        return self

    def score(self, query: str) -> np.ndarray:
        """BM25 score of every candidate for ``query``."""
        self._check_fitted()
        q_terms = simple_tokenize(query)
        scores = np.zeros(len(self._doc_tfs), dtype=np.float32)
        if not q_terms or self._avgdl == 0:
            return scores
        for i, tf in enumerate(self._doc_tfs):
            dl = self._doc_lens[i]
            norm = self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            s = 0.0
            for term in q_terms:
                f = tf.get(term)
                if not f:
                    continue
                s += self._idf[term] * f * (self.k1 + 1) / (f + norm)
            scores[i] = s
        return scores

    def select(self, query: str) -> list[int]:
        scores = self.score(query)
        k = min(self.k, len(scores))
        order = np.lexsort((np.arange(len(scores)), -scores))[:k].tolist()
        return order[::-1] if self.reverse else order

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "k1": self.k1, "b": self.b, "reverse": self.reverse}
