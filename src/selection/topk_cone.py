"""
TopK + CoNE: embedding retrieval refined by conditional entropy.

Reference: Peng et al., "Revisiting Demonstration Selection Strategies in
In-Context Learning" (ACL 2024), which proposes TopK + ConE: re-rank
similarity-retrieved candidates by the conditional entropy of the query
given each candidate. The pipeline:

1. retrieve ``retrieve_k`` candidates by sentence-embedding similarity;
2. score each candidate ``d`` by ``H(query | d) = -log p_LM(query | d)``,
   the conditional entropy of the query given that demonstration under a
   small causal LM;
3. keep the ``k`` candidates with the lowest conditional entropy.

The archived notebook computed ``H(query | d)`` as ``CE(d + query) -
CE(d)`` with two separate forward passes and the batch-mean-loss caveat
described in :mod:`src.models.scoring`; :class:`LMScorer.conditional_nll`
yields the same quantity exactly, in one masked pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..models.scoring import LMScorer
from ..utils.embeddings import DEFAULT_EMBEDDING_MODEL, Embedder, cosine_top_k
from .base import BaseSelector


class TopKCoNE(BaseSelector):
    """
    Top-K retrieval + CoNE re-ranking.

    Args:
        k: final number of demonstrations.
        retrieve_k: candidates retrieved before re-ranking.
        scorer: :class:`LMScorer` for the CoNE stage. If omitted, one is
            created lazily from ``scorer_model`` on first use.
        scorer_model: HF checkpoint for the lazily created scorer.
        embedder / embedding_model / device: sentence-embedding model.
        separator: placed between the demonstration and the query.
        query_prefix: prepended to the query before scoring (template).

    Backwards compatibility: ``TopKCoNE(embeddings=..., raw_texts=...)``
    still works and calls :meth:`fit` immediately.
    """

    name = "topk_cone"

    def __init__(
        self,
        k: int = 5,
        retrieve_k: int = 30,
        scorer: LMScorer | None = None,
        scorer_model: str = "gpt2",
        embedder: Embedder | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        separator: str = "\n",
        query_prefix: str = "",
        embeddings: np.ndarray | None = None,
        raw_texts: Sequence[str] | None = None,
        model_name: str | None = None,
    ):
        super().__init__(k=k)
        if retrieve_k < 1:
            raise ValueError("retrieve_k must be >= 1")
        self.retrieve_k = retrieve_k
        self.scorer = scorer
        self.scorer_model = model_name or scorer_model
        self.device = device
        self.embedder = embedder or Embedder(embedding_model, device=device)
        self.embedding_model = self.embedder.model_name
        self.separator = separator
        self.query_prefix = query_prefix
        self.candidate_embeddings: np.ndarray | None = None
        self.last_scores: dict[int, float] = {}

        if raw_texts is not None:
            self.fit(raw_texts, candidate_embeddings=embeddings)

    # ------------------------------------------------------------------ #
    def _get_scorer(self) -> LMScorer:
        if self.scorer is None:
            self.scorer = LMScorer.from_pretrained(self.scorer_model, device=self.device)
        return self.scorer

    def fit(
        self,
        candidates: Sequence[str],
        candidate_labels: Sequence[Any] | None = None,
        candidate_embeddings: np.ndarray | None = None,
        scorer: LMScorer | None = None,
    ) -> TopKCoNE:
        super().fit(candidates, candidate_labels)
        if scorer is not None:
            self.scorer = scorer
        if candidate_embeddings is not None:
            if len(candidate_embeddings) != len(self.candidates):
                raise ValueError("candidate_embeddings must have one row per candidate")
            self.candidate_embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
        else:
            self.candidate_embeddings = self.embedder.encode(self.candidates)
        return self

    # Legacy attribute names.
    @property
    def embeddings(self) -> np.ndarray | None:
        return self.candidate_embeddings

    @property
    def raw_texts(self) -> list[str]:
        return self.candidates

    # ------------------------------------------------------------------ #
    def get_topk(self, query_embedding: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.candidate_embeddings is not None
        return cosine_top_k(query_embedding, self.candidate_embeddings, self.retrieve_k)

    def apply_cone(self, query_text: str, topk_indices: Sequence[int]) -> list[int]:
        """Re-rank ``topk_indices`` by conditional entropy of the query."""
        idx = [int(i) for i in topk_indices]
        if not idx:
            return []
        scorer = self._get_scorer()
        contexts = [self.candidates[i] + self.separator for i in idx]
        target = self.query_prefix + query_text
        nll = scorer.conditional_nll(contexts, [target] * len(contexts))
        self.last_scores = {i: float(n) for i, n in zip(idx, nll)}
        order = sorted(range(len(idx)), key=lambda j: (nll[j], idx[j]))
        return [idx[j] for j in order[: self.k]]

    def select(self, query: str) -> list[int]:
        q_emb = self.embedder.encode(query)
        return self.apply_cone(query, self.get_topk(q_emb))

    def select_demonstrations(self, query_embedding: np.ndarray, query_text: str) -> list[int]:
        """Backwards-compatible API taking a precomputed query embedding."""
        return self.apply_cone(query_text, self.get_topk(query_embedding))

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "retrieve_k": self.retrieve_k,
            "embedding_model": self.embedding_model,
            "scorer_model": self.scorer_model,
            "separator": self.separator,
            "query_prefix": self.query_prefix,
        }
