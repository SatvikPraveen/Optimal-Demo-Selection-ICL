"""
Iterative Demonstration Selection (IDS).

Reference: Qin et al., "In-Context Learning with Iterative Demonstration
Selection" (2023). IDS asks the model for a zero-shot chain-of-thought
(CoT) rationale for the query, retrieves the ``k`` training examples most
similar to *that rationale* (rather than to the raw query), runs ICL with
them, and repeats ``q`` times using the newly generated rationale. The
paper takes a majority vote over the ``q`` ICL answers; the per-iteration
answers are exposed through :attr:`last_answers` so the runner can do so.

The selector needs two model callbacks:

* ``zero_shot_cot_fn(query) -> str``: a rationale for the query.
* ``icl_fn(query, demonstrations) -> str``: the model's ICL response given
  the demonstration *texts*; its rationale drives the next iteration.

They can be given at construction time or bound later with :meth:`bind`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..utils.embeddings import DEFAULT_EMBEDDING_MODEL, Embedder, cosine_top_k
from .base import BaseSelector


def default_reasoning_extractor(response: str) -> str:
    """
    Keep the reasoning part of an ICL answer: everything before the final
    "Therefore" (the paper's convention); the whole response otherwise.
    """
    marker = "Therefore"
    idx = response.find(marker)
    return response[:idx].strip() if idx > 0 else response.strip()


class IDS(BaseSelector):
    """
    Iterative Demonstration Selection.

    Args:
        k: demonstrations per prompt.
        q: number of selection/ICL iterations.
        zero_shot_cot_fn, icl_fn: model callbacks (see module docstring).
        reasoning_extractor: maps an ICL response to the text embedded for the
            next retrieval round.
        embedder / embedding_model / device: sentence-embedding model.
    """

    name = "ids"

    def __init__(
        self,
        k: int = 4,
        q: int = 3,
        zero_shot_cot_fn: Callable[[str], str] | None = None,
        icl_fn: Callable[[str, list[str]], str] | None = None,
        reasoning_extractor: Callable[[str], str] = default_reasoning_extractor,
        embedder: Embedder | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
    ):
        super().__init__(k=k)
        if q < 1:
            raise ValueError("q must be >= 1")
        self.q = q
        self.zero_shot_cot_fn = zero_shot_cot_fn
        self.icl_fn = icl_fn
        self.reasoning_extractor = reasoning_extractor
        self.embedder = embedder or Embedder(embedding_model, device=device)
        self.embedding_model = self.embedder.model_name
        self.candidate_embeddings: np.ndarray | None = None
        self.last_answers: list[str] = []
        self.last_selections: list[list[int]] = []

    # ------------------------------------------------------------------ #
    def bind(
        self,
        zero_shot_cot_fn: Callable[[str], str],
        icl_fn: Callable[[str, list[str]], str],
    ) -> IDS:
        self.zero_shot_cot_fn = zero_shot_cot_fn
        self.icl_fn = icl_fn
        return self

    def fit(
        self,
        candidates: Sequence[str],
        candidate_labels: Sequence[Any] | None = None,
        candidate_embeddings: np.ndarray | None = None,
    ) -> IDS:
        super().fit(candidates, candidate_labels)
        if candidate_embeddings is not None:
            if len(candidate_embeddings) != len(self.candidates):
                raise ValueError("candidate_embeddings must have one row per candidate")
            self.candidate_embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
        else:
            self.candidate_embeddings = self.embedder.encode(self.candidates)
        return self

    # Backwards-compatible alias used by the first version of this module.
    def precompute_train_embeddings(self, train_samples: Sequence[str]) -> np.ndarray:
        self.fit(train_samples)
        assert self.candidate_embeddings is not None
        return self.candidate_embeddings

    def select_top_k(
        self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray, k: int
    ) -> np.ndarray:
        return cosine_top_k(query_embedding, candidate_embeddings, k)

    # ------------------------------------------------------------------ #
    def select(self, query: str) -> list[int]:
        self._check_fitted()
        if self.zero_shot_cot_fn is None or self.icl_fn is None:
            raise RuntimeError("IDS needs zero_shot_cot_fn and icl_fn: pass them or call bind()")
        assert self.candidate_embeddings is not None

        self.last_answers = []
        self.last_selections = []
        reasoning = self.zero_shot_cot_fn(query)
        selected: list[int] = []
        for _ in range(self.q):
            q_emb = self.embedder.encode(reasoning)
            selected = cosine_top_k(q_emb, self.candidate_embeddings, self.k).tolist()
            demos = [self.candidates[i] for i in selected]
            answer = self.icl_fn(query, demos)
            self.last_answers.append(answer)
            self.last_selections.append(selected)
            reasoning = self.reasoning_extractor(answer) or answer
        return selected

    def select_demonstrations(
        self,
        test_sample: str,
        train_samples: Sequence[str],
        zero_shot_cot_fn: Callable[[str], str],
        icl_fn: Callable[[str, list[str]], str],
        precomputed_train_embeddings: np.ndarray | None = None,
    ) -> list[int]:
        """Backwards-compatible one-shot API (fits the pool if needed)."""
        if not self._fitted or list(train_samples) != self.candidates:
            self.fit(train_samples, candidate_embeddings=precomputed_train_embeddings)
        self.bind(zero_shot_cot_fn, icl_fn)
        return self.select(test_sample)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "q": self.q, "embedding_model": self.embedding_model}
