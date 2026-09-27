"""
Se² - Sequential Example Selection via beam search.

Reference: Liu et al., "Se²: Sequential Example Selection for In-Context
Learning" (ACL Findings 2024). The original method trains a retriever with
sequence-aware LM feedback; this repository implements the *training-free*
variant used in the archived notebooks
(``notebooks_archive/SE2/``): retrieve a candidate set by embedding
similarity, then grow the demonstration sequence one example at a time with
beam search, scoring every partial sequence with a causal LM.

Differences from the notebook code (all deliberate; see the module-level
tests and docs/methods.md):

* **The score is genuinely sequential.** A candidate is scored by how well
  the *whole prefix so far plus the candidate* explains the query, i.e.
  ``-NLL(query | d_1 ... d_t)``, not by an independent per-candidate term.
  With the notebook's scoring, beam search degenerated into a top-k sort.
* **No test-label leakage.** The gold label of the query is never used at
  selection time. The query text is the scoring target.
* **Real per-example likelihoods.** Scores come from :class:`LMScorer`,
  which masks padding and context tokens instead of rescaling a batch-mean
  loss by sequence length.
* **The query is not in its own pool.** ``fit`` binds a separate training
  pool; the benchmark runner never puts test items in it.

Cost: at most ``retrieve_k * beam_size * k`` LM forward passes per query
(fewer at the first step and when candidates are exhausted).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..models.scoring import LMScorer
from ..utils.embeddings import DEFAULT_EMBEDDING_MODEL, Embedder, cosine_top_k
from .base import BaseSelector


class Se2(BaseSelector):
    """
    Sequential example selection with LM-scored beam search.

    Args:
        k: number of demonstrations (``shot`` in the paper).
        retrieve_k: size of the embedding-retrieved candidate set.
        beam_size: number of partial sequences kept per step. ``1`` gives
            greedy sequential selection.
        scorer: an :class:`LMScorer` used for sequence scoring. Required at
            ``select`` time; can also be passed to :meth:`fit`.
        embedder / embedding_model / device: sentence-embedding model for the
            retrieval stage (see :class:`TopKSelector`).
        separator: string placed between demonstrations and before the query.
        query_prefix: optional string prepended to the query when scoring
            (e.g. ``"Review: "``) so that the scored continuation matches the
            template used at inference time.
    """

    name = "se2"

    def __init__(
        self,
        k: int = 3,
        retrieve_k: int = 20,
        beam_size: int = 3,
        scorer: LMScorer | None = None,
        embedder: Embedder | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        separator: str = "\n\n",
        query_prefix: str = "",
    ):
        super().__init__(k=k)
        if retrieve_k < 1 or beam_size < 1:
            raise ValueError("retrieve_k and beam_size must be >= 1")
        self.retrieve_k = retrieve_k
        self.beam_size = beam_size
        self.scorer = scorer
        self.embedder = embedder or Embedder(embedding_model, device=device)
        self.embedding_model = self.embedder.model_name
        self.separator = separator
        self.query_prefix = query_prefix
        self.candidate_embeddings: np.ndarray | None = None
        self.last_beam_scores: list[float] = []

    def fit(
        self,
        candidates: Sequence[str],
        candidate_labels: Sequence[Any] | None = None,
        candidate_embeddings: np.ndarray | None = None,
        scorer: LMScorer | None = None,
    ) -> Se2:
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

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str) -> list[int]:
        """Embedding-similarity candidate set (most similar first)."""
        self._check_fitted()
        assert self.candidate_embeddings is not None
        q = self.embedder.encode(query)
        return cosine_top_k(q, self.candidate_embeddings, self.retrieve_k).tolist()

    def _context(self, seq: Sequence[int]) -> str:
        demos = [self.candidates[i] for i in seq]
        return self.separator.join(demos) + self.separator

    def select(self, query: str) -> list[int]:
        self._check_fitted()
        if self.scorer is None:
            raise RuntimeError("Se2 needs an LMScorer: pass scorer= to __init__ or fit()")
        candidates = self.retrieve(query)
        if self.k == 0 or not candidates:
            return []
        target = self.query_prefix + query

        beams: list[tuple[list[int], float]] = [([], 0.0)]
        for _ in range(min(self.k, len(candidates))):
            expansions: list[list[int]] = []
            for seq, _score in beams:
                used = set(seq)
                for c in candidates:
                    if c not in used:
                        expansions.append(seq + [c])
            if not expansions:
                break
            contexts = [self._context(seq) for seq in expansions]
            nll = self.scorer.conditional_nll(contexts, [target] * len(contexts))
            scored = [(seq, float(-n)) for seq, n in zip(expansions, nll)]
            # Higher log-likelihood first; deterministic tie-break on sequence.
            scored.sort(key=lambda x: (-x[1], x[0]))
            beams = scored[: self.beam_size]

        self.last_beam_scores = [s for _, s in beams]
        return list(beams[0][0])

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "retrieve_k": self.retrieve_k,
            "beam_size": self.beam_size,
            "embedding_model": self.embedding_model,
            "scorer_model": getattr(getattr(self.scorer, "model", None), "name_or_path", None),
            "separator": self.separator,
            "query_prefix": self.query_prefix,
        }
