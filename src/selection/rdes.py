"""
RDES - Reinforcement-learning-based Demonstration Selection.

Reference: Wang et al., "RDES: Balancing Relevance and Diversity in
Demonstration Selection with Reinforcement Learning" (2024). This module is
a port of the tabular Q-learning variant used consistently across the
SST-5 / AG News notebooks in ``notebooks_archive/RDES/`` (the CommonsenseQA
notebook contains two further, mutually inconsistent and unfinished
variants that were not ported).

RL formulation:

* **State**: the sorted tuple of pool indices already chosen for the
  current prompt. The query is *not* part of the state, so the Q-table is
  shared across queries. That is a property of the original notebooks and
  is kept here; it is documented rather than "fixed" so results remain
  comparable.
* **Action**: choosing the next not-yet-selected pool index.
* **Reward**: ``0.5 * normalised label entropy of the selected demos +
  0.5 * mean cosine similarity of the selected demos to the query``, i.e.
  an explicit relevance/diversity trade-off.
* **Learning is online**: every :meth:`select` call acts epsilon-greedily
  on the current Q-table *and* updates it with the reward observed after
  each pick. :meth:`fit_policy` offers an optional offline warm-start.

Reproducibility: the selector owns a ``numpy.random.Generator``. If
``seed`` is ``None`` it is seeded from the global NumPy RNG, so calling
``src.utils.set_seed`` beforehand still makes runs deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..utils.embeddings import DEFAULT_EMBEDDING_MODEL, Embedder
from .base import BaseSelector

QTable = dict[tuple[tuple[int, ...], int], float]


class RDES(BaseSelector):
    """
    Tabular Q-learning demonstration selection.

    Args:
        k: demonstrations per query.
        num_classes: number of label classes for entropy normalisation
            (inferred from ``candidate_labels`` at ``fit`` time if omitted).
        alpha: Q-learning step size.
        gamma: discount factor.
        epsilon: exploration rate of the epsilon-greedy policy.
        relevance_weight: weight of the relevance term (diversity gets
            ``1 - relevance_weight``). Default 0.5 as in the notebooks.
        q_table: optional pre-existing Q-table to continue from.
        seed: RNG seed (``None`` derives one from the global NumPy RNG).
        embedder / embedding_model / device: sentence-embedding model.
    """

    name = "rdes"

    def __init__(
        self,
        k: int = 5,
        num_classes: int | None = None,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        relevance_weight: float = 0.5,
        q_table: QTable | None = None,
        seed: int | None = None,
        embedder: Embedder | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
    ):
        super().__init__(k=k)
        if not 0.0 <= relevance_weight <= 1.0:
            raise ValueError("relevance_weight must be in [0, 1]")
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.relevance_weight = relevance_weight
        self.q_table: QTable = q_table if q_table is not None else {}
        self.seed = seed
        self.reset_rng(seed)
        self.embedder = embedder or Embedder(embedding_model, device=device)
        self.embedding_model = self.embedder.model_name
        self.candidate_embeddings: np.ndarray | None = None
        self._label_ids: list[int] = []

    def reset_rng(self, seed: int | None = None) -> None:
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    def fit(
        self,
        candidates: Sequence[str],
        candidate_labels: Sequence[Any] | None = None,
        candidate_embeddings: np.ndarray | None = None,
    ) -> RDES:
        if candidate_labels is None:
            raise ValueError("RDES needs candidate_labels for its diversity reward")
        super().fit(candidates, candidate_labels)
        assert self.candidate_labels is not None
        classes = sorted(set(self.candidate_labels), key=str)
        class_to_id = {c: i for i, c in enumerate(classes)}
        self._label_ids = [class_to_id[c] for c in self.candidate_labels]
        if self.num_classes is None:
            self.num_classes = len(classes)
        elif self.num_classes < len(classes):
            raise ValueError("num_classes is smaller than the number of distinct labels")

        if candidate_embeddings is not None:
            if len(candidate_embeddings) != len(self.candidates):
                raise ValueError("candidate_embeddings must have one row per candidate")
            self.candidate_embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
        else:
            self.candidate_embeddings = self.embedder.encode(self.candidates)
        return self

    # ------------------------------------------------------------------ #
    def _diversity(self, selected: Sequence[int]) -> float:
        assert self.num_classes is not None
        counts = np.zeros(self.num_classes)
        for i in selected:
            counts[self._label_ids[i]] += 1
        probs = counts / counts.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        max_entropy = np.log(self.num_classes)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _relevance(self, query_embedding: np.ndarray, selected: Sequence[int]) -> float:
        assert self.candidate_embeddings is not None
        return float(np.mean(self.candidate_embeddings[list(selected)] @ query_embedding))

    def reward(self, query_embedding: np.ndarray, selected: Sequence[int]) -> float:
        w = self.relevance_weight
        return (1 - w) * self._diversity(selected) + w * self._relevance(query_embedding, selected)

    @staticmethod
    def _state(selected: Sequence[int]) -> tuple[int, ...]:
        return tuple(sorted(selected))

    def select(self, query: str) -> list[int]:
        self._check_fitted()
        n = len(self.candidates)
        query_embedding = self.embedder.encode(query)
        selected: list[int] = []

        for _ in range(min(self.k, n)):
            valid = [i for i in range(n) if i not in selected]
            state = self._state(selected)

            if self._rng.random() < self.epsilon:
                action = int(self._rng.choice(valid))
            else:
                q_values = np.array([self.q_table.get((state, a), 0.0) for a in valid])
                action = int(valid[int(np.argmax(q_values))])

            selected.append(action)
            r = self.reward(query_embedding, selected)

            next_state = self._state(selected)
            remaining = [a for a in valid if a != action]
            next_max = max((self.q_table.get((next_state, a), 0.0) for a in remaining), default=0.0)
            old = self.q_table.get((state, action), 0.0)
            self.q_table[(state, action)] = (1 - self.alpha) * old + self.alpha * (
                r + self.gamma * next_max
            )
        return selected

    def select_demonstrations(self, query: str) -> list[int]:
        """Backwards-compatible alias for :meth:`select`."""
        return self.select(query)

    def fit_policy(self, queries: Sequence[str], num_epochs: int = 1) -> RDES:
        """Optional offline warm-start: run the online learner over ``queries``."""
        for _ in range(num_epochs):
            for q in queries:
                self.select(q)
        return self

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "num_classes": self.num_classes,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "relevance_weight": self.relevance_weight,
            "seed": self.seed,
            "embedding_model": self.embedding_model,
        }
