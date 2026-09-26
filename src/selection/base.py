"""
Common interface for demonstration-selection methods.

Every selector follows the same two-phase protocol so that the benchmark
runner (``experiments/run_benchmark.py``) can treat them interchangeably:

1. ``fit(candidates, candidate_labels)`` binds the demonstration pool. Any
   pool-level pre-computation (embeddings, BM25 index, offline scoring) is
   done here, exactly once per experiment.
2. ``select(query)`` returns the indices (into ``candidates``) of the ``k``
   demonstrations to place in the prompt for ``query``, in prompt order.

Selectors that need to call the language model during selection (IDS, Se²,
Influence) receive the callables they need at construction time, so the
runner never has to special-case them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any


class BaseSelector(ABC):
    """Abstract base class for all demonstration selectors."""

    #: Short identifier used in configs, result files and the registry.
    name: str = "base"

    def __init__(self, k: int = 5):
        if k < 0:
            raise ValueError("k must be non-negative")
        self.k = k
        self.candidates: list[str] = []
        self.candidate_labels: list[Any] | None = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Pool binding
    # ------------------------------------------------------------------ #
    def fit(
        self,
        candidates: Sequence[str],
        candidate_labels: Sequence[Any] | None = None,
    ) -> BaseSelector:
        """
        Bind the demonstration pool. Subclasses should call
        ``super().fit(...)`` and then do their own pre-computation.
        """
        if candidate_labels is not None and len(candidate_labels) != len(candidates):
            raise ValueError("candidates and candidate_labels must have the same length")
        self.candidates = list(candidates)
        self.candidate_labels = list(candidate_labels) if candidate_labels is not None else None
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__}.fit(candidates) must be called before select()"
            )

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    @abstractmethod
    def select(self, query: str) -> list[int]:
        """Return indices into the fitted pool for ``query`` (prompt order)."""

    def select_batch(self, queries: Sequence[str]) -> list[list[int]]:
        """Default batch implementation: one ``select`` per query."""
        return [self.select(q) for q in queries]

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def get_config(self) -> dict[str, Any]:
        """Hyper-parameters worth recording alongside results."""
        return {"name": self.name, "k": self.k}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        cfg = ", ".join(f"{k}={v!r}" for k, v in self.get_config().items() if k != "name")
        return f"{type(self).__name__}({cfg})"
