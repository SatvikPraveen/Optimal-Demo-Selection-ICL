"""
Influence-based demonstration selection (in-context influence).

Reference: Nguyen & Wong, "In-context Example Selection with Influences"
(2023). The "influence" of a training example on ICL performance is
estimated *without gradients* by datamodel-style subset sampling:

1. Sample ``num_subsets`` random demonstration subsets ``S_i`` of size
   ``subset_size`` from the pool (class-balanced when labels are given).
2. Evaluate each subset as a fixed prompt on a held-out validation set,
   giving a score ``D_i`` (accuracy by default).
3. Estimate the influence of candidate ``j``:

   * ``estimator="difference"`` (the paper's estimator, and what the
     archived notebooks do):
     ``I_j = mean(D_i : j in S_i) - mean(D_i : j not in S_i)``
   * ``estimator="ridge"``: the datamodel regression
     ``D ~ w . 1[j in S]``, solved with ridge; ``I_j = w_j``. This shares
     statistical strength across subsets and is far less noisy when each
     candidate appears only a handful of times.

4. Select the ``k`` candidates with the highest influence. Selection is
   query-independent: the same demonstrations are used for every query.

Fixes relative to ``notebooks_archive/ICINF/``: subset membership is
tracked by index (not dict equality), the RNG is seeded, subsets never
contain duplicates, class balancing works for any ``k`` vs. ``num_classes``
(the AG News notebooks sampled with replacement when ``k > num_classes``),
and candidates that were never sampled are reported as ``NaN`` rather than
silently scored 0. The ``coverage`` option guarantees every candidate is
sampled a minimum number of times so that the difference estimator is
defined for the whole pool.

The expensive part, evaluating a subset on the validation set, is delegated
to a caller-supplied ``evaluate_subset_fn(indices) -> float`` so the
selector is model-agnostic (and unit-testable without an LM).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseSelector


class InfluenceSelection(BaseSelector):
    """
    Subset-sampling influence estimation for demonstration selection.

    Args:
        k: number of demonstrations to select.
        evaluate_subset_fn: callable mapping a list of pool indices (the demo
            subset, in prompt order) to a scalar validation score (higher is
            better). Required for ``fit`` unless precomputed scores are
            loaded with :meth:`load_scores`.
        num_subsets: number of random subsets ``M``.
        subset_size: size of each subset (defaults to ``k``).
        estimator: ``"difference"`` or ``"ridge"`` (see module docstring).
        ridge_alpha: L2 penalty for the ridge estimator.
        class_balanced: spread each subset's demonstrations across classes
            when ``candidate_labels`` are provided.
        coverage: minimum number of subsets every candidate must appear in.
            ``0`` means purely random sampling (as in the reference code).
            Any positive value first cycles through random permutations of
            the pool so each candidate appears at least that many times,
            then fills the remaining subsets randomly.
        seed: RNG seed for subset sampling.
    """

    name = "influence"

    def __init__(
        self,
        k: int = 5,
        evaluate_subset_fn: Callable[[list[int]], float] | None = None,
        num_subsets: int = 100,
        subset_size: int | None = None,
        estimator: str = "difference",
        ridge_alpha: float = 1.0,
        class_balanced: bool = True,
        coverage: int = 0,
        seed: int = 0,
    ):
        super().__init__(k=k)
        if estimator not in {"difference", "ridge"}:
            raise ValueError("estimator must be 'difference' or 'ridge'")
        if num_subsets < 1:
            raise ValueError("num_subsets must be >= 1")
        self.evaluate_subset_fn = evaluate_subset_fn
        self.num_subsets = num_subsets
        self.subset_size = subset_size if subset_size is not None else k
        self.estimator = estimator
        self.ridge_alpha = ridge_alpha
        self.class_balanced = class_balanced
        self.coverage = coverage
        self.seed = seed

        self.subsets: list[list[int]] = []
        self.subset_scores: np.ndarray = np.zeros(0)
        self.influence_scores: np.ndarray = np.zeros(0)
        self.selected_indices: list[int] = []

    # ------------------------------------------------------------------ #
    # Subset sampling
    # ------------------------------------------------------------------ #
    def _sample_one_subset(self, rng: np.random.Generator, size: int) -> list[int]:
        n = len(self.candidates)
        size = min(size, n)
        if not self.class_balanced or self.candidate_labels is None:
            return rng.choice(n, size=size, replace=False).tolist()

        by_class: dict[Any, list[int]] = defaultdict(list)
        for i, lab in enumerate(self.candidate_labels):
            by_class[lab].append(i)
        classes = list(by_class.keys())
        rng.shuffle(classes)

        chosen: list[int] = []
        # Round-robin over classes until the subset is full: one per class
        # first, then a second per class, etc. Never re-uses an index.
        pools = {c: rng.permutation(by_class[c]).tolist() for c in classes}
        while len(chosen) < size:
            progressed = False
            for c in classes:
                if pools[c]:
                    chosen.append(pools[c].pop())
                    progressed = True
                    if len(chosen) == size:
                        break
            if not progressed:
                break
        rng.shuffle(chosen)
        return chosen

    def _sample_subsets(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed)
        n = len(self.candidates)
        size = min(self.subset_size, n)
        subsets: list[list[int]] = []

        if self.coverage > 0:
            # Cycle through fresh permutations so every index shows up
            # `coverage` times before random sampling takes over.
            for _ in range(self.coverage):
                perm = rng.permutation(n).tolist()
                for start in range(0, n, size):
                    chunk = perm[start : start + size]
                    if len(chunk) < size:
                        # Top up the tail chunk with unused indices.
                        extra = [i for i in rng.permutation(n).tolist() if i not in chunk]
                        chunk = chunk + extra[: size - len(chunk)]
                    subsets.append(chunk)
            if len(subsets) > self.num_subsets:
                raise ValueError(
                    f"coverage={self.coverage} over {n} candidates needs at least "
                    f"{len(subsets)} subsets, but num_subsets={self.num_subsets}"
                )

        while len(subsets) < self.num_subsets:
            subsets.append(self._sample_one_subset(rng, size))
        return subsets

    # ------------------------------------------------------------------ #
    # Influence estimation
    # ------------------------------------------------------------------ #
    def _membership_matrix(self) -> np.ndarray:
        m = np.zeros((len(self.subsets), len(self.candidates)), dtype=np.float64)
        for i, s in enumerate(self.subsets):
            m[i, s] = 1.0
        return m

    def _estimate_influence(self) -> np.ndarray:
        x = self._membership_matrix()
        d = np.asarray(self.subset_scores, dtype=np.float64)
        n_in = x.sum(axis=0)
        n_out = len(d) - n_in

        if self.estimator == "difference":
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_in = (x * d[:, None]).sum(axis=0) / n_in
                mean_out = ((1 - x) * d[:, None]).sum(axis=0) / n_out
            inf = mean_in - mean_out
            inf[(n_in == 0) | (n_out == 0)] = np.nan
            return inf

        # Ridge datamodel: D = b + X w, minimise ||D - b - Xw||^2 + alpha ||w||^2
        xc = x - x.mean(axis=0, keepdims=True)
        dc = d - d.mean()
        a = xc.T @ xc + self.ridge_alpha * np.eye(x.shape[1])
        w = np.linalg.solve(a, xc.T @ dc)
        w[n_in == 0] = np.nan
        return w

    def _rank(self) -> list[int]:
        scores = np.where(np.isnan(self.influence_scores), -np.inf, self.influence_scores)
        order = np.lexsort((np.arange(len(scores)), -scores))
        return order[: min(self.k, len(scores))].tolist()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        candidates: Sequence[str],
        candidate_labels: Sequence[Any] | None = None,
        evaluate_subset_fn: Callable[[list[int]], float] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> InfluenceSelection:
        """
        Sample subsets, evaluate them and rank the pool. ``progress(i, M)``
        is called after each subset evaluation if given.
        """
        super().fit(candidates, candidate_labels)
        if evaluate_subset_fn is not None:
            self.evaluate_subset_fn = evaluate_subset_fn
        if self.evaluate_subset_fn is None:
            raise RuntimeError("InfluenceSelection.fit needs evaluate_subset_fn")

        self.subsets = self._sample_subsets()
        scores = np.empty(len(self.subsets))
        for i, s in enumerate(self.subsets):
            scores[i] = float(self.evaluate_subset_fn(list(s)))
            if progress is not None:
                progress(i + 1, len(self.subsets))
        self.subset_scores = scores
        self.influence_scores = self._estimate_influence()
        self.selected_indices = self._rank()
        return self

    def select(self, query: str) -> list[int]:
        """Query-independent: returns the top-``k`` pool indices by influence."""
        self._check_fitted()
        if not self.selected_indices and len(self.influence_scores) == 0:
            raise RuntimeError("fit() (or load_scores()) must run before select()")
        return list(self.selected_indices)

    def coverage_stats(self) -> dict[str, float]:
        """How often candidates were sampled (useful for reporting)."""
        counts = self._membership_matrix().sum(axis=0)
        return {
            "num_candidates": float(len(counts)),
            "never_sampled": float((counts == 0).sum()),
            "min_appearances": float(counts.min()) if len(counts) else 0.0,
            "mean_appearances": float(counts.mean()) if len(counts) else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Persistence: the subset evaluations are the expensive part.
    # ------------------------------------------------------------------ #
    def save_scores(self, path: str | Path) -> None:
        payload = {
            "config": self.get_config(),
            "num_candidates": len(self.candidates),
            "subsets": self.subsets,
            "subset_scores": self.subset_scores.tolist(),
            "influence_scores": [None if np.isnan(v) else float(v) for v in self.influence_scores],
            "selected_indices": self.selected_indices,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    def load_scores(
        self,
        path: str | Path,
        candidates: Sequence[str],
        candidate_labels: Sequence[Any] | None = None,
    ) -> InfluenceSelection:
        payload = json.loads(Path(path).read_text())
        if payload["num_candidates"] != len(candidates):
            raise ValueError("saved scores were computed for a different pool size")
        BaseSelector.fit(self, candidates, candidate_labels)
        self.subsets = [list(s) for s in payload["subsets"]]
        self.subset_scores = np.asarray(payload["subset_scores"], dtype=np.float64)
        self.influence_scores = np.asarray(
            [np.nan if v is None else v for v in payload["influence_scores"]], dtype=np.float64
        )
        self.selected_indices = self._rank()
        return self

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "num_subsets": self.num_subsets,
            "subset_size": self.subset_size,
            "estimator": self.estimator,
            "ridge_alpha": self.ridge_alpha,
            "class_balanced": self.class_balanced,
            "coverage": self.coverage,
            "seed": self.seed,
        }
