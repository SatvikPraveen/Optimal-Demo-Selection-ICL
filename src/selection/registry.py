"""
Build selectors by name with the shared resources a benchmark run provides.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..utils.embeddings import Embedder
from .base import BaseSelector
from .baselines import BM25Selector, RandomSelector, TopKSelector
from .ids import IDS
from .influence import InfluenceSelection
from .rdes import RDES
from .se2 import Se2
from .topk_cone import TopKCoNE

SELECTORS: dict[str, type[BaseSelector]] = {
    "random": RandomSelector,
    "topk": TopKSelector,
    "sbert": TopKSelector,  # alias
    "knn": TopKSelector,  # alias
    "bm25": BM25Selector,
    "topk_cone": TopKCoNE,
    "ids": IDS,
    "rdes": RDES,
    "se2": Se2,
    "influence": InfluenceSelection,
}

#: Selectors that call the evaluated language model during selection.
NEEDS_INFERENCE = {"ids"}
#: Selectors that score with a causal LM.
NEEDS_SCORER = {"topk_cone", "se2"}
#: Selectors that need a validation set and a subset evaluator.
NEEDS_VALIDATION = {"influence"}


def available_selectors() -> list[str]:
    return sorted(SELECTORS)


def build_selector(
    name: str,
    params: dict[str, Any] | None = None,
    *,
    k: int = 5,
    seed: int = 0,
    embedder: Embedder | None = None,
    scorer: Any | None = None,
    zero_shot_cot_fn: Callable[[str], str] | None = None,
    icl_fn: Callable[[str, list[str]], str] | None = None,
    evaluate_subset_fn: Callable[[list[int]], float] | None = None,
) -> BaseSelector:
    """
    Instantiate selector ``name`` with ``params`` (method hyper-parameters,
    which may override ``k``) and the shared resources it needs.
    """
    if name not in SELECTORS:
        raise KeyError(f"Unknown selector '{name}'. Available: {available_selectors()}")
    cls = SELECTORS[name]
    kwargs: dict[str, Any] = {"k": k, **(params or {})}
    kwargs.pop("name", None)

    if name in ("topk", "sbert", "knn", "se2", "topk_cone", "ids", "rdes") and embedder is not None:
        kwargs.setdefault("embedder", embedder)
    if name in ("random", "rdes", "influence"):
        kwargs.setdefault("seed", seed)
    if name in NEEDS_SCORER:
        if scorer is None:
            raise ValueError(f"selector '{name}' needs a scorer")
        kwargs.setdefault("scorer", scorer)
    if name in NEEDS_INFERENCE:
        if zero_shot_cot_fn is None or icl_fn is None:
            raise ValueError("selector 'ids' needs zero_shot_cot_fn and icl_fn")
        kwargs.setdefault("zero_shot_cot_fn", zero_shot_cot_fn)
        kwargs.setdefault("icl_fn", icl_fn)
    if name in NEEDS_VALIDATION:
        if evaluate_subset_fn is None:
            raise ValueError("selector 'influence' needs evaluate_subset_fn")
        kwargs.setdefault("evaluate_subset_fn", evaluate_subset_fn)
    return cls(**kwargs)
