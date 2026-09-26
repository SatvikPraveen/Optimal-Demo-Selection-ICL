"""
Demonstration selection algorithms for ICL.

All selectors share the :class:`BaseSelector` protocol: ``fit(candidates,
labels)`` binds the demonstration pool, ``select(query)`` returns pool
indices in prompt order.
"""

from .base import BaseSelector
from .baselines import BM25Selector, RandomSelector, TopKSelector
from .ids import IDS
from .influence import InfluenceSelection
from .rdes import RDES
from .se2 import Se2
from .topk_cone import TopKCoNE

__all__ = [
    "BaseSelector",
    "RandomSelector",
    "TopKSelector",
    "BM25Selector",
    "TopKCoNE",
    "IDS",
    "RDES",
    "Se2",
    "InfluenceSelection",
]
