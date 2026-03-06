"""
Optimal Demonstration Selection for In-Context Learning
========================================================

A modular framework for benchmarking demonstration selection methods for ICL.

Modules:
--------
- datasets: Dataset loading and preprocessing
- models: LLM model interfaces (GPT, LLaMA, Gemma)
- selection: Demonstration selection algorithms
- prompting: Prompt construction and inference
- evaluation: Metrics and benchmarking
- utils: Utilities and helpers
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import datasets
from . import models
from . import selection
from . import prompting
from . import evaluation
from . import utils

__all__ = [
    "datasets",
    "models",
    "selection",
    "prompting",
    "evaluation",
    "utils",
]
