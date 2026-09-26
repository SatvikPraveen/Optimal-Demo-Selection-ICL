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

__version__ = "0.2.0"
__author__ = "Satvik Praveen, Jonathan Tong, Kamisetty Yamini Preethi, Vinay Chandra Bandi"

from . import datasets, evaluation, models, prompting, selection, utils

__all__ = [
    "datasets",
    "models",
    "selection",
    "prompting",
    "evaluation",
    "utils",
]
