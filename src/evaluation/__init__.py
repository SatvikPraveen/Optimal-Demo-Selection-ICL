"""
Evaluation: metrics, statistics and prediction parsing.
"""

from .metrics import compute_accuracy, compute_f1, compute_metrics
from .parsing import UNKNOWN, normalize_label, parse_prediction, parse_predictions
from .stats import (
    bootstrap_ci,
    compare_methods,
    mcnemar_test,
    paired_bootstrap_test,
    permutation_test,
    summarize_seeds,
)

__all__ = [
    "compute_accuracy",
    "compute_f1",
    "compute_metrics",
    "UNKNOWN",
    "normalize_label",
    "parse_prediction",
    "parse_predictions",
    "bootstrap_ci",
    "paired_bootstrap_test",
    "mcnemar_test",
    "permutation_test",
    "summarize_seeds",
    "compare_methods",
]
