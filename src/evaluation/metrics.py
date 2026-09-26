"""
Evaluation metrics
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_accuracy(predictions: list[str], labels: list[str]) -> float:
    """
    Compute accuracy.

    Args:
        predictions: Model predictions
        labels: Ground truth labels

    Returns:
        Accuracy score
    """
    return accuracy_score(labels, predictions)


def compute_f1(predictions: list[str], labels: list[str], average: str = "macro") -> float:
    """
    Compute F1 score.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        average: Averaging method ('micro', 'macro', 'weighted')

    Returns:
        F1 score
    """
    return f1_score(labels, predictions, average=average)


def compute_metrics(
    predictions: list[str], labels: list[str], average: str = "macro"
) -> dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        average: Averaging method for multi-class metrics

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average=average, zero_division=0),
        "precision": precision_score(labels, predictions, average=average, zero_division=0),
        "recall": recall_score(labels, predictions, average=average, zero_division=0),
    }

    return metrics


def compute_confidence_interval(scores: list[float], confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval using bootstrap.

    Args:
        scores: List of scores from multiple runs
        confidence: Confidence level

    Returns:
        (mean, lower_bound, upper_bound)
    """
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    n = len(scores)

    # Using normal approximation
    from scipy import stats

    interval = stats.t.interval(confidence, n - 1, loc=mean, scale=std / np.sqrt(n))

    return mean, interval[0], interval[1]
