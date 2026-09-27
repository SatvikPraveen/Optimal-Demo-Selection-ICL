"""
Evaluation metrics
"""

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
    Mean and t-interval across runs. Kept for backwards compatibility;
    prefer :func:`src.evaluation.stats.summarize_seeds` /
    :func:`src.evaluation.stats.bootstrap_ci`.
    """
    from .stats import summarize_seeds

    s = summarize_seeds(scores, confidence)
    return s["mean"], s["lower"], s["upper"]
