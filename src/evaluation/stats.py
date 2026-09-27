"""
Statistical utilities for reporting benchmark results.

All functions operate on *per-example* outcomes (typically 0/1 correctness
vectors) so that uncertainty reflects the finite test set, and on paired
outcomes when two methods were evaluated on the same test examples, which
is far more powerful than comparing two independent accuracies.

* :func:`bootstrap_ci` - percentile bootstrap CI of a statistic.
* :func:`paired_bootstrap_test` - bootstrap p-value for ``mean(a) > mean(b)``.
* :func:`mcnemar_test` - exact McNemar test for paired binary outcomes.
* :func:`permutation_test` - paired sign-flip permutation test.
* :func:`summarize_seeds` - mean / std / CI across independent runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats


def _as_array(x: Sequence[float]) -> np.ndarray:
    return np.asarray(list(x), dtype=np.float64)


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """
    Percentile bootstrap confidence interval of ``statistic(values)``.

    Returns ``{"point", "lower", "upper", "std_err"}``.
    """
    v = _as_array(values)
    if v.size == 0:
        return {
            "point": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "std_err": float("nan"),
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = np.array([statistic(v[i]) for i in idx])
    alpha = (1 - confidence) / 2
    return {
        "point": float(statistic(v)),
        "lower": float(np.quantile(boots, alpha)),
        "upper": float(np.quantile(boots, 1 - alpha)),
        "std_err": float(boots.std(ddof=1)) if n_boot > 1 else float("nan"),
    }


def paired_bootstrap_test(
    a: Sequence[float], b: Sequence[float], n_boot: int = 5000, seed: int = 0
) -> dict[str, float]:
    """
    Paired bootstrap for the difference ``mean(a) - mean(b)`` over the same
    examples. ``p_value`` is the one-sided probability that the bootstrap
    difference is <= 0 (i.e. evidence *against* ``a`` being better).
    """
    a, b = _as_array(a), _as_array(b)
    if a.shape != b.shape:
        raise ValueError("a and b must be paired (same length)")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    return {
        "delta": float(d.mean()),
        "lower": float(np.quantile(boots, 0.025)),
        "upper": float(np.quantile(boots, 0.975)),
        "p_value": float(np.mean(boots <= 0)),
    }


def mcnemar_test(a: Sequence[int], b: Sequence[int]) -> dict[str, float]:
    """
    Exact McNemar test for paired binary outcomes (e.g. correctness of two
    methods on the same test examples). Two-sided p-value from the binomial
    distribution of the discordant pairs.
    """
    a, b = _as_array(a).astype(int), _as_array(b).astype(int)
    if a.shape != b.shape:
        raise ValueError("a and b must be paired (same length)")
    n01 = int(((a == 0) & (b == 1)).sum())  # b right, a wrong
    n10 = int(((a == 1) & (b == 0)).sum())  # a right, b wrong
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        p = min(1.0, 2 * stats.binom.cdf(k, n, 0.5))
    return {"n_a_only": n10, "n_b_only": n01, "n_discordant": n, "p_value": float(p)}


def permutation_test(
    a: Sequence[float], b: Sequence[float], n_perm: int = 5000, seed: int = 0
) -> dict[str, float]:
    """
    Paired sign-flip permutation test of ``mean(a - b) == 0`` (two-sided).
    """
    d = _as_array(a) - _as_array(b)
    if d.size == 0:
        return {"delta": float("nan"), "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    observed = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    p = (np.sum(perm >= observed) + 1) / (n_perm + 1)
    return {"delta": float(d.mean()), "p_value": float(p)}


def summarize_seeds(scores: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    """
    Mean, sample std and t-interval across independent runs (seeds).
    With a single run the interval collapses to the point estimate.
    """
    v = _as_array(scores)
    n = v.size
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "n": 0,
        }
    mean = float(v.mean())
    if n == 1:
        return {"mean": mean, "std": 0.0, "lower": mean, "upper": mean, "n": 1}
    std = float(v.std(ddof=1))
    half = stats.t.ppf(0.5 + confidence / 2, n - 1) * std / np.sqrt(n)
    return {"mean": mean, "std": std, "lower": mean - half, "upper": mean + half, "n": int(n)}


def compare_methods(
    correct_by_method: dict[str, Sequence[int]], baseline: str
) -> dict[str, dict[str, float]]:
    """
    Compare every method to ``baseline`` on paired correctness vectors:
    McNemar p-value plus paired-bootstrap delta and CI.
    """
    if baseline not in correct_by_method:
        raise KeyError(f"baseline '{baseline}' not among methods")
    base = correct_by_method[baseline]
    out: dict[str, dict[str, float]] = {}
    for name, corr in correct_by_method.items():
        if name == baseline:
            continue
        pb = paired_bootstrap_test(corr, base)
        mc = mcnemar_test(corr, base)
        out[name] = {**pb, "mcnemar_p_value": mc["p_value"], "n_discordant": mc["n_discordant"]}
    return out
