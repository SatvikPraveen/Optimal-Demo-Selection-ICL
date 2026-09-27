import numpy as np
import pytest

from src.evaluation import (
    bootstrap_ci,
    compare_methods,
    mcnemar_test,
    paired_bootstrap_test,
    permutation_test,
    summarize_seeds,
)
from src.evaluation.metrics import compute_confidence_interval


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    x = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    ci = bootstrap_ci(x, n_boot=500, seed=1)
    assert ci["point"] == pytest.approx(0.7)
    assert ci["lower"] <= ci["point"] <= ci["upper"]
    assert ci == bootstrap_ci(x, n_boot=500, seed=1)
    assert np.isnan(bootstrap_ci([])["point"])


def test_paired_tests_detect_clear_difference():
    rng = np.random.default_rng(0)
    base = rng.random(300) < 0.5
    better = base | (rng.random(300) < 0.4)  # strictly dominates
    pb = paired_bootstrap_test(better.astype(int), base.astype(int), n_boot=1000)
    assert pb["delta"] > 0 and pb["p_value"] < 0.01 and pb["lower"] > 0
    mc = mcnemar_test(better.astype(int), base.astype(int))
    assert mc["p_value"] < 0.001 and mc["n_b_only"] == 0
    pt = permutation_test(better.astype(int), base.astype(int), n_perm=1000)
    assert pt["p_value"] < 0.01


def test_paired_tests_null_case():
    a = [1, 0, 1, 0, 1, 0]
    assert mcnemar_test(a, a)["p_value"] == 1.0
    assert mcnemar_test(a, a)["n_discordant"] == 0
    assert permutation_test(a, a)["p_value"] > 0.5
    with pytest.raises(ValueError):
        paired_bootstrap_test([1, 0], [1])


def test_summarize_seeds_and_legacy_ci():
    s = summarize_seeds([0.8, 0.82, 0.78])
    assert s["n"] == 3 and s["mean"] == pytest.approx(0.8)
    assert s["lower"] < 0.8 < s["upper"]
    one = summarize_seeds([0.5])
    assert one["lower"] == one["upper"] == 0.5 and one["std"] == 0.0
    mean, lo, hi = compute_confidence_interval([0.8, 0.82, 0.78])
    assert (mean, lo, hi) == (s["mean"], s["lower"], s["upper"])


def test_compare_methods():
    rng = np.random.default_rng(1)
    base = (rng.random(200) < 0.5).astype(int)
    res = compare_methods({"random": base, "good": np.minimum(base + 1, 1), "same": base}, "random")
    assert set(res) == {"good", "same"}
    assert res["good"]["delta"] > 0 and res["good"]["mcnemar_p_value"] < 0.001
    assert res["same"]["delta"] == 0
    with pytest.raises(KeyError):
        compare_methods({"a": base}, "zzz")
