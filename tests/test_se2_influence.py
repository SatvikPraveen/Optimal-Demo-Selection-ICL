"""
Tests for the Se² and Influence selectors.

The LM behind Se² is replaced with a deterministic fake scorer so the
beam-search logic (sequentiality, beam pruning, tie-breaking) is what gets
tested, independent of any model weights.
"""

import numpy as np
import pytest

from src.selection import InfluenceSelection, Se2

POOL = [f"demo {i}" for i in range(8)]
LABELS = [i % 4 for i in range(8)]


class _FakeScorer:
    """
    conditional_nll(context, target) = sum over demos in the context of a
    fixed per-demo cost, plus an interaction penalty when demo 0 and demo 1
    are both present. Lower is better, so the best single demo is #1 but
    the best *pair* avoids {0,1}.
    """

    cost = {i: 10.0 - i for i in range(8)}  # demo 7 cheapest, demo 0 dearest
    cost[1] = 0.5  # demo 1 is by far the best single demo

    def __init__(self):
        self.calls = 0

    def conditional_nll(self, contexts, targets):
        self.calls += 1
        out = []
        for ctx in contexts:
            demos = [int(t.split()[1]) for t in ctx.split("\n\n") if t.strip()]
            s = sum(self.cost[d] for d in demos)
            if 1 in demos and 7 in demos:
                s += 100.0  # strong negative interaction
            out.append(s)
        return np.array(out)


def test_se2_beam_search_is_sequential_and_order_aware(mock_sentence_transformer):
    scorer = _FakeScorer()
    sel = Se2(k=2, retrieve_k=8, beam_size=3, scorer=scorer).fit(POOL, LABELS)
    chosen = sel.select("a query")
    assert len(chosen) == 2 and len(set(chosen)) == 2
    # Greedy-by-single-score would take {1, 7}, but their interaction is
    # penalised, so the sequential score must avoid that pair.
    assert set(chosen) != {1, 7}
    assert 1 in chosen  # demo 1 is still worth having
    # first step scores 8 expansions in one batched call, second step 3*7
    assert scorer.calls == 2
    assert len(sel.last_beam_scores) == 3


def test_se2_greedy_beam_one(mock_sentence_transformer):
    sel = Se2(k=1, retrieve_k=8, beam_size=1, scorer=_FakeScorer()).fit(POOL)
    assert sel.select("q") == [1]


def test_se2_requires_scorer_and_fit(mock_sentence_transformer):
    with pytest.raises(RuntimeError):
        Se2(k=1).select("q")
    sel = Se2(k=1).fit(POOL)
    with pytest.raises(RuntimeError):
        sel.select("q")
    with pytest.raises(ValueError):
        Se2(k=1, beam_size=0)


def test_se2_retrieve_limits_candidate_set(mock_sentence_transformer):
    sel = Se2(k=2, retrieve_k=3, beam_size=2, scorer=_FakeScorer()).fit(POOL)
    cands = sel.retrieve(POOL[5])
    assert len(cands) == 3 and cands[0] == 5
    assert set(sel.select(POOL[5])) <= set(cands)


def test_se2_config(mock_sentence_transformer):
    cfg = Se2(k=3, retrieve_k=10, beam_size=2).get_config()
    assert cfg["name"] == "se2" and cfg["retrieve_k"] == 10 and cfg["beam_size"] == 2


# ---------------------------------------------------------------------- #
# Influence
# ---------------------------------------------------------------------- #
def _planted_evaluator(good=(2, 5), bad=(0,)):
    """Subset score = 0.5 + 0.2 per 'good' demo - 0.3 per 'bad' demo."""

    def f(indices):
        return 0.5 + 0.2 * sum(i in good for i in indices) - 0.3 * sum(i in bad for i in indices)

    return f


@pytest.mark.parametrize("estimator", ["difference", "ridge"])
def test_influence_recovers_planted_signal(estimator):
    sel = InfluenceSelection(
        k=2,
        evaluate_subset_fn=_planted_evaluator(),
        num_subsets=60,
        subset_size=3,
        estimator=estimator,
        coverage=1,
        seed=0,
    ).fit(POOL, LABELS)
    assert set(sel.select("any query")) == {2, 5}
    assert sel.select("another") == sel.select("any query")  # query-independent
    scores = sel.influence_scores
    assert np.nanargmin(scores) == 0
    assert not np.isnan(scores).any()  # coverage=1 => every candidate scored
    assert sel.coverage_stats()["never_sampled"] == 0


def test_influence_subsets_are_balanced_and_deduplicated():
    sel = InfluenceSelection(k=4, evaluate_subset_fn=lambda s: 0.0, num_subsets=20, seed=1)
    sel.fit(POOL, LABELS)
    for s in sel.subsets:
        assert len(s) == 4 and len(set(s)) == 4
        # 4 classes, subset of 4 -> exactly one per class
        assert sorted(LABELS[i] for i in s) == [0, 1, 2, 3]
    # k > num_classes still has no duplicates (the AG News notebook bug)
    sel6 = InfluenceSelection(k=6, evaluate_subset_fn=lambda s: 0.0, num_subsets=5, seed=1)
    sel6.fit(POOL, LABELS)
    assert all(len(set(s)) == 6 for s in sel6.subsets)


def test_influence_is_seeded_and_reports_unsampled():
    a = InfluenceSelection(k=1, evaluate_subset_fn=len, num_subsets=3, subset_size=2, seed=3)
    b = InfluenceSelection(k=1, evaluate_subset_fn=len, num_subsets=3, subset_size=2, seed=3)
    assert a.fit(POOL).subsets == b.fit(POOL).subsets
    # 3 subsets x 2 = 6 draws over 8 candidates: at least 2 never sampled
    assert np.isnan(a.influence_scores).sum() >= 2
    assert a.coverage_stats()["never_sampled"] >= 2


def test_influence_coverage_validation_and_persistence(tmp_path):
    with pytest.raises(ValueError):
        InfluenceSelection(k=2, evaluate_subset_fn=len, num_subsets=2, coverage=1).fit(POOL)

    sel = InfluenceSelection(
        k=2, evaluate_subset_fn=_planted_evaluator(), num_subsets=30, coverage=1, seed=0
    ).fit(POOL, LABELS)
    path = tmp_path / "scores.json"
    sel.save_scores(path)

    fresh = InfluenceSelection(k=2).load_scores(path, POOL, LABELS)
    assert fresh.select("q") == sel.select("q")
    assert np.allclose(fresh.influence_scores, sel.influence_scores, equal_nan=True)
    with pytest.raises(ValueError):
        InfluenceSelection(k=2).load_scores(path, POOL[:3])


def test_influence_argument_validation():
    with pytest.raises(ValueError):
        InfluenceSelection(estimator="magic")
    with pytest.raises(RuntimeError):
        InfluenceSelection(k=1).fit(POOL)
