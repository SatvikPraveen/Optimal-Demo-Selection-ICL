"""
Unit tests for core package functionality.

No live API keys, network access, or GPU are required: the
mock_sentence_transformer / mock_gpt2_cone_backend fixtures (see conftest.py)
stand in for the real embedding/CoNE model downloads.
"""

import random

import numpy as np
import pytest

from src.datasets import load_agnews, load_sst5
from src.evaluation import compute_accuracy, compute_metrics
from src.prompting import PromptBuilder
from src.selection import IDS, RDES, TopKCoNE
from src.utils import set_seed, setup_logger


def test_imports():
    assert load_sst5 and load_agnews
    assert TopKCoNE and IDS
    assert compute_metrics and compute_accuracy
    assert set_seed and setup_logger
    assert PromptBuilder


def test_seed_control():
    set_seed(42)
    np1 = np.random.rand(5)
    r1 = random.random()

    set_seed(42)
    np2 = np.random.rand(5)
    r2 = random.random()

    assert np.allclose(np1, np2), "NumPy random state not reproducible"
    assert r1 == r2, "Python random state not reproducible"


def test_prompt_builder():
    builder = PromptBuilder(task_instruction="Classify sentiment")

    demos = [
        "Text: Great movie! Sentiment: positive",
        "Text: Terrible film. Sentiment: negative",
    ]
    query = "Text: Amazing experience! Sentiment:"

    prompt = builder.build_prompt(demos, query)
    assert "Classify sentiment" in prompt
    assert "Great movie" in prompt
    assert "Amazing experience" in prompt


def test_evaluation_metrics():
    predictions = ["positive", "negative", "positive", "neutral"]
    labels = ["positive", "positive", "positive", "neutral"]

    accuracy = compute_accuracy(predictions, labels)
    metrics = compute_metrics(predictions, labels)

    assert 0.0 <= accuracy <= 1.0
    assert accuracy == 0.75
    assert "accuracy" in metrics
    assert "f1" in metrics


def test_logger():
    logger = setup_logger("test", level=30)  # WARNING level
    logger.info("This should not appear")
    logger.warning("This warning should appear")


def test_topk_cone_init(mock_sentence_transformer, mock_gpt2_cone_backend):
    embeddings = np.random.rand(10, 384)  # 10 samples, 384-dim embeddings
    texts = [f"Sample text {i}" for i in range(10)]

    # Legacy constructor form still fits the pool immediately.
    selector = TopKCoNE(embeddings=embeddings, raw_texts=texts, k=3, retrieve_k=5)
    assert selector.k == 3
    assert selector.retrieve_k == 5
    assert selector.embeddings.shape == (10, 384)
    assert selector.raw_texts == texts


def test_topk_cone_reranks_within_retrieved_set(mock_sentence_transformer, mock_gpt2_cone_backend):
    texts = [f"Sample text {i}" for i in range(10)]
    selector = TopKCoNE(k=2, retrieve_k=4).fit(texts)
    q_emb = selector.embedder.encode(texts[3])
    retrieved = selector.get_topk(q_emb).tolist()
    assert retrieved[0] == 3 and len(retrieved) == 4
    chosen = selector.select(texts[3])
    assert len(chosen) == 2 and set(chosen) <= set(retrieved)
    # legacy entry point agrees with the new one
    assert selector.select_demonstrations(q_emb, texts[3]) == chosen
    # CoNE ordering is by ascending conditional NLL
    nll = [selector.last_scores[i] for i in chosen]
    assert nll == sorted(nll)


def test_ids_init(mock_sentence_transformer):
    selector = IDS(k=5, q=3)
    assert selector.k == 5
    assert selector.q == 3


def test_ids_iterates_on_reasoning_path(mock_sentence_transformer):
    pool = [f"demo {i}" for i in range(6)]
    calls = []

    def zero_shot_cot(query):
        calls.append(("cot", query))
        return "step one. Therefore, positive"

    def icl(query, demos):
        calls.append(("icl", query, tuple(demos)))
        # make each iteration's rationale different so retrieval can move
        return f"reasoning {len(calls)}. Therefore, positive"

    selector = IDS(k=2, q=3, zero_shot_cot_fn=zero_shot_cot, icl_fn=icl).fit(pool)
    chosen = selector.select("the query")
    assert len(chosen) == 2 and len(set(chosen)) == 2
    assert [c[0] for c in calls] == ["cot", "icl", "icl", "icl"]
    assert len(selector.last_answers) == 3
    assert len(selector.last_selections) == 3 and selector.last_selections[-1] == chosen

    # Legacy one-shot API produces the same selection.
    calls.clear()
    legacy = IDS(k=2, q=3)
    assert legacy.select_demonstrations("the query", pool, zero_shot_cot, icl) == chosen


def test_ids_requires_callbacks(mock_sentence_transformer):
    import pytest

    with pytest.raises(RuntimeError):
        IDS(k=1).fit(["a", "b"]).select("q")


def test_rdes_init_and_selection(mock_sentence_transformer):
    # Tiny synthetic pool, not a full-scale training run -- this only checks
    # the online Q-learning loop runs and returns something sane, not that
    # it has converged to a good policy.
    candidates = [f"demo text {i}" for i in range(6)]
    labels = [0, 0, 1, 1, 2, 2]

    selector = RDES(k=2).fit(candidates, labels)
    assert selector.k == 2
    assert selector.num_classes == 3

    selected = selector.select("a query")
    assert len(selected) == 2
    assert len(set(selected)) == 2  # no duplicate picks
    assert all(0 <= i < len(candidates) for i in selected)
    assert len(selector.q_table) > 0  # the online update actually ran
    assert selector.select_demonstrations("a query")  # legacy alias


def test_rdes_reward_balances_relevance_and_diversity(mock_sentence_transformer):
    candidates = [f"demo text {i}" for i in range(6)]
    labels = [0, 0, 1, 1, 2, 2]
    selector = RDES(k=2, relevance_weight=0.0).fit(candidates, labels)
    q = selector.embedder.encode("a query")
    # two different labels -> maximal entropy for 2 picks over... (2 of 3 classes)
    assert selector.reward(q, [0, 2]) > selector.reward(q, [0, 1])
    # pure relevance: a demo identical to the query scores 1.0
    rel = RDES(k=1, relevance_weight=1.0).fit(candidates, labels)
    assert rel.reward(rel.embedder.encode(candidates[4]), [4]) == pytest.approx(1.0, abs=1e-5)


def test_rdes_reproducible_with_seed(mock_sentence_transformer):
    candidates = [f"demo text {i}" for i in range(6)]
    labels = [0, 0, 1, 1, 2, 2]

    def run(seed=None):
        set_seed(42)
        selector = RDES(k=2, epsilon=0.5, seed=seed).fit(candidates, labels)
        return [selector.select("a query") for _ in range(5)]

    assert run() == run()  # derived from the global RNG set by set_seed
    assert run(7) == run(7)  # explicit seed
