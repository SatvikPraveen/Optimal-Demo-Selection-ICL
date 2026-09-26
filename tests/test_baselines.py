"""
Unit tests for the baseline selectors and the shared embedding helper.
"""

import numpy as np
import pytest

from src.selection import BM25Selector, RandomSelector, TopKSelector
from src.selection.baselines import simple_tokenize
from src.utils.embeddings import Embedder, cosine_top_k, l2_normalize

POOL = [
    "the movie was wonderful and moving",
    "a dull, lifeless film",
    "stock markets rallied on strong earnings",
    "the team won the championship game",
    "an absolutely wonderful movie",
    "government announces new trade policy",
]
LABELS = ["positive", "negative", "business", "sports", "positive", "world"]


def test_random_selector_is_seeded_and_without_replacement():
    a = RandomSelector(k=3, seed=0).fit(POOL)
    b = RandomSelector(k=3, seed=0).fit(POOL)
    sa, sb = a.select("q"), b.select("q")
    assert sa == sb
    assert len(sa) == 3 and len(set(sa)) == 3
    assert all(0 <= i < len(POOL) for i in sa)

    # Different seeds should (almost surely) differ over a few draws.
    c = RandomSelector(k=3, seed=1).fit(POOL)
    assert any(a.select("q") != c.select("q") for _ in range(5))


def test_random_selector_k_larger_than_pool():
    s = RandomSelector(k=10, seed=0).fit(POOL)
    assert sorted(s.select("q")) == list(range(len(POOL)))


def test_select_before_fit_raises():
    with pytest.raises(RuntimeError):
        RandomSelector(k=2).select("q")


def test_bm25_prefers_lexical_overlap():
    s = BM25Selector(k=2).fit(POOL, LABELS)
    top = s.select("a wonderful movie")
    assert set(top) <= {0, 4}
    assert s.select("championship game")[0] == 3
    # Reverse ordering puts the best match last.
    r = BM25Selector(k=2, reverse=True).fit(POOL)
    assert r.select("a wonderful movie")[-1] == s.select("a wonderful movie")[0]


def test_bm25_empty_query_is_deterministic():
    s = BM25Selector(k=2).fit(POOL)
    assert s.select("") == [0, 1]
    assert simple_tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_topk_selector_uses_cosine_similarity(mock_sentence_transformer):
    s = TopKSelector(k=2).fit(POOL, LABELS)
    assert s.candidate_embeddings.shape == (len(POOL), 384)
    # A query identical to a pool item must retrieve that item first.
    assert s.select(POOL[2])[0] == 2
    # Precomputed embeddings are accepted and used as-is.
    emb = np.eye(len(POOL), 8, dtype=np.float32)
    s2 = TopKSelector(k=1).fit(POOL, candidate_embeddings=emb)
    assert s2.candidate_embeddings is not None and s2.candidate_embeddings.shape == (6, 8)
    with pytest.raises(ValueError):
        TopKSelector(k=1).fit(POOL, candidate_embeddings=emb[:3])


def test_embedder_caches_and_normalizes(mock_sentence_transformer):
    e = Embedder("all-MiniLM-L6-v2")
    v = e.encode("hello")
    assert v.shape == (384,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5
    m = e.encode(["hello", "world", "hello"])
    assert m.shape == (3, 384)
    assert np.allclose(m[0], m[2])
    assert np.allclose(m[0], v)
    assert e.dim == 384


def test_cosine_top_k_is_deterministic_on_ties():
    pool = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    assert cosine_top_k(np.array([1, 0]), pool, 2).tolist() == [0, 1]
    assert cosine_top_k(np.array([0, 1]), pool, 5).tolist() == [2, 0, 1]
    assert np.allclose(np.linalg.norm(l2_normalize(np.array([[3.0, 4.0]])), axis=1), 1.0)


def test_get_config_round_trips_hyperparameters():
    cfg = BM25Selector(k=3, k1=1.2, b=0.5).get_config()
    assert cfg == {"name": "bm25", "k": 3, "k1": 1.2, "b": 0.5, "reverse": False}
    assert RandomSelector(k=2, seed=7).get_config()["seed"] == 7
