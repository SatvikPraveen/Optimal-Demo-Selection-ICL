"""
Unit tests for core package functionality.

No live API keys, network access, or GPU are required: the
mock_sentence_transformer / mock_gpt2_cone_backend fixtures (see conftest.py)
stand in for the real embedding/CoNE model downloads.
"""

import random

import numpy as np

from src.datasets import load_agnews, load_sst5
from src.evaluation import compute_accuracy, compute_metrics
from src.prompting import PromptBuilder
from src.selection import IDS, TopKCoNE
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


def test_topk_cone_init(mock_gpt2_cone_backend):
    embeddings = np.random.rand(10, 384)  # 10 samples, 384-dim embeddings
    texts = [f"Sample text {i}" for i in range(10)]

    selector = TopKCoNE(embeddings=embeddings, raw_texts=texts, k=3, retrieve_k=5)

    assert selector.k == 3
    assert selector.retrieve_k == 5


def test_ids_init(mock_sentence_transformer):
    selector = IDS(k=5, q=3)
    assert selector.k == 5
    assert selector.q == 3
