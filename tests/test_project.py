"""
Integration-style tests exercising dataset loading, selection, prompting,
and evaluation together.

No live API keys, network access, or GPU are required: the mock_hf_datasets
fixture stands in for the real SST-5 / AG News downloads, and
mock_sentence_transformer stands in for the real embedding model download
(see conftest.py). The original version of this test used a real
sentence-transformers encoder directly; here we get embeddings via
IDS.encode_text() instead, so there is a single mocked embedding boundary.
"""

import numpy as np

from src.datasets import load_agnews, load_sst5
from src.evaluation import compute_accuracy, compute_f1
from src.prompting import PromptBuilder
from src.selection import IDS, TopKCoNE


def test_dataset_loading(mock_hf_datasets):
    texts_sst, labels_sst = load_sst5("test", 50, seed=42)
    texts_ag, labels_ag = load_agnews("test", 20, seed=42)

    assert len(texts_sst) == 50
    assert len(texts_ag) == 20
    assert set(labels_sst) <= {
        "very negative",
        "negative",
        "neutral",
        "positive",
        "very positive",
    }
    assert set(labels_ag) <= {"World", "Sports", "Business", "Technology"}


def test_selection_algorithms_and_embeddings(
    mock_sentence_transformer, mock_gpt2_cone_backend, mock_hf_datasets
):
    texts_sst, _ = load_sst5("test", 50, seed=42)

    ids_selector = IDS(k=5, q=3)
    assert ids_selector.k == 5
    assert ids_selector.q == 3

    sample_texts = texts_sst[:30]
    embeddings = np.array([ids_selector.encode_text(t) for t in sample_texts])
    topk_selector = TopKCoNE(embeddings, sample_texts, k=3, retrieve_k=10)

    assert topk_selector.k == 3
    assert embeddings.shape == (30, 384)


def test_prompt_building():
    demonstrations = [
        "Input: Great movie! Output: positive",
        "Input: Terrible film. Output: negative",
    ]
    builder = PromptBuilder(task_instruction="Classify the sentiment:")
    prompt = builder.build_prompt(demonstrations, "Input: Good acting.")

    assert "Classify the sentiment" in prompt
    assert len(prompt) > 0


def test_evaluation_metrics():
    predictions = ["positive", "negative", "neutral", "positive"]
    labels = ["positive", "positive", "neutral", "negative"]

    accuracy = compute_accuracy(predictions, labels)
    f1 = compute_f1(predictions, labels, average="weighted")

    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_similarity_based_retrieval(mock_sentence_transformer, mock_hf_datasets):
    texts_sst, _ = load_sst5("test", 50, seed=42)

    ids_selector = IDS(k=5, q=3)
    embeddings = np.array([ids_selector.encode_text(t) for t in texts_sst[:30]])

    test_embedding = embeddings[0].reshape(1, -1)
    similarities = np.dot(embeddings, test_embedding.T).flatten()
    top_k_indices = np.argsort(similarities)[-5:][::-1]

    assert len(top_k_indices) == 5
    assert top_k_indices[0] == 0  # a sample is most similar to itself
