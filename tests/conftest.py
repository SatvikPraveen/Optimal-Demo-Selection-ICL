"""
Shared pytest fixtures.

CI has no OPENAI_API_KEY, no GPU, and shouldn't depend on network access being
fast or available. The fixtures below stand in for the three things in this
test suite that would otherwise reach out over the network: the
sentence-transformers embedding model, the gpt2 model used by TopKCoNE's CoNE
scoring, and the SST-5/AG News dataset downloads. They only replace the
network boundary (model/dataset loading) - the surrounding logic under test
(dataset filtering/shuffling/sampling, selector initialization, embedding
shapes) still runs for real.
"""

import numpy as np
import pandas as pd
import pytest

from datasets import Dataset


class _FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer with no download."""

    _DIM = 384

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text, **kwargs):
        if isinstance(text, (list, tuple)):
            return np.stack([self._encode_one(t) for t in text])
        return self._encode_one(text)

    def get_sentence_embedding_dimension(self):
        return self._DIM

    def _encode_one(self, text):
        # Deterministic per-text pseudo-embedding, no model weights involved.
        # hash() is salted per process for str, so use a stable digest.
        import zlib

        rng = np.random.RandomState(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)
        return rng.rand(self._DIM).astype(np.float32)


@pytest.fixture
def mock_sentence_transformer(monkeypatch):
    """Avoid downloading the real all-MiniLM-L6-v2 weights during CI."""
    import src.utils.embeddings as emb_module

    monkeypatch.setattr(emb_module, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer)
    # The shared Embedder caches model instances by name; make sure a real
    # model loaded by an earlier test never leaks into a mocked one.
    emb_module.Embedder.clear_shared_models()
    yield
    emb_module.Embedder.clear_shared_models()


class FakeScorer:
    """
    Deterministic stand-in for src.models.scoring.LMScorer: the NLL of a
    continuation given a context is a fixed function of the two strings,
    so selectors that rank by it can be tested without model weights.
    """

    def __init__(self, *args, **kwargs):
        self.num_forward_passes = 0
        self.num_scored_tokens = 0
        self.model = None

    def conditional_nll(self, contexts, continuations):
        import zlib

        self.num_forward_passes += 1
        out = []
        for c, x in zip(contexts, continuations):
            h = zlib.crc32((c + "\x00" + x).encode("utf-8")) & 0xFFFFFFFF
            out.append(1.0 + (h % 1000) / 100.0)
        return np.array(out)

    def sequence_nll(self, texts):
        return self.conditional_nll([""] * len(texts), texts)

    def choice_logprobs(self, prompt, choices):
        return -self.conditional_nll([prompt] * len(choices), list(choices))

    def choice_logprobs_batch(self, prompts, choices):
        return np.stack([self.choice_logprobs(p, choices) for p in prompts])

    def reset_counters(self):
        self.num_forward_passes = 0
        self.num_scored_tokens = 0


@pytest.fixture
def mock_gpt2_cone_backend(monkeypatch):
    """Avoid downloading gpt2 weights: LMScorer.from_pretrained -> FakeScorer."""
    from src.models import scoring

    monkeypatch.setattr(
        scoring.LMScorer, "from_pretrained", classmethod(lambda cls, *a, **k: FakeScorer())
    )
    return FakeScorer


@pytest.fixture
def mock_hf_datasets(monkeypatch):
    """Avoid hitting the HuggingFace Hub for SST-5 / AG News during CI."""
    import importlib

    # src/datasets/__init__.py does `from .load_sst5 import load_sst5`, which
    # shadows the `load_sst5` submodule name with the function on the package
    # object - so `import src.datasets.load_sst5 as x` would resolve `x` to
    # that function, not the submodule. Go through sys.modules explicitly.
    sst5_module = importlib.import_module("src.datasets.load_sst5")
    agnews_module = importlib.import_module("src.datasets.load_agnews")

    labels = ["very negative", "negative", "neutral", "positive", "very positive"]
    fake_sst5 = Dataset.from_dict(
        {
            "text": [f"sample sentence {i}" for i in range(100)],
            "label_text": [labels[i % len(labels)] for i in range(100)],
        }
    )
    monkeypatch.setattr(sst5_module, "load_dataset", lambda *a, **k: fake_sst5)

    fake_agnews = pd.DataFrame(
        {
            "text": [f"news blurb {i}" for i in range(100)],
            "label": [i % 4 for i in range(100)],
        }
    )
    monkeypatch.setattr(agnews_module.pd, "read_parquet", lambda *a, **k: fake_agnews)
