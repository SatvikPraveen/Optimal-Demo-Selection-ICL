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
        if isinstance(text, list):
            return np.stack([self._encode_one(t) for t in text])
        return self._encode_one(text)

    def _encode_one(self, text):
        # Deterministic per-text pseudo-embedding, no model weights involved.
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        return rng.rand(self._DIM).astype(np.float32)


@pytest.fixture
def mock_sentence_transformer(monkeypatch):
    """Avoid downloading the real all-MiniLM-L6-v2 weights during CI."""
    import src.selection.ids as ids_module

    monkeypatch.setattr(ids_module, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer)


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _FakeGPT2Tokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, text, return_tensors="pt"):
        import torch

        n_tokens = max(1, len(text.split()))
        return _FakeBatchEncoding(input_ids=torch.randint(0, 1000, (1, n_tokens)))


class _FakeGPT2Output:
    def __init__(self, loss):
        self.loss = loss


class _FakeGPT2Model:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, labels=None):
        import torch

        return _FakeGPT2Output(torch.tensor(0.5))


@pytest.fixture
def mock_gpt2_cone_backend(monkeypatch):
    """Avoid downloading the real gpt2 weights used by TopKCoNE's CoNE scoring."""
    import src.selection.topk_cone as topk_module

    monkeypatch.setattr(topk_module, "GPT2TokenizerFast", _FakeGPT2Tokenizer)
    monkeypatch.setattr(topk_module, "GPT2LMHeadModel", _FakeGPT2Model)


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
