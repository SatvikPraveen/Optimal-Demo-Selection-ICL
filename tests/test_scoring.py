"""
Tests for src.models.scoring.LMScorer.

Uses a tiny randomly-initialised GPT-2 (built from a config, no download)
and a whitespace "tokenizer" so the masking / batching arithmetic is tested
on real tensors without any network access.
"""

import numpy as np
import pytest
import torch

from src.models.scoring import LMScorer


class _WhitespaceTokenizer:
    """Maps each whitespace-separated word to a stable id in [1, vocab)."""

    vocab_size = 97
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True, **kwargs):
        ids = [1 + (abs(hash(w)) % (self.vocab_size - 1)) for w in text.split()]
        return {"input_ids": ids}


@pytest.fixture(scope="module")
def tiny_scorer():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=97, n_positions=32, n_embd=16, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(cfg)
    return LMScorer(model, _WhitespaceTokenizer(), device="cpu", max_length=32, batch_size=3)


def test_conditional_nll_matches_manual_computation(tiny_scorer):
    ctx, cont = "a b c", "d e"
    nll = tiny_scorer.conditional_nll([ctx], [cont])[0]

    tok = tiny_scorer.tokenizer
    ids = tok(ctx)["input_ids"] + tok(cont, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        logits = tiny_scorer.model(input_ids=torch.tensor([ids])).logits[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    # continuation tokens are positions 3 and 4, predicted from 2 and 3
    manual = -(logp[2, ids[3]] + logp[3, ids[4]]).item()
    assert nll == pytest.approx(manual, abs=1e-5)
    assert nll > 0


def test_padding_does_not_change_per_example_scores(tiny_scorer):
    ctxs = ["a b c", "x y z w v u t s", "q"]
    conts = ["d e", "r", "m n o p"]
    batched = tiny_scorer.conditional_nll(ctxs, conts)
    single = np.array([tiny_scorer.conditional_nll([c], [x])[0] for c, x in zip(ctxs, conts)])
    assert np.allclose(batched, single, atol=1e-4)


def test_sequence_nll_skips_first_token(tiny_scorer):
    one = tiny_scorer.sequence_nll(["hello"])[0]
    assert one == 0.0  # nothing to condition on
    two = tiny_scorer.sequence_nll(["hello world"])[0]
    assert two > 0


def test_choice_logprobs_shapes_and_consistency(tiny_scorer):
    lp = tiny_scorer.choice_logprobs("the movie was", ["good", "bad", "very good"])
    assert lp.shape == (3,)
    assert (lp < 0).all()
    batch = tiny_scorer.choice_logprobs_batch(["the movie was", "it is"], ["good", "bad"])
    assert batch.shape == (2, 2)
    assert np.allclose(batch[0], tiny_scorer.choice_logprobs("the movie was", ["good", "bad"]))


def test_left_truncation_keeps_continuation(tiny_scorer):
    long_ctx = " ".join(f"w{i}" for i in range(100))
    nll = tiny_scorer.conditional_nll([long_ctx], ["end token"])[0]
    assert np.isfinite(nll)
    with pytest.raises(ValueError):
        tiny_scorer.conditional_nll(["ctx"], [""])
    with pytest.raises(ValueError):
        tiny_scorer.conditional_nll(["a", "b"], ["c"])


def test_counters(tiny_scorer):
    tiny_scorer.reset_counters()
    tiny_scorer.conditional_nll(["a"] * 7, ["b c"] * 7)
    assert tiny_scorer.num_forward_passes == 3  # batch_size 3 -> 3,3,1
    assert tiny_scorer.num_scored_tokens == 14
