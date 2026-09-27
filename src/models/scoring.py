"""
Language-model scoring utilities.

Several selection methods (TopK+CoNE, Se², label scoring for classification)
need *per-example* log-likelihoods from a causal LM. The archived notebooks
computed these with ``model(ids, labels=ids).loss * length``, but
``outputs.loss`` is a single scalar averaged over *every token in the batch*
(padding included), so every "per-example" number in a batch was the same
mean rescaled by that example's length. :class:`LMScorer` computes the real
quantity: the sum of token log-probabilities of a continuation given a
context, with padding and context tokens masked out.

Everything is batched and runs under ``torch.inference_mode``. Contexts that
exceed ``max_length`` are truncated from the *left* so the query (which is
always at the end of an ICL prompt) is never cut off.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LMScorer:
    """
    Per-example conditional log-likelihood scoring with a causal LM.

    Args:
        model: a loaded ``transformers`` causal LM (already on ``device``).
        tokenizer: its tokenizer.
        device: torch device string; inferred from the model if omitted.
        max_length: maximum total sequence length (context + continuation).
        batch_size: number of (context, continuation) pairs per forward pass.

    Use :meth:`from_pretrained` to construct one from a Hub checkpoint.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str | None = None,
        max_length: int = 1024,
        batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        if device is None:
            try:
                device = str(next(model.parameters()).device)
            except (StopIteration, AttributeError):
                device = "cpu"
        self.device = device
        self.model.eval()
        self._pad_id = self._resolve_pad_id()
        self.num_forward_passes = 0
        self.num_scored_tokens = 0

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "gpt2",
        device: str | None = None,
        max_length: int | None = None,
        batch_size: int = 8,
        torch_dtype: torch.dtype | None = None,
        **model_kwargs: Any,
    ) -> LMScorer:
        """Load ``model_name`` from the HuggingFace Hub."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, **model_kwargs
        ).to(device)
        if max_length is None:
            cfg = model.config
            max_length = int(
                getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", 1024)
            )
        return cls(model, tokenizer, device=device, max_length=max_length, batch_size=batch_size)

    # ------------------------------------------------------------------ #
    # Tokenisation helpers
    # ------------------------------------------------------------------ #
    def _resolve_pad_id(self) -> int:
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tokenizer, "eos_token_id", None)
        return 0 if pad is None else int(pad)

    def _encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])

    def _build_example(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """
        Return ``(input_ids, labels)`` where ``labels`` is ``-100`` on context
        and padding positions and the token id on continuation positions.
        """
        ctx_ids = self._encode(context, add_special_tokens=True) if context else []
        cont_ids = self._encode(continuation, add_special_tokens=False)
        if not cont_ids:
            raise ValueError("continuation must contain at least one token")

        # Left-truncate the context so the continuation always fits.
        budget = self.max_length - len(cont_ids)
        if budget < 1:
            # Pathological: continuation alone exceeds max_length. Keep its tail.
            cont_ids = cont_ids[-(self.max_length - 1) :]
            budget = 1
        if len(ctx_ids) > budget:
            ctx_ids = ctx_ids[-budget:]

        ids = ctx_ids + cont_ids
        labels = [-100] * len(ctx_ids) + cont_ids
        if not ctx_ids:
            # No context: the first token has nothing to condition on, so it
            # cannot be scored by a causal LM.
            labels[0] = -100
        return ids, labels

    # ------------------------------------------------------------------ #
    # Core scoring
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _score_batch(self, examples: list[tuple[list[int], list[int]]]) -> np.ndarray:
        max_len = max(len(ids) for ids, _ in examples)
        input_ids = torch.full((len(examples), max_len), self._pad_id, dtype=torch.long)
        attention = torch.zeros((len(examples), max_len), dtype=torch.long)
        labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
        for i, (ids, lab) in enumerate(examples):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attention[i, : len(ids)] = 1
            labels[i, : len(lab)] = torch.tensor(lab)

        input_ids = input_ids.to(self.device)
        attention = attention.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention).logits
        # Predict token t from position t-1.
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100
        gathered = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        token_ll = (gathered * mask).sum(dim=1)

        self.num_forward_passes += 1
        self.num_scored_tokens += int(mask.sum().item())
        return (-token_ll).cpu().numpy().astype(np.float64)

    def conditional_nll(self, contexts: Sequence[str], continuations: Sequence[str]) -> np.ndarray:
        """
        Negative log-likelihood (nats, summed over tokens) of each
        ``continuation`` given its ``context``. Only continuation tokens are
        scored. Returns an array of shape ``(len(contexts),)``.
        """
        if len(contexts) != len(continuations):
            raise ValueError("contexts and continuations must have the same length")
        if len(contexts) == 0:
            return np.zeros(0)
        examples = [self._build_example(c, x) for c, x in zip(contexts, continuations)]
        out = np.empty(len(examples))
        for start in range(0, len(examples), self.batch_size):
            chunk = examples[start : start + self.batch_size]
            out[start : start + len(chunk)] = self._score_batch(chunk)
        return out

    def sequence_nll(self, texts: Sequence[str]) -> np.ndarray:
        """Unconditional NLL of each text (first token unscored)."""
        return self.conditional_nll([""] * len(texts), texts)

    def choice_logprobs(self, prompt: str, choices: Sequence[str]) -> np.ndarray:
        """
        ``log p(choice | prompt)`` for each choice string (summed over the
        choice's tokens). Use ``argmax`` for constrained classification.
        """
        return -self.conditional_nll([prompt] * len(choices), list(choices))

    def choice_logprobs_batch(self, prompts: Sequence[str], choices: Sequence[str]) -> np.ndarray:
        """Vectorised :meth:`choice_logprobs`; returns shape ``(n_prompts, n_choices)``."""
        ctx = [p for p in prompts for _ in choices]
        cont = [c for _ in prompts for c in choices]
        nll = self.conditional_nll(ctx, cont)
        return -nll.reshape(len(prompts), len(choices))

    def reset_counters(self) -> None:
        self.num_forward_passes = 0
        self.num_scored_tokens = 0
