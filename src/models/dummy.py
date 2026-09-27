"""
Deterministic stand-in model for smoke tests and CI.

``DummyModel`` needs no weights, network or GPU. Its behaviour is simple
enough to reason about in tests yet exercises every code path of the
benchmark runner: generation returns a label chosen by a stable hash of the
prompt (optionally biased towards the label that appears most often among
the demonstrations, which makes retrieval-based selectors measurably
better than random), and choice scoring returns log-probabilities with the
same preference.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np

from .base import BaseModel


class DummyScorer:
    """
    Deterministic stand-in for :class:`src.models.scoring.LMScorer`, so that
    LM-scored selectors (TopK+CoNE, Se²) can run against ``DummyModel``.
    The NLL is a stable hash of (context, continuation).
    """

    def __init__(self):
        self.model = None
        self.num_forward_passes = 0
        self.num_scored_tokens = 0

    def conditional_nll(self, contexts, continuations):
        self.num_forward_passes += 1
        out = []
        for c, x in zip(contexts, continuations):
            h = zlib.crc32((c + "\x00" + x).encode("utf-8")) & 0xFFFFFFFF
            out.append(1.0 + (h % 1000) / 100.0)
            self.num_scored_tokens += len(x.split())
        return np.asarray(out, dtype=float)

    def sequence_nll(self, texts):
        return self.conditional_nll([""] * len(texts), texts)

    def choice_logprobs(self, prompt, choices):
        return -self.conditional_nll([prompt] * len(choices), list(choices))

    def choice_logprobs_batch(self, prompts, choices):
        return np.stack([self.choice_logprobs(p, choices) for p in prompts])

    def reset_counters(self):
        self.num_forward_passes = 0
        self.num_scored_tokens = 0


class DummyModel(BaseModel):
    supports_scoring = True

    def __init__(
        self,
        model_name: str = "dummy",
        label_names: Sequence[str] | None = None,
        output_prefix: str = "",
        copy_majority_demo_label: bool = True,
        **kwargs,
    ):
        super().__init__(model_name)
        self.label_names = list(label_names or [])
        self.output_prefix = output_prefix
        self.copy_majority_demo_label = copy_majority_demo_label
        self.scorer = DummyScorer()

    # ------------------------------------------------------------------ #
    def _majority_demo_label(self, prompt: str) -> str | None:
        if not (self.copy_majority_demo_label and self.output_prefix and self.label_names):
            return None
        counts: dict[str, int] = {}
        for line in prompt.splitlines():
            if line.startswith(self.output_prefix):
                answer = line[len(self.output_prefix) :].strip()
                if answer in self.label_names:
                    counts[answer] = counts.get(answer, 0) + 1
        if not counts:
            return None
        return max(sorted(counts), key=lambda k: counts[k])

    def _hash_label(self, prompt: str) -> str:
        if not self.label_names:
            return "dummy output"
        h = zlib.crc32(prompt.encode("utf-8")) & 0xFFFFFFFF
        return self.label_names[h % len(self.label_names)]

    def _generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        label = self._majority_demo_label(prompt) or self._hash_label(prompt)
        self._last = (len(prompt.split()), len(label.split()))
        return f" {label}\n"

    def _last_token_counts(self) -> tuple[int | None, int | None]:
        return getattr(self, "_last", (None, None))

    def _score_choices(self, prompt: str, choices: Sequence[str]) -> list[float]:
        preferred = self._majority_demo_label(prompt) or self._hash_label(prompt)
        scores = []
        for c in choices:
            base = -2.0 - (zlib.crc32((prompt + c).encode()) % 100) / 100.0
            if c.strip() == preferred:
                base += 1.5
            scores.append(base)
        return list(np.asarray(scores, dtype=float))

    def count_tokens(self, text: str) -> int | None:
        return len(text.split())
