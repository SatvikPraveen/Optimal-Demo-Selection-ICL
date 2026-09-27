"""
Base model interface.

Every model backend exposes free-form :meth:`BaseModel.generate` and,
where the backend gives access to token log-probabilities, constrained
:meth:`BaseModel.score_choices`. A :class:`Usage` counter records calls,
characters and (when known) tokens so that the benchmark can report the
inference cost of each selection method.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field


@dataclass
class Usage:
    """Cumulative usage counters for one model instance."""

    calls: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    scoring_calls: int = 0
    scored_choices: int = 0
    wall_time_s: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def record_generation(
        self,
        prompt: str,
        completion: str,
        elapsed: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        self.calls += 1
        self.prompt_chars += len(prompt)
        self.completion_chars += len(completion or "")
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        self.wall_time_s += elapsed

    def record_scoring(
        self, prompt: str, num_choices: int, elapsed: float, prompt_tokens: int | None = None
    ) -> None:
        self.scoring_calls += 1
        self.scored_choices += num_choices
        self.prompt_chars += len(prompt) * num_choices
        self.prompt_tokens += int(prompt_tokens or 0)
        self.wall_time_s += elapsed

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def to_dict(self) -> dict:
        return asdict(self)


class BaseModel(ABC):
    """Abstract base class for language models."""

    #: Whether :meth:`score_choices` is available for this backend.
    supports_scoring: bool = False

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.usage = Usage()

    # ------------------------------------------------------------------ #
    @abstractmethod
    def _generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Backend-specific generation (no bookkeeping)."""

    def generate(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.0, **kwargs
    ) -> str:
        """Generate text from ``prompt`` and record usage."""
        start = time.perf_counter()
        out = self._generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
        elapsed = time.perf_counter() - start
        tokens = self._last_token_counts()
        self.usage.record_generation(prompt, out, elapsed, *tokens)
        return out

    def batch_generate(
        self, prompts: Sequence[str], max_tokens: int = 100, temperature: float = 0.0, **kwargs
    ) -> list[str]:
        return [self.generate(p, max_tokens, temperature, **kwargs) for p in prompts]

    # ------------------------------------------------------------------ #
    def _score_choices(self, prompt: str, choices: Sequence[str]) -> list[float]:
        raise NotImplementedError(f"{type(self).__name__} does not support choice scoring")

    def score_choices(self, prompt: str, choices: Sequence[str]) -> list[float]:
        """
        ``log p(choice | prompt)`` for each continuation string. Use
        ``argmax`` for constrained classification.
        """
        start = time.perf_counter()
        scores = self._score_choices(prompt, choices)
        self.usage.record_scoring(prompt, len(choices), time.perf_counter() - start)
        return scores

    # ------------------------------------------------------------------ #
    def _last_token_counts(self) -> tuple[int | None, int | None]:
        """(prompt_tokens, completion_tokens) of the last call, if known."""
        return None, None

    def count_tokens(self, text: str) -> int | None:
        """Token count of ``text`` under this model's tokenizer, if available."""
        return None

    def info(self) -> dict:
        return {"model_name": self.model_name, "backend": type(self).__name__}
