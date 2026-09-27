"""
OpenAI chat-completions backend.
"""

from __future__ import annotations

import os
from typing import Any

from .base import BaseModel


class GPTModel(BaseModel):
    """
    OpenAI GPT models via the chat-completions API.

    The API does not expose log-probabilities of arbitrary continuations,
    so :attr:`supports_scoring` is ``False`` and classification falls back
    to generation + :func:`src.evaluation.parsing.parse_prediction`.
    """

    supports_scoring = False

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        system_prompt: str | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        from openai import OpenAI  # imported lazily so the package is optional

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key, max_retries=max_retries)
        self.system_prompt = system_prompt
        self._last_prompt_tokens: int | None = None
        self._last_completion_tokens: int | None = None

    def _generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        usage = getattr(response, "usage", None)
        self._last_prompt_tokens = getattr(usage, "prompt_tokens", None)
        self._last_completion_tokens = getattr(usage, "completion_tokens", None)
        return response.choices[0].message.content or ""

    def _last_token_counts(self) -> tuple[int | None, int | None]:
        return self._last_prompt_tokens, self._last_completion_tokens
