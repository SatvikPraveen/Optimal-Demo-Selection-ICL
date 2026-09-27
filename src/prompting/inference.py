"""
ICL inference utilities.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..datasets.tasks import Task
from ..evaluation.parsing import parse_prediction
from ..models import BaseModel
from .prompt_builder import PromptBuilder


class ICLInference:
    """
    In-context learning inference engine.

    Args:
        model: language model backend.
        prompt_builder: assembles instruction + demonstrations + query.
        prediction_mode: ``"score"`` ranks the task's label verbalizers by
            log-probability (only for backends with ``supports_scoring``),
            ``"generate"`` decodes text and parses it, ``"auto"`` picks
            ``"score"`` when available.
    """

    def __init__(
        self,
        model: BaseModel,
        prompt_builder: PromptBuilder | None = None,
        prediction_mode: str = "auto",
    ):
        if prediction_mode not in ("auto", "score", "generate"):
            raise ValueError("prediction_mode must be 'auto', 'score' or 'generate'")
        self.model = model
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.prediction_mode = prediction_mode

    @property
    def effective_mode(self) -> str:
        if self.prediction_mode == "auto":
            return "score" if self.model.supports_scoring else "generate"
        if self.prediction_mode == "score" and not self.model.supports_scoring:
            raise RuntimeError(f"{type(self.model).__name__} does not support scoring")
        return self.prediction_mode

    # ------------------------------------------------------------------ #
    def build_prompt(self, demonstrations: Sequence[str], query: str) -> str:
        return self.prompt_builder.build_prompt(list(demonstrations), query)

    def run_icl(
        self,
        demonstrations: Sequence[str],
        query: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Free-form ICL generation (raw model output)."""
        prompt = self.build_prompt(demonstrations, query)
        return self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)

    def run_zero_shot_cot(
        self, query: str, max_tokens: int = 200, temperature: float = 0.0, **kwargs
    ) -> str:
        """Zero-shot chain-of-thought (used by IDS)."""
        prompt = self.prompt_builder.build_zero_shot_cot_prompt(query)
        return self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)

    # ------------------------------------------------------------------ #
    def predict(
        self,
        demonstrations: Sequence[str],
        query: str,
        task: Task,
        max_tokens: int = 10,
    ) -> dict:
        """
        Predict a label from ``task.label_names`` for ``query``.

        Returns a dict with ``prediction`` (a label name or
        :data:`UNKNOWN`), ``raw`` (generated text or per-label log-probs),
        ``prompt`` and ``mode``.
        """
        prompt = self.build_prompt(demonstrations, query)
        mode = self.effective_mode
        if mode == "score":
            scores = self.model.score_choices(prompt, task.choices())
            idx = int(np.argmax(scores))
            return {
                "prediction": task.label_names[idx],
                "raw": [float(s) for s in scores],
                "prompt": prompt,
                "mode": mode,
            }
        text = self.model.generate(prompt, max_tokens=max_tokens, temperature=0.0)
        pred = parse_prediction(text, task.label_names, output_prefix=task.output_prefix or None)
        return {"prediction": pred, "raw": text, "prompt": prompt, "mode": mode}
