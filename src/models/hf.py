"""
Local HuggingFace causal-LM backend (LLaMA, Gemma, GPT-2, GPT-Neo, ...).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseModel
from .scoring import LMScorer


class HFCausalModel(BaseModel):
    """
    Any ``AutoModelForCausalLM`` checkpoint.

    Args:
        model_name: Hub id or local path.
        device: torch device (``None`` = cuda if available, else cpu).
        torch_dtype: e.g. ``torch.float16``; ``None`` keeps the checkpoint's.
        max_length: context window used for left-truncation when scoring.
        use_chat_template: wrap prompts with the tokenizer's chat template
            (for instruction-tuned checkpoints) before generation.
        **model_kwargs: forwarded to ``from_pretrained``.
    """

    supports_scoring = True

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str | None = None,
        torch_dtype: torch.dtype | str | None = None,
        max_length: int | None = None,
        use_chat_template: bool = False,
        **model_kwargs: Any,
    ):
        super().__init__(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, **model_kwargs
        ).to(self.device)
        self.model.eval()
        self.use_chat_template = use_chat_template
        self._last_prompt_tokens: int | None = None
        self._last_completion_tokens: int | None = None

        cfg = self.model.config
        self.max_length = int(
            max_length
            or getattr(cfg, "n_positions", None)
            or getattr(cfg, "max_position_embeddings", 2048)
        )
        self.scorer = LMScorer(
            self.model, self.tokenizer, device=self.device, max_length=self.max_length
        )

    # ------------------------------------------------------------------ #
    def _render(self, prompt: str) -> str:
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        return prompt

    @torch.inference_mode()
    def _generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        rendered = self._render(prompt)
        enc = self.tokenizer(
            rendered, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        n_prompt = int(enc["input_ids"].shape[1])
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        gen_kwargs.update(kwargs)
        out = self.model.generate(**enc, **gen_kwargs)
        new_tokens = out[0, n_prompt:]
        self._last_prompt_tokens = n_prompt
        self._last_completion_tokens = int(new_tokens.shape[0])
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _last_token_counts(self) -> tuple[int | None, int | None]:
        return self._last_prompt_tokens, self._last_completion_tokens

    def _score_choices(self, prompt: str, choices: Sequence[str]) -> list[float]:
        return self.scorer.choice_logprobs(self._render(prompt), choices).tolist()

    def count_tokens(self, text: str) -> int | None:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def info(self) -> dict:
        return {
            **super().info(),
            "device": self.device,
            "dtype": str(next(self.model.parameters()).dtype),
            "max_length": self.max_length,
            "use_chat_template": self.use_chat_template,
        }


class LlamaModel(HFCausalModel):
    """Meta LLaMA checkpoints (defaults to Llama-3.2-3B-Instruct)."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", **kwargs: Any):
        super().__init__(model_name, **kwargs)


class GemmaModel(HFCausalModel):
    """Google Gemma checkpoints (defaults to gemma-2b)."""

    def __init__(self, model_name: str = "google/gemma-2b", **kwargs: Any):
        super().__init__(model_name, **kwargs)
