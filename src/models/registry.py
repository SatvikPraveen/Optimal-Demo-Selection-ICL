"""
Build model backends from ``configs/models.yaml`` entries.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import BaseModel
from .dummy import DummyModel
from .gpt import GPTModel
from .hf import HFCausalModel

MODEL_TYPES = {"openai", "huggingface", "dummy"}


def build_model(
    name: str,
    models_config: dict[str, dict[str, Any]],
    label_names: Sequence[str] | None = None,
    output_prefix: str = "",
    overrides: dict[str, Any] | None = None,
) -> BaseModel:
    """
    Instantiate the model registered under ``name``.

    ``models_config`` is the ``models:`` mapping of ``configs/models.yaml``.
    Each entry has a ``type`` (``openai`` / ``huggingface`` / ``dummy``), a
    ``model_name`` and an optional ``kwargs`` mapping forwarded to the
    backend. Unknown names whose ``type`` can be guessed (``dummy``,
    ``gpt-*``) are accepted as a convenience.
    """
    entry = dict(models_config.get(name) or {})
    if not entry:
        if name == "dummy":
            entry = {"type": "dummy", "model_name": "dummy"}
        elif name.startswith("gpt-"):
            entry = {"type": "openai", "model_name": name}
        else:
            entry = {"type": "huggingface", "model_name": name}
    mtype = entry.get("type", "huggingface")
    if mtype not in MODEL_TYPES:
        raise ValueError(
            f"Unknown model type '{mtype}' for '{name}' (expected {sorted(MODEL_TYPES)})"
        )
    model_name = entry.get("model_name", name)
    kwargs = dict(entry.get("kwargs") or {})
    kwargs.update(overrides or {})

    if mtype == "dummy":
        return DummyModel(
            model_name, label_names=label_names, output_prefix=output_prefix, **kwargs
        )
    if mtype == "openai":
        return GPTModel(model_name, **kwargs)
    return HFCausalModel(model_name, **kwargs)
