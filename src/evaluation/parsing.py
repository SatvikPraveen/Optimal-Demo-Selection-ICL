"""
Mapping free-form model output onto a task's label set.

Generation-based ICL returns text such as ``" positive\\n\\nReview: ..."`` or
``"The sentiment is Very Positive."``. Comparing that string to the gold
label verbatim (what the first version of this repository did) silently
counts every formatting difference as an error. :func:`parse_prediction`
implements a deterministic, documented decoding rule so that the reported
accuracy reflects the model's decision, and so that parse failures are
counted and reported separately rather than hidden.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

UNKNOWN = "<unparsed>"

_PUNCT_RE = re.compile(r"[^\w\s/]+")
_WS_RE = re.compile(r"\s+")
_LETTER_RE = re.compile(r"^\W*\(?([A-Za-z])\)?(?:[.:)\s]|$)")


def normalize_label(text: str) -> str:
    """Lower-case, strip punctuation (except '/') and collapse whitespace."""
    text = _PUNCT_RE.sub(" ", text.lower())
    return _WS_RE.sub(" ", text).strip()


def parse_prediction(
    text: str,
    label_names: Sequence[str],
    output_prefix: str | None = None,
    first_line_only: bool = True,
) -> str:
    """
    Map ``text`` to one of ``label_names`` or :data:`UNKNOWN`.

    Decoding rule, applied in order:

    1. If ``output_prefix`` (e.g. ``"Sentiment:"``) occurs in the text, the
       text is split at its *first* occurrence. Because ICL prompts end with
       the prefix, the generation normally starts with the answer, so the
       part *before* the prefix is decoded first; if that yields nothing,
       the part after it is used (models that repeat the prefix). Later
       occurrences belong to hallucinated extra examples and are ignored.
    2. Optionally keep only the first non-empty line.
    3. Exact match after :func:`normalize_label`.
    4. If the labels are single letters (multiple choice), match a leading
       letter such as ``"B"``, ``"(b)"`` or ``"C."``.
    5. Otherwise return the *longest* label name that occurs as a whole
       phrase in the normalised text (so ``"very positive"`` beats
       ``"positive"``). Ties are broken by label order.
    6. :data:`UNKNOWN`.
    """
    if text is None:
        return UNKNOWN
    if output_prefix:
        idx = text.find(output_prefix)
        if idx >= 0:
            before, after = text[:idx], text[idx + len(output_prefix) :]
            if before.strip():
                first = parse_prediction(before, label_names, None, first_line_only)
                if first != UNKNOWN:
                    return first
            return parse_prediction(after, label_names, None, first_line_only)

    candidate = text
    if first_line_only:
        lines = [ln for ln in candidate.splitlines() if ln.strip()]
        candidate = lines[0] if lines else ""

    norm = normalize_label(candidate)
    normalized_labels = [normalize_label(lb) for lb in label_names]
    if norm in normalized_labels:
        return label_names[normalized_labels.index(norm)]

    if all(len(lb) == 1 and lb.isalpha() for lb in label_names):
        m = _LETTER_RE.match(candidate)
        if m:
            letter = m.group(1).upper()
            for lb in label_names:
                if lb.upper() == letter:
                    return lb
        return UNKNOWN

    padded = f" {norm} "
    best: str | None = None
    for lb, nlb in zip(label_names, normalized_labels):
        if nlb and f" {nlb} " in padded and (best is None or len(nlb) > len(normalize_label(best))):
            best = lb
    return best if best is not None else UNKNOWN


def parse_predictions(
    texts: Sequence[str], label_names: Sequence[str], output_prefix: str | None = None
) -> list[str]:
    return [parse_prediction(t, label_names, output_prefix) for t in texts]
