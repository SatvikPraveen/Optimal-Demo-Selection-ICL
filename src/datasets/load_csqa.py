"""
CommonsenseQA dataset loader.
"""

from __future__ import annotations

from datasets import load_dataset


def load_commonsense_qa(
    split: str = "train",
    num_samples: int | None = None,
    seed: int = 42,
    answer_format: str = "letter",
) -> tuple[list[str], list[str]]:
    """
    Load CommonsenseQA.

    Args:
        split: ``'train'`` or ``'validation'`` (the official test split has
            no labels).
        num_samples: number of samples to keep after shuffling (``None`` = all).
        seed: shuffle seed.
        answer_format: ``'letter'`` returns the answer key (``A``-``E``),
            which is how the benchmark scores the task; ``'text'`` returns
            the answer option's text.

    Returns:
        texts: question followed by the lettered options, one per line.
        labels: the gold answers.
    """
    if answer_format not in ("letter", "text"):
        raise ValueError("answer_format must be 'letter' or 'text'")

    ds = load_dataset("tau/commonsense_qa", split=split)
    ds = ds.shuffle(seed=seed)
    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    texts: list[str] = []
    labels: list[str] = []
    for q, choice_dict, key in zip(ds["question"], ds["choices"], ds["answerKey"]):
        options = "\n".join(
            f"{lbl}. {txt}" for lbl, txt in zip(choice_dict["label"], choice_dict["text"])
        )
        texts.append(f"Question: {q}\n{options}")
        if answer_format == "letter":
            labels.append(key)
        else:
            labels.append(choice_dict["text"][choice_dict["label"].index(key)])
    return texts, labels


def format_csqa_prompt(question: str, choices: dict, answer: str | None = None) -> str:
    """Format a CommonsenseQA sample as a prompt (legacy helper)."""
    prompt = f"Question: {question}\n"
    prompt += "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(choices["label"], choices["text"]))
    prompt += "\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt
