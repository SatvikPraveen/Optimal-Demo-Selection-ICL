"""
Task definitions: prompt templates, label verbalizers and data loading.

A :class:`Task` is the single source of truth for how a dataset is turned
into text. Every selector sees demonstrations formatted with
:meth:`Task.format_demo` and queries formatted with :meth:`Task.format_query`,
and every model prediction is mapped back onto :attr:`Task.label_names`
with :func:`src.evaluation.parsing.parse_prediction`. Keeping this in one
place is what makes results comparable across methods and models.

The registry (:data:`TASKS`, :func:`get_task`, :func:`load_split`) is
what the benchmark runner uses; the individual ``load_*`` functions in the
sibling modules remain available for direct use.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .load_agnews import load_agnews
from .load_csqa import load_commonsense_qa
from .load_sst5 import load_sst5

Loader = Callable[..., tuple[list[str], list[str]]]


@dataclass(frozen=True)
class Task:
    """
    Args:
        name: registry key (``sst5``, ``agnews``, ``csqa``).
        label_names: the verbalizers, in canonical order. Gold labels and
            parsed predictions are always drawn from this list.
        instruction: optional task instruction placed at the top of a prompt.
        input_prefix / output_prefix: template pieces, e.g. ``"Text: "`` and
            ``"Sentiment:"``. A demonstration is
            ``f"{input_prefix}{text}\\n{output_prefix} {label}"`` and a query
            is the same string without the label.
        splits: mapping from the logical split names ``train`` / ``validation``
            / ``test`` to the split names understood by the loader. A logical
            split mapped to ``None`` does not exist upstream and has to be
            carved out of ``train`` (see :func:`load_split`).
        loader: ``loader(split, num_samples, seed) -> (texts, labels)``.
        description: one-line human description for reports.
    """

    name: str
    label_names: tuple[str, ...]
    instruction: str
    input_prefix: str
    output_prefix: str
    splits: dict[str, str | None]
    loader: Loader = field(repr=False, compare=False)
    description: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    def format_query(self, text: str) -> str:
        return f"{self.input_prefix}{text}\n{self.output_prefix}"

    def format_demo(self, text: str, label: str) -> str:
        return f"{self.format_query(text)} {label}"

    def format_demos(self, texts: Sequence[str], labels: Sequence[str]) -> list[str]:
        return [self.format_demo(t, y) for t, y in zip(texts, labels)]

    def choices(self) -> list[str]:
        """Continuation strings scored for constrained classification."""
        return [f" {name}" for name in self.label_names]

    def label_index(self, label: str) -> int:
        return self.label_names.index(label)


def _csqa_loader(split: str, num_samples: int | None, seed: int) -> tuple[list[str], list[str]]:
    return load_commonsense_qa(split=split, num_samples=num_samples, seed=seed)


TASKS: dict[str, Task] = {
    "sst5": Task(
        name="sst5",
        label_names=("very negative", "negative", "neutral", "positive", "very positive"),
        instruction="Classify the sentiment of each movie review.",
        input_prefix="Review: ",
        output_prefix="Sentiment:",
        splits={"train": "train", "validation": "validation", "test": "test"},
        loader=lambda split, num_samples, seed: load_sst5(split, num_samples, seed),
        description="SST-5 five-way sentiment classification (SetFit/sst5).",
    ),
    "agnews": Task(
        name="agnews",
        label_names=("World", "Sports", "Business", "Technology"),
        instruction="Classify the topic of each news article.",
        input_prefix="Article: ",
        output_prefix="Topic:",
        splits={"train": "train", "validation": None, "test": "test"},
        loader=lambda split, num_samples, seed: load_agnews(split, num_samples, seed),
        description="AG News four-way topic classification.",
    ),
    "csqa": Task(
        name="csqa",
        label_names=("A", "B", "C", "D", "E"),
        instruction="Answer each multiple-choice commonsense question with the letter of the correct option.",
        input_prefix="",
        output_prefix="Answer:",
        # The official CSQA test split is unlabeled, so validation serves as
        # test and a validation slice is carved out of train.
        splits={"train": "train", "validation": None, "test": "validation"},
        loader=_csqa_loader,
        description="CommonsenseQA five-way multiple choice (answer letter).",
    ),
}

# Backwards-compatible aliases.
TASK_ALIASES = {"commonsense_qa": "csqa", "ag_news": "agnews", "sst-5": "sst5"}


def get_task(name: str) -> Task:
    key = TASK_ALIASES.get(name.lower(), name.lower())
    if key not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(TASKS)}")
    return TASKS[key]


def load_split(
    task: Task | str,
    split: str,
    num_samples: int | None = None,
    seed: int = 42,
    holdout_fraction: float = 0.1,
) -> tuple[list[str], list[str]]:
    """
    Load a logical split (``train`` / ``validation`` / ``test``) for a task.

    When the task has no upstream validation split, ``validation`` is
    carved deterministically (by ``seed``) from the *end* of a shuffled
    train sample and ``train`` returns the complementary prefix, so the two
    never overlap for the same ``seed``. ``num_samples`` bounds the size of
    the returned split.
    """
    task = get_task(task) if isinstance(task, str) else task
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be 'train', 'validation' or 'test'")

    upstream = task.splits[split]
    if upstream is not None:
        return task.loader(upstream, num_samples, seed)

    # Carve validation out of train.
    if split != "validation":
        raise ValueError(f"Task {task.name} has no '{split}' split")
    train_upstream = task.splits["train"]
    assert train_upstream is not None
    texts, labels = task.loader(train_upstream, None, seed)
    n_val = max(1, int(len(texts) * holdout_fraction))
    val_texts, val_labels = texts[-n_val:], labels[-n_val:]
    if num_samples is not None:
        val_texts, val_labels = val_texts[:num_samples], val_labels[:num_samples]
    return list(val_texts), list(val_labels)


def load_train_without_holdout(
    task: Task | str,
    num_samples: int | None = None,
    seed: int = 42,
    holdout_fraction: float = 0.1,
) -> tuple[list[str], list[str]]:
    """
    Training pool that is guaranteed disjoint from :func:`load_split`'s carved
    validation slice (a no-op restriction for tasks with a real validation
    split).
    """
    task = get_task(task) if isinstance(task, str) else task
    train_upstream = task.splits["train"]
    assert train_upstream is not None
    if task.splits["validation"] is not None:
        return task.loader(train_upstream, num_samples, seed)
    texts, labels = task.loader(train_upstream, None, seed)
    n_val = max(1, int(len(texts) * holdout_fraction))
    texts, labels = texts[:-n_val], labels[:-n_val]
    if num_samples is not None:
        texts, labels = texts[:num_samples], labels[:num_samples]
    return list(texts), list(labels)
