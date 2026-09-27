"""
Tests for the task registry, prompt templates, prediction parsing, the
dummy model and ICLInference.predict.
"""

import pytest

from src.datasets import TASKS, get_task, load_split, load_train_without_holdout
from src.evaluation.parsing import UNKNOWN, normalize_label, parse_prediction
from src.models import DummyModel
from src.prompting import ICLInference, PromptBuilder


def test_task_templates():
    sst5 = get_task("sst5")
    assert sst5.format_query("great") == "Review: great\nSentiment:"
    assert sst5.format_demo("great", "positive") == "Review: great\nSentiment: positive"
    assert sst5.choices() == [f" {lb}" for lb in sst5.label_names]
    assert sst5.num_classes == 5 and sst5.label_index("neutral") == 2
    assert get_task("commonsense_qa") is TASKS["csqa"]
    with pytest.raises(KeyError):
        get_task("nope")


def test_load_split_and_holdout(mock_hf_datasets):
    # AG News has no validation split: it is carved from the end of train and
    # the training pool excludes it.
    val_t, val_l = load_split("agnews", "validation", seed=1)
    train_t, train_l = load_train_without_holdout("agnews", seed=1)
    assert len(val_t) == 10 and len(train_t) == 90
    assert not set(val_t) & set(train_t)
    assert len(load_split("agnews", "validation", num_samples=3)[0]) == 3
    with pytest.raises(ValueError):
        load_split("agnews", "dev")

    # CSQA labels are answer letters; its 'test' is the upstream validation split.
    t, y = load_split("csqa", "test", num_samples=10)
    assert len(t) == 10 and set(y) <= set("ABCDE")
    assert t[0].startswith("Question:")

    # SST-5 has a real validation split.
    assert len(load_split("sst5", "validation", num_samples=7)[0]) == 7
    assert len(load_train_without_holdout("sst5", num_samples=7)[0]) == 7


@pytest.mark.parametrize(
    "text,expected",
    [
        (" positive\n\nReview: more", "positive"),
        ("Very Positive.", "very positive"),
        ("The sentiment is very negative", "very negative"),
        ("I think it's Positive, mostly", "positive"),
        ("Sentiment: neutral", "neutral"),
        ("nothing here", UNKNOWN),
        ("", UNKNOWN),
        (None, UNKNOWN),
    ],
)
def test_parse_prediction_classification(text, expected):
    labels = get_task("sst5").label_names
    assert parse_prediction(text, labels, output_prefix="Sentiment:") == expected


@pytest.mark.parametrize(
    "text,expected",
    [(" B", "B"), ("(c)", "C"), ("D. because", "D"), ("Answer: E", "E"), ("banana", UNKNOWN)],
)
def test_parse_prediction_letters(text, expected):
    assert parse_prediction(text, list("ABCDE"), output_prefix="Answer:") == expected


def test_parse_prediction_uses_first_answer_and_longest_label():
    labels = ["World", "Sports", "Business", "Technology"]
    text = "Sports\nArticle: foo\nTopic: World"
    assert parse_prediction(text, labels, output_prefix="Topic:") == "Sports"
    assert parse_prediction("Topic: Business\nTopic: World", labels, "Topic:") == "Business"
    assert normalize_label("  Sci/Tech!! ") == "sci/tech"


def test_dummy_model_and_predict_modes():
    task = get_task("sst5")
    model = DummyModel(label_names=task.label_names, output_prefix=task.output_prefix)
    demos = task.format_demos(["a", "b", "c"], ["negative", "negative", "positive"])
    query = task.format_query("d")

    gen = ICLInference(model, PromptBuilder(task.instruction), prediction_mode="generate")
    out = gen.predict(demos, query, task)
    assert out["mode"] == "generate" and out["prediction"] == "negative"
    assert task.instruction in out["prompt"] and out["prompt"].endswith(query)

    sc = ICLInference(model, PromptBuilder(task.instruction), prediction_mode="auto")
    out2 = sc.predict(demos, query, task)
    assert out2["mode"] == "score" and out2["prediction"] == "negative"
    assert len(out2["raw"]) == 5

    # Usage counters were updated by both paths.
    assert model.usage.calls == 1 and model.usage.scoring_calls == 1
    assert model.usage.prompt_tokens > 0 and model.usage.wall_time_s >= 0
    assert model.count_tokens("a b c") == 3

    # No demos: falls back to a stable hash label.
    assert gen.predict([], query, task)["prediction"] in task.label_names

    m = DummyModel()
    m.supports_scoring = False
    with pytest.raises(RuntimeError):
        _ = ICLInference(m, prediction_mode="score").effective_mode
    with pytest.raises(ValueError):
        ICLInference(m, prediction_mode="magic")
