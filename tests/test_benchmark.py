"""
End-to-end smoke test of the benchmark runner, aggregator and plotter using
the dummy model and mocked datasets / embedding model.
"""

import json
from pathlib import Path

import pytest

from experiments import aggregate_results, run_benchmark
from experiments.run_benchmark import Resources, expand_benchmark, load_yaml, run_single

REPO = Path(__file__).resolve().parent.parent
ALL_METHODS = ["random", "topk", "bm25", "topk_cone", "ids", "rdes", "se2", "influence"]


@pytest.fixture
def smoke_runs():
    cfg = load_yaml(REPO / "configs" / "experiments.yaml")
    return expand_benchmark(cfg, "smoke")


def test_expand_benchmark_applies_defaults_and_overrides(smoke_runs):
    assert [r.method for r in smoke_runs] == ALL_METHODS
    r = {x.method: x for x in smoke_runs}
    assert r["random"].num_test == 12 and r["random"].scorer_model == "self"
    assert r["influence"].method_params["num_subsets"] == 12
    assert r["influence"].method_params["coverage"] == 1  # inherited from methods:
    assert r["se2"].method_params == {"k": 3, "retrieve_k": 8, "beam_size": 2}
    assert r["ids"].method_params["q"] == 3
    assert r["random"].run_id == "sst5__dummy__random__seed0"

    cfg = load_yaml(REPO / "configs" / "experiments.yaml")
    over = expand_benchmark(cfg, "smoke", {"methods": ["topk"], "seeds": [1, 2], "k": 2})
    assert [(x.method, x.seed, x.k) for x in over] == [("topk", 1, 2), ("topk", 2, 2)]
    multi_k = expand_benchmark(cfg, "ablation_k", {"models": ["dummy"]})
    assert {x.k for x in multi_k} == {1, 3, 5, 8}
    assert all(f"k{x.k}" in x.output_dir for x in multi_k)
    with pytest.raises(KeyError):
        expand_benchmark(cfg, "nope")


@pytest.mark.parametrize("method", ALL_METHODS)
def test_run_single_every_method(method, smoke_runs, mock_sentence_transformer, mock_hf_datasets):
    cfg = next(r for r in smoke_runs if r.method == method)
    models_cfg = load_yaml(REPO / "configs" / "models.yaml")["models"]
    result = run_single(cfg, Resources(models_cfg), progress_every=0)

    assert result["run_id"] == cfg.run_id
    assert len(result["records"]) == 12
    m = result["metrics"]
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["accuracy_ci95"]["lower"] <= m["accuracy"] <= m["accuracy_ci95"]["upper"]
    assert m["parse_failure_rate"] == 0.0
    k = cfg.method_params.get("k", cfg.k)
    for rec in result["records"]:
        assert len(rec["demo_indices"]) == k and len(set(rec["demo_indices"])) == k
        assert rec["prediction"] in result["task"]["label_names"]
    assert result["selector"]["name"] == method
    assert result["manifest"]["python"]
    assert result["cost"]["eval_usage"]["scoring_calls"] == 12  # dummy supports scoring
    if method == "influence":
        assert result["diagnostics"]["influence_coverage"]["never_sampled"] == 0
        assert result["cost"]["fit_usage"]["scoring_calls"] == 12 * 8  # subsets x val
        # query-independent: identical demos for every query
        assert len({tuple(r["demo_indices"]) for r in result["records"]}) == 1
    if method == "ids":
        assert "ids_majority_vote_accuracy" in m
        assert result["cost"]["eval_usage"]["calls"] == 12 * 4  # 1 CoT + q=3 ICL calls
    if method in ("topk_cone", "se2"):
        assert result["cost"]["scorer_forward_passes"] > 0
    # JSON-serialisable
    json.dumps(result, default=run_benchmark._json_default)


def test_cli_end_to_end_and_aggregation(
    tmp_path, mock_sentence_transformer, mock_hf_datasets, capsys
):
    out = tmp_path / "raw"
    rc = run_benchmark.main(
        [
            "--benchmark",
            "smoke",
            "--methods",
            "random",
            "topk",
            "--seeds",
            "0",
            "1",
            "--output-dir",
            str(out),
            "--num-test",
            "10",
        ]
    )
    assert rc == 0
    files = sorted(p.name for p in out.glob("*.json"))
    assert files == [
        "sst5__dummy__random__seed0.json",
        "sst5__dummy__random__seed1.json",
        "sst5__dummy__topk__seed0.json",
        "sst5__dummy__topk__seed1.json",
    ]
    # --resume skips everything that exists
    rc = run_benchmark.main(
        [
            "--benchmark",
            "smoke",
            "--methods",
            "random",
            "--seeds",
            "0",
            "--output-dir",
            str(out),
            "--resume",
        ]
    )
    assert rc == 0

    processed = tmp_path / "processed"
    rc = aggregate_results.main(
        ["--raw-dir", str(out), "--output-dir", str(processed), "--baseline", "random"]
    )
    assert rc == 0
    assert (processed / "runs.csv").exists() and (processed / "summary.md").exists()
    md = (processed / "summary.md").read_text()
    assert "| topk |" in md and "± " in md  # two seeds -> std shown
    assert "Paired significance vs `random`" in md
    sig = (processed / "significance.csv").read_text()
    assert "topk" in sig

    # Plotting (matplotlib is in the dev extras).
    pytest.importorskip("matplotlib")
    from experiments import plot_results

    plots = tmp_path / "plots"
    assert plot_results.main(["--processed-dir", str(processed), "--output-dir", str(plots)]) == 0
    assert (plots / "cost_vs_accuracy.png").exists()
    assert any(p.name.startswith("accuracy_dummy") for p in plots.glob("*.png"))


def test_aggregate_with_no_results(tmp_path):
    assert (
        aggregate_results.main(["--raw-dir", str(tmp_path), "--output-dir", str(tmp_path / "p")])
        == 1
    )
