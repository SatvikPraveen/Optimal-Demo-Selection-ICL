#!/usr/bin/env python
"""
Config-driven benchmark runner.

Runs every (dataset x model x method x seed) combination of a benchmark
defined in ``configs/experiments.yaml`` and writes one JSON file per run to
``results/raw/``. Each file contains the per-example records (query,
selected demonstration indices, prediction, gold label), the metrics with
bootstrap confidence intervals, cost counters and an environment manifest.

Examples::

    # 8 methods x dummy model on 12 SST-5 queries: runs in seconds, no GPU
    python experiments/run_benchmark.py --benchmark smoke

    # a real run, overriding parts of the config from the command line
    python experiments/run_benchmark.py --benchmark quick_test \\
        --models gpt2 --methods random topk topk_cone --seeds 0 1 2

    # skip runs whose result file already exists
    python experiments/run_benchmark.py --benchmark full_benchmark --resume

Aggregate the outputs with ``experiments/aggregate_results.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.datasets import Task, get_task, load_split, load_train_without_holdout
from src.evaluation import UNKNOWN, bootstrap_ci, compute_metrics
from src.models import BaseModel, LMScorer
from src.models.registry import build_model
from src.prompting import ICLInference, PromptBuilder
from src.selection import build_selector
from src.selection.registry import NEEDS_INFERENCE, NEEDS_SCORER, NEEDS_VALIDATION
from src.utils import Embedder, set_seed, setup_logger
from src.utils.manifest import build_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments.yaml"
DEFAULT_MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"

log = logging.getLogger("icl.benchmark")


# ---------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------- #
@dataclass
class RunConfig:
    dataset: str
    model: str
    method: str
    seed: int
    k: int = 5
    num_train: int = 1000
    num_val: int = 200
    num_test: int = 200
    prediction_mode: str = "auto"
    embedding_model: str = "all-MiniLM-L6-v2"
    scorer_model: str = "gpt2"
    max_new_tokens: int = 10
    method_params: dict[str, Any] = field(default_factory=dict)
    model_params: dict[str, Any] = field(default_factory=dict)
    output_dir: str = "results/raw"
    benchmark: str | None = None

    @property
    def run_id(self) -> str:
        return f"{self.dataset}__{self.model}__{self.method}__seed{self.seed}"

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / f"{self.run_id}.json"


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def expand_benchmark(
    config: dict, benchmark: str, overrides: dict[str, Any] | None = None
) -> list[RunConfig]:
    """Turn a benchmark block of ``experiments.yaml`` into concrete runs."""
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    defaults = dict(config.get("defaults") or {})
    bench = dict((config.get("benchmarks") or {}).get(benchmark) or {})
    if not bench:
        raise KeyError(
            f"benchmark '{benchmark}' not in config (have {sorted(config.get('benchmarks') or {})})"
        )
    method_defaults = dict(config.get("methods") or {})

    merged = {**defaults, **bench, **overrides}
    seeds = merged.get("seeds") or [merged.get("seed", 42)]
    k_values = merged.get("k_values") or [merged.get("k", 5)]
    runs: list[RunConfig] = []
    for dataset in merged["datasets"]:
        for model in merged["models"]:
            for method in merged["methods"]:
                params = dict(method_defaults.get(method) or {})
                params.update((bench.get("method_params") or {}).get(method) or {})
                params.update((overrides.get("method_params") or {}).get(method) or {})
                for k in k_values:
                    for seed in seeds:
                        run = RunConfig(
                            dataset=dataset,
                            model=model,
                            method=method,
                            seed=int(seed),
                            k=int(k),
                            num_train=int(merged.get("num_train", 1000)),
                            num_val=int(merged.get("num_val", 200)),
                            num_test=int(merged.get("num_test", 200)),
                            prediction_mode=merged.get("prediction_mode", "auto"),
                            embedding_model=merged.get("embedding_model", "all-MiniLM-L6-v2"),
                            scorer_model=str(merged.get("scorer_model", "gpt2")),
                            max_new_tokens=int(merged.get("max_new_tokens", 10)),
                            method_params=params,
                            model_params=dict(merged.get("model_params") or {}),
                            output_dir=str(merged.get("output_dir", "results/raw")),
                            benchmark=benchmark,
                        )
                        if len(k_values) > 1:
                            run.output_dir = str(Path(run.output_dir) / f"k{k}")
                        runs.append(run)
    return runs


# ---------------------------------------------------------------------- #
# Shared resources (cached per process so the pool is embedded once)
# ---------------------------------------------------------------------- #
class Resources:
    def __init__(self, models_config: dict):
        self.models_config = models_config
        self._models: dict[str, BaseModel] = {}
        self._embedders: dict[str, Embedder] = {}
        self._scorers: dict[str, Any] = {}

    def model(self, cfg: RunConfig, task: Task) -> BaseModel:
        key = f"{cfg.model}|{task.name}"
        if key not in self._models:
            self._models[key] = build_model(
                cfg.model,
                self.models_config,
                label_names=task.label_names,
                output_prefix=task.output_prefix,
                overrides=cfg.model_params,
            )
        return self._models[key]

    def embedder(self, cfg: RunConfig) -> Embedder:
        if cfg.embedding_model not in self._embedders:
            self._embedders[cfg.embedding_model] = Embedder(cfg.embedding_model)
        return self._embedders[cfg.embedding_model]

    def scorer(self, cfg: RunConfig, model: BaseModel) -> Any:
        if cfg.scorer_model == "self":
            scorer = getattr(model, "scorer", None)
            if scorer is None:
                raise ValueError(
                    f"model '{cfg.model}' cannot score; set scorer_model to a checkpoint"
                )
            return scorer
        if cfg.scorer_model not in self._scorers:
            self._scorers[cfg.scorer_model] = LMScorer.from_pretrained(cfg.scorer_model)
        return self._scorers[cfg.scorer_model]


# ---------------------------------------------------------------------- #
# One run
# ---------------------------------------------------------------------- #
def _majority_vote(answers: Sequence[str]) -> str | None:
    if not answers:
        return None
    counts: dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    return max(sorted(counts), key=lambda a: counts[a])


def run_single(cfg: RunConfig, resources: Resources, progress_every: int = 25) -> dict:
    set_seed(cfg.seed)
    task = get_task(cfg.dataset)
    t_start = time.perf_counter()

    train_texts, train_labels = load_train_without_holdout(task, cfg.num_train, cfg.seed)
    test_texts, test_labels = load_split(task, "test", cfg.num_test, cfg.seed)
    demo_pool = task.format_demos(train_texts, train_labels)

    model = resources.model(cfg, task)
    model.usage.reset()
    inference = ICLInference(model, PromptBuilder(task.instruction), cfg.prediction_mode)
    embedder = resources.embedder(cfg)

    # --- selector-specific resources ------------------------------------
    scorer = resources.scorer(cfg, model) if cfg.method in NEEDS_SCORER else None
    if scorer is not None and hasattr(scorer, "reset_counters"):
        scorer.reset_counters()

    zero_shot_cot_fn = icl_fn = None
    if cfg.method in NEEDS_INFERENCE:
        zero_shot_cot_fn = inference.run_zero_shot_cot
        icl_fn = lambda q, demos: inference.run_icl(demos, q, max_tokens=64)  # noqa: E731

    evaluate_subset_fn = None
    val_size = 0
    if cfg.method in NEEDS_VALIDATION:
        val_texts, val_labels = load_split(task, "validation", cfg.num_val, cfg.seed)
        val_size = len(val_texts)
        val_queries = [task.format_query(t) for t in val_texts]

        def evaluate_subset_fn(indices: list[int]) -> float:
            demos = [demo_pool[i] for i in indices]
            correct = 0
            for q, y in zip(val_queries, val_labels):
                correct += inference.predict(demos, q, task, cfg.max_new_tokens)["prediction"] == y
            return correct / max(1, len(val_queries))

    selector = build_selector(
        cfg.method,
        cfg.method_params,
        k=cfg.k,
        seed=cfg.seed,
        embedder=embedder,
        scorer=scorer,
        zero_shot_cot_fn=zero_shot_cot_fn,
        icl_fn=icl_fn,
        evaluate_subset_fn=evaluate_subset_fn,
    )
    t_fit = time.perf_counter()
    selector.fit(demo_pool, train_labels)
    fit_time = time.perf_counter() - t_fit
    # Cost of fitting (e.g. influence's subset evaluations) is reported separately.
    fit_usage = model.usage.to_dict()
    model.usage.reset()

    # --- evaluation loop --------------------------------------------------
    records: list[dict[str, Any]] = []
    select_time = infer_time = 0.0
    for i, (text, gold) in enumerate(zip(test_texts, test_labels)):
        query = task.format_query(text)
        t0 = time.perf_counter()
        demo_idx = selector.select(query)
        t1 = time.perf_counter()
        out = inference.predict([demo_pool[j] for j in demo_idx], query, task, cfg.max_new_tokens)
        t2 = time.perf_counter()
        select_time += t1 - t0
        infer_time += t2 - t1
        rec = {
            "index": i,
            "gold": gold,
            "prediction": out["prediction"],
            "correct": int(out["prediction"] == gold),
            "demo_indices": [int(j) for j in demo_idx],
            "demo_labels": [train_labels[j] for j in demo_idx],
            "mode": out["mode"],
            "raw": out["raw"] if isinstance(out["raw"], str) else None,
        }
        if cfg.method == "ids":
            votes = [inference_parse(a, task) for a in getattr(selector, "last_answers", [])]
            rec["ids_majority_vote"] = _majority_vote([v for v in votes if v != UNKNOWN])
        records.append(rec)
        if progress_every and (i + 1) % progress_every == 0:
            acc = np.mean([r["correct"] for r in records])
            log.info("%s: %d/%d  running acc=%.3f", cfg.run_id, i + 1, len(test_texts), acc)

    # --- metrics ------------------------------------------------------------
    preds = [r["prediction"] for r in records]
    correct = [r["correct"] for r in records]
    metrics = compute_metrics(preds, test_labels)
    metrics["parse_failure_rate"] = float(np.mean([p == UNKNOWN for p in preds])) if preds else 0.0
    metrics["accuracy_ci95"] = bootstrap_ci(correct, seed=cfg.seed)
    if cfg.method == "ids":
        mv = [r["ids_majority_vote"] for r in records]
        metrics["ids_majority_vote_accuracy"] = float(
            np.mean([m == y for m, y in zip(mv, test_labels)])
        )
    demo_label_entropy = _mean_label_entropy(records, task)

    result = {
        "run_id": cfg.run_id,
        "config": asdict(cfg),
        "task": {
            "name": task.name,
            "label_names": list(task.label_names),
            "instruction": task.instruction,
            "num_train": len(train_texts),
            "num_val": val_size,
            "num_test": len(test_texts),
        },
        "model": model.info(),
        "selector": selector.get_config(),
        "metrics": metrics,
        "cost": {
            "fit_time_s": fit_time,
            "selection_time_s": select_time,
            "inference_time_s": infer_time,
            "total_time_s": time.perf_counter() - t_start,
            "fit_usage": fit_usage,
            "eval_usage": model.usage.to_dict(),
            "scorer_forward_passes": getattr(scorer, "num_forward_passes", None),
            "scorer_scored_tokens": getattr(scorer, "num_scored_tokens", None),
        },
        "diagnostics": {
            "mean_demo_label_entropy": demo_label_entropy,
            "influence_coverage": (
                selector.coverage_stats() if hasattr(selector, "coverage_stats") else None
            ),
        },
        "records": records,
        "manifest": build_manifest(REPO_ROOT),
    }
    return result


def inference_parse(answer: str, task: Task) -> str:
    from src.evaluation import parse_prediction

    return parse_prediction(answer, task.label_names, task.output_prefix or None)


def _mean_label_entropy(records: list[dict], task: Task) -> float | None:
    if not records or not records[0]["demo_labels"]:
        return None
    ents = []
    for r in records:
        counts = np.array([r["demo_labels"].count(lb) for lb in task.label_names], dtype=float)
        p = counts / counts.sum()
        ents.append(float(-np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(task.num_classes)))
    return float(np.mean(ents))


def save_result(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--models-config", default=str(DEFAULT_MODELS_CONFIG))
    p.add_argument("--benchmark", default="smoke")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--models", nargs="+")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--k", type=int)
    p.add_argument("--num-train", type=int)
    p.add_argument("--num-val", type=int)
    p.add_argument("--num-test", type=int)
    p.add_argument("--prediction-mode", choices=["auto", "score", "generate"])
    p.add_argument("--scorer-model")
    p.add_argument("--output-dir")
    p.add_argument("--resume", action="store_true", help="skip runs whose result file exists")
    p.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    p.add_argument("--log-file")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logger("icl", log_file=args.log_file)
    config = load_yaml(args.config)
    models_config = (load_yaml(args.models_config) or {}).get("models") or {}

    overrides = {
        "datasets": args.datasets,
        "models": args.models,
        "methods": args.methods,
        "seeds": args.seeds,
        "k": args.k,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "num_test": args.num_test,
        "prediction_mode": args.prediction_mode,
        "scorer_model": args.scorer_model,
        "output_dir": args.output_dir,
    }
    runs = expand_benchmark(config, args.benchmark, overrides)
    log.info("benchmark '%s': %d runs", args.benchmark, len(runs))
    if args.dry_run:
        for r in runs:
            print(r.run_id, "->", r.output_path)
        return 0

    resources = Resources(models_config)
    failures = 0
    for i, cfg in enumerate(runs, 1):
        if args.resume and cfg.output_path.exists():
            log.info("[%d/%d] %s exists, skipping", i, len(runs), cfg.run_id)
            continue
        log.info("[%d/%d] %s", i, len(runs), cfg.run_id)
        try:
            result = run_single(cfg, resources)
        except Exception:
            failures += 1
            log.exception("run %s failed", cfg.run_id)
            continue
        save_result(result, cfg.output_path)
        m = result["metrics"]
        log.info(
            "%s  acc=%.3f [%.3f, %.3f]  f1=%.3f  unparsed=%.1f%%  time=%.1fs",
            cfg.run_id,
            m["accuracy"],
            m["accuracy_ci95"]["lower"],
            m["accuracy_ci95"]["upper"],
            m["f1"],
            100 * m["parse_failure_rate"],
            result["cost"]["total_time_s"],
        )
    if failures:
        log.error("%d run(s) failed", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
