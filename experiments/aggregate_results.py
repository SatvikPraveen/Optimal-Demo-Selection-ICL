#!/usr/bin/env python
"""
Aggregate ``results/raw/*.json`` into tables and significance tests.

Outputs (in ``--output-dir``, default ``results/processed``):

* ``runs.csv``     - one row per run (dataset, model, method, seed, metrics, cost)
* ``summary.csv``  - mean / std / n over seeds per (dataset, model, method)
* ``summary.md``   - the same as Markdown tables, one per model, plus paired
                     significance of every method against ``--baseline``
                     (McNemar and paired bootstrap on per-example correctness,
                     pooled over seeds).

Example::

    python experiments/aggregate_results.py --baseline random
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.evaluation import compare_methods, summarize_seeds


def load_runs(raw_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(raw_dir.rglob("*.json")):
        with open(path) as f:
            r = json.load(f)
        if "metrics" in r and "config" in r:
            r["_path"] = str(path)
            runs.append(r)
    return runs


def runs_dataframe(runs: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        c, m, cost = r["config"], r["metrics"], r["cost"]
        eval_usage = cost.get("eval_usage", {})
        rows.append(
            {
                "dataset": c["dataset"],
                "model": c["model"],
                "method": c["method"],
                "k": c["k"],
                "seed": c["seed"],
                "num_test": r["task"]["num_test"],
                "accuracy": m["accuracy"],
                "acc_ci_lower": m["accuracy_ci95"]["lower"],
                "acc_ci_upper": m["accuracy_ci95"]["upper"],
                "f1_macro": m["f1"],
                "parse_failure_rate": m["parse_failure_rate"],
                "fit_time_s": cost["fit_time_s"],
                "selection_time_s": cost["selection_time_s"],
                "inference_time_s": cost["inference_time_s"],
                "total_time_s": cost["total_time_s"],
                "eval_prompt_tokens": eval_usage.get("prompt_tokens"),
                "eval_calls": eval_usage.get("calls", 0) + eval_usage.get("scoring_calls", 0),
                "fit_calls": cost.get("fit_usage", {}).get("calls", 0)
                + cost.get("fit_usage", {}).get("scoring_calls", 0),
                "commit": (r.get("manifest", {}).get("git") or {}).get("commit"),
                "path": r["_path"],
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for (dataset, model, method, k), g in df.groupby(
        ["dataset", "model", "method", "k"], sort=True
    ):
        acc = summarize_seeds(g["accuracy"].tolist())
        f1 = summarize_seeds(g["f1_macro"].tolist())
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "k": k,
                "n_seeds": acc["n"],
                "accuracy_mean": acc["mean"],
                "accuracy_std": acc["std"],
                "accuracy_ci_lower": acc["lower"],
                "accuracy_ci_upper": acc["upper"],
                "f1_mean": f1["mean"],
                "f1_std": f1["std"],
                "parse_failure_rate": g["parse_failure_rate"].mean(),
                "selection_time_s": g["selection_time_s"].mean(),
                "inference_time_s": g["inference_time_s"].mean(),
                "fit_time_s": g["fit_time_s"].mean(),
            }
        )
    return pd.DataFrame(rows)


def significance(runs: Sequence[dict], baseline: str) -> pd.DataFrame:
    """Paired tests vs baseline, pooling per-example correctness over seeds."""
    correct: dict[tuple, dict[str, list[int]]] = defaultdict(dict)
    for r in runs:
        c = r["config"]
        key = (c["dataset"], c["model"], c["k"], c["seed"])
        correct[key][c["method"]] = [rec["correct"] for rec in r["records"]]

    pooled: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for (dataset, model, k, _seed), by_method in correct.items():
        if baseline not in by_method:
            continue
        n = len(by_method[baseline])
        for method, corr in by_method.items():
            if len(corr) == n:
                pooled[(dataset, model, k)][method].extend(corr)

    rows = []
    for (dataset, model, k), by_method in sorted(pooled.items()):
        if len(by_method) < 2:
            continue
        res = compare_methods(by_method, baseline)
        for method, stats in res.items():
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "k": k,
                    "method": method,
                    "baseline": baseline,
                    "n_examples": len(by_method[method]),
                    "delta_accuracy": stats["delta"],
                    "delta_ci_lower": stats["lower"],
                    "delta_ci_upper": stats["upper"],
                    "bootstrap_p": stats["p_value"],
                    "mcnemar_p": stats["mcnemar_p_value"],
                    "n_discordant": stats["n_discordant"],
                }
            )
    return pd.DataFrame(rows)


def _fmt(mean: float, std: float, n: int) -> str:
    return f"{mean:.3f} ± {std:.3f}" if n > 1 else f"{mean:.3f}"


def to_markdown(summary: pd.DataFrame, sig: pd.DataFrame, baseline: str) -> str:
    if summary.empty:
        return "_No runs found._\n"
    out = ["# Benchmark summary", ""]
    out.append("Accuracy (mean ± sample std over seeds; single seed shown without ±).")
    out.append("")
    for (model, k), g in summary.groupby(["model", "k"], sort=True):
        datasets = sorted(g["dataset"].unique())
        out.append(f"## Model: `{model}` (k = {k})")
        out.append("")
        out.append("| Method | " + " | ".join(datasets) + " | Avg | n seeds |")
        out.append("|---|" + "---|" * (len(datasets) + 2))
        for method, gm in g.groupby("method", sort=True):
            cells, means = [], []
            for d in datasets:
                row = gm[gm["dataset"] == d]
                if row.empty:
                    cells.append("—")
                else:
                    r = row.iloc[0]
                    cells.append(_fmt(r["accuracy_mean"], r["accuracy_std"], int(r["n_seeds"])))
                    means.append(r["accuracy_mean"])
            avg = f"{sum(means) / len(means):.3f}" if means else "—"
            n = int(gm["n_seeds"].max())
            out.append(f"| {method} | " + " | ".join(cells) + f" | {avg} | {n} |")
        out.append("")
    if not sig.empty:
        out.append(f"## Paired significance vs `{baseline}`")
        out.append("")
        out.append(
            "Δ accuracy with 95% paired-bootstrap CI and exact McNemar p-value, "
            "per-example correctness pooled over seeds."
        )
        out.append("")
        out.append("| Dataset | Model | k | Method | Δ acc | 95% CI | McNemar p | n |")
        out.append("|---|---|---|---|---|---|---|---|")
        for _, r in sig.sort_values(["model", "dataset", "k", "method"]).iterrows():
            out.append(
                f"| {r['dataset']} | {r['model']} | {r['k']} | {r['method']} | "
                f"{r['delta_accuracy']:+.3f} | [{r['delta_ci_lower']:+.3f}, {r['delta_ci_upper']:+.3f}] | "
                f"{r['mcnemar_p']:.3g} | {int(r['n_examples'])} |"
            )
        out.append("")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw-dir", default="results/raw")
    p.add_argument("--output-dir", default="results/processed")
    p.add_argument("--baseline", default="random")
    args = p.parse_args(argv)

    runs = load_runs(Path(args.raw_dir))
    if not runs:
        print(f"No result files under {args.raw_dir}")
        return 1
    df = runs_dataframe(runs)
    summary = summarize(df)
    sig = significance(runs, args.baseline)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    if not sig.empty:
        sig.to_csv(out / "significance.csv", index=False)
    md = to_markdown(summary, sig, args.baseline)
    (out / "summary.md").write_text(md)
    print(md)
    print(f"Wrote {out / 'runs.csv'}, {out / 'summary.csv'}, {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
