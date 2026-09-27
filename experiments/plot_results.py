#!/usr/bin/env python
"""
Plot aggregated benchmark results (requires the ``plots`` extra).

Reads ``results/processed/summary.csv`` and ``runs.csv`` (produced by
``aggregate_results.py``) and writes to ``results/plots/``:

* ``accuracy_<model>.png``: grouped bars, methods x datasets, with CI bars
* ``cost_vs_accuracy.png``: per-run selection+inference time vs accuracy
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def plot_accuracy(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    paths = []
    for (model, k), g in summary.groupby(["model", "k"]):
        datasets = sorted(g["dataset"].unique())
        methods = sorted(g["method"].unique())
        x = np.arange(len(datasets))
        width = 0.8 / max(1, len(methods))
        fig, ax = plt.subplots(figsize=(1.8 * len(datasets) + 3, 4))
        for j, method in enumerate(methods):
            means, errs = [], []
            for d in datasets:
                row = g[(g["dataset"] == d) & (g["method"] == method)]
                if row.empty:
                    means.append(np.nan)
                    errs.append(0)
                else:
                    r = row.iloc[0]
                    means.append(r["accuracy_mean"])
                    errs.append(r["accuracy_mean"] - r["accuracy_ci_lower"])
            ax.bar(
                x + (j - len(methods) / 2 + 0.5) * width,
                means,
                width,
                yerr=errs,
                label=method,
                capsize=2,
            )
        ax.set_xticks(x, datasets)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"{model} (k={k})")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"accuracy_{model}_k{k}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_cost(runs: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    runs = runs.assign(eval_time=runs["selection_time_s"] + runs["inference_time_s"])
    for method, g in runs.groupby("method"):
        ax.scatter(g["eval_time"] / g["num_test"], g["accuracy"], label=method, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Seconds per test query (selection + inference)")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "cost_vs_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--processed-dir", default="results/processed")
    p.add_argument("--output-dir", default="results/plots")
    args = p.parse_args(argv)
    processed = Path(args.processed_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(processed / "summary.csv")
    runs = pd.read_csv(processed / "runs.csv")
    paths = plot_accuracy(summary, out) + [plot_cost(runs, out)]
    for path in paths:
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
