#!/usr/bin/env python
"""
Enumerate the (dataset, model, method) cells of a benchmark so a Slurm array
index maps to exactly one cell.

    python scripts/slurm/grid.py --benchmark full_benchmark            # print the table
    python scripts/slurm/grid.py --benchmark full_benchmark --index 7  # "agnews llama-3.2-3b topk"
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def cells(
    benchmark: str, config_path: Path, models: list[str] | None = None
) -> list[tuple[str, str, str]]:
    cfg = yaml.safe_load(config_path.read_text())
    merged = {**(cfg.get("defaults") or {}), **(cfg["benchmarks"][benchmark])}
    model_list = models or merged["models"]
    return list(itertools.product(merged["datasets"], model_list, merged["methods"]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", default="full_benchmark")
    p.add_argument("--config", default=str(REPO / "configs" / "experiments.yaml"))
    p.add_argument(
        "--models", nargs="+", help="restrict to these models (default: the benchmark's)"
    )
    p.add_argument("--index", type=int, help="print the cell for this array index")
    args = p.parse_args()
    grid = cells(args.benchmark, Path(args.config), args.models)
    if args.index is not None:
        if not 0 <= args.index < len(grid):
            raise SystemExit(f"index {args.index} out of range (0-{len(grid) - 1})")
        print(*grid[args.index])
        return 0
    for i, (d, m, s) in enumerate(grid):
        print(f"{i:3d}  {d:8s} {m:16s} {s}")
    print(f"# {len(grid)} cells -> sbatch --array=0-{len(grid) - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
