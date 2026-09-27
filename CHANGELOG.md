# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-09-26

Research-grade rewrite of the benchmark.

### Added
- `BaseSelector` protocol (`fit` / `select` / `get_config`) shared by all methods.
- Baselines: `RandomSelector`, `TopKSelector` (SBERT/kNN), `BM25Selector`.
- `Se2` (sequential beam search) and `InfluenceSelection` (subset-sampling
  influence with difference and ridge estimators); both were previously
  `NotImplementedError` stubs.
- `LMScorer`: exact per-example conditional log-likelihoods with padding and
  context masking; used by TopK+ConE, Se² and HF label scoring.
- `Task` registry with prompt templates, label verbalizers and logical
  train/validation/test splits kept disjoint.
- Prediction parsing with a documented decoding rule and explicit
  parse-failure reporting; constrained label scoring for local models.
- Model layer: `HFCausalModel` (LLaMA/Gemma aliases), token/call usage
  tracking, `DummyModel` for CI.
- Statistics: bootstrap CIs, paired bootstrap, exact McNemar, permutation
  test, seed summaries, method-vs-baseline comparison.
- `experiments/run_benchmark.py` (config-driven grid runner with resume and
  environment manifests), `aggregate_results.py`, `plot_results.py`.
- Packaging via `pyproject.toml`; ruff/black/pre-commit; CI lint + smoke benchmark.
- `docs/methods.md`, `docs/legacy_results.md`, `CITATION.cff`.

### Changed
- `IDS`, `TopKCoNE` and `RDES` moved onto `BaseSelector`; legacy entry
  points (`select_demonstrations`, `precompute_train_embeddings`,
  `TopKCoNE(embeddings=, raw_texts=)`) are kept as thin wrappers.
- `RDES` constructor now takes hyper-parameters only; the pool is bound with
  `fit(candidates, labels)`. It owns a seeded RNG instead of the global one.
- `load_commonsense_qa` returns answer letters by default
  (`answer_format="text"` restores the previous behaviour).
- `compute_confidence_interval` delegates to `stats.summarize_seeds`.
- `experiments/run_ids.py` and `run_topk_cone.py` replaced by the unified runner.
- README rewritten to describe the actual state of the code; the
  unverified placeholder results table was removed.

### Removed
- `setup.py` (superseded by `pyproject.toml`).
- Stale "transformation complete" documents under `docs/`.

## [0.1.0] - 2026-03

- Modular `src/` layout ported from the original notebooks; IDS and
  TopK+ConE implementations; RDES port; CI with mocked network boundaries.
