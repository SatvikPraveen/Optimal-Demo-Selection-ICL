# Contributing

Thanks for considering a contribution. This repository benchmarks
demonstration-selection methods for in-context learning; contributions that
add methods, tasks, models, tests or documentation are all welcome.

## Workflow

1. Fork the repository and create a branch (`git checkout -b feature/my-change`).
2. Set up the environment: `./setup_env.sh && source venv/bin/activate`
   (installs the package with dev extras and the pre-commit hooks).
3. Make your change with clear, focused commits.
4. Run the checks locally:

   ```bash
   ruff check src tests experiments
   black --check src tests experiments
   pytest
   python experiments/run_benchmark.py --benchmark smoke   # optional end-to-end check
   ```

5. Open a pull request describing what changed, why, and any experiments
   you ran to validate it.

## Code style

- `black` (line length 100) and `ruff` are enforced in CI; `pre-commit`
  runs them for you.
- Type hints and docstrings on public classes and functions.
- Reusable code goes in `src/`; runnable entry points in `experiments/`.

## Adding a selection method

- Subclass `src.selection.BaseSelector`; implement `fit`, `select` and
  `get_config`.
- Register it in `src/selection/registry.py` and add default
  hyper-parameters under `methods:` in `configs/experiments.yaml`.
- Add unit tests under `tests/` that run without network access (see
  `tests/conftest.py` for the fixtures that mock the embedding model, the
  LM scorer and the dataset downloads) and add the method to the `smoke`
  benchmark so `tests/test_benchmark.py` exercises it end to end.
- Document the algorithm, its reference and any deviations in
  `docs/methods.md`.

## Reporting results

Please attach the JSON produced by `experiments/run_benchmark.py` (it
carries the commit hash, library versions and every hyper-parameter) rather
than numbers copied from a terminal, and state the seeds used.

## Reporting bugs

Open an issue with the command you ran, the full traceback, and the output
of `python tests/verify_setup.py`.
