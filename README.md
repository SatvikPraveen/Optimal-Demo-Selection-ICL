# Optimal Demonstration Selection for In-Context Learning

![Tests](https://github.com/SatvikPraveen/Optimal-Demo-Selection-ICL/actions/workflows/tests.yml/badge.svg)
![MIT License](https://img.shields.io/github/license/SatvikPraveen/Optimal-Demo-Selection-ICL)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A reproducible benchmark of demonstration-selection methods for few-shot
in-context learning (ICL). Eight selectors, three tasks and any causal LM
(local HuggingFace checkpoints or the OpenAI API) share one prompt format, one
prediction rule, one evaluation protocol and one config-driven runner, so
methods can be compared like-for-like with confidence intervals and paired
significance tests.

**Team:** Kamisetty Yamini Preethi, Jonathan Tong, Satvik Praveen, Vinay Chandra Bandi (Texas A&M University).

## Contents

- [What is in the box](#what-is-in-the-box)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Running the benchmark](#running-the-benchmark)
- [Evaluation protocol](#evaluation-protocol)
- [Results](#results)
- [Project structure](#project-structure)
- [Extending the benchmark](#extending-the-benchmark)
- [Development](#development)
- [Citation](#citation)

## What is in the box

**Selection methods** (`src/selection/`, details and references in [docs/methods.md](docs/methods.md)):

| Method | Key idea | Uses the LM at selection time |
|---|---|---|
| `random` | uniform random demonstrations (lower bound) | no |
| `topk` (SBERT / kNN) | nearest neighbours in sentence-embedding space | no |
| `bm25` | lexical retrieval (Okapi BM25) | no |
| `topk_cone` | Top-K retrieval re-ranked by conditional entropy of the query (Peng et al., 2024) | scorer LM |
| `ids` | iterative selection driven by zero-shot chain-of-thought rationales (Qin et al., 2023) | evaluated model |
| `rdes` | tabular Q-learning balancing relevance and label diversity (Wang et al., 2024) | no |
| `se2` | sequential beam search over demonstration order scored by the LM (Liu et al., 2024) | scorer LM |
| `influence` | datamodel-style subset-sampling influence scores, one fixed prompt (Nguyen & Wong, 2023) | evaluated model, once |

**Tasks** (`src/datasets/tasks.py`): SST-5 (5-way sentiment), AG News (4-way
topic), CommonsenseQA (5-way multiple choice, scored on the answer letter).

**Models** (`src/models/`): any `AutoModelForCausalLM` checkpoint
(`HFCausalModel`, with LLaMA/Gemma aliases), OpenAI chat models (`GPTModel`),
and a deterministic `DummyModel` so the whole pipeline runs in CI.

**Evaluation** (`src/evaluation/`): accuracy and macro-F1, percentile
bootstrap CIs over test examples, mean ± std over seeds, exact McNemar and
paired-bootstrap tests against a baseline, parse-failure rate, and per-run
cost (fit / selection / inference time, calls, tokens).

## Installation

Python 3.10 or newer. A GPU is recommended for local models but nothing here
requires one.

```bash
git clone https://github.com/SatvikPraveen/Optimal-Demo-Selection-ICL.git
cd Optimal-Demo-Selection-ICL
./setup_env.sh              # creates ./venv, installs the package with dev extras and pre-commit hooks
source venv/bin/activate
```

or manually:

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"     # core + pytest/black/ruff/mypy/matplotlib
```

For a bit-for-bit environment use the lock file: `pip install -r requirements-lock.txt`.

API keys and tokens are read from the environment (`cp .env.example .env`
and fill in `OPENAI_API_KEY` / `HF_TOKEN`; the runner loads `.env` if
`python-dotenv` is installed).

## Quick start

```python
from src.datasets import get_task, load_split, load_train_without_holdout
from src.models import HFCausalModel
from src.prompting import ICLInference, PromptBuilder
from src.selection import TopKCoNE
from src.evaluation import compute_metrics, bootstrap_ci
from src.utils import set_seed

set_seed(0)
task = get_task("sst5")
train_texts, train_labels = load_train_without_holdout(task, num_samples=500, seed=0)
test_texts, test_labels = load_split(task, "test", num_samples=100, seed=0)

model = HFCausalModel("gpt2")                                  # or GPTModel("gpt-4o-mini")
inference = ICLInference(model, PromptBuilder(task.instruction))  # scores label log-probs for HF models

selector = TopKCoNE(k=5, retrieve_k=30, scorer=model.scorer)   # any BaseSelector works here
selector.fit(task.format_demos(train_texts, train_labels), train_labels)

preds = []
for text in test_texts:
    query = task.format_query(text)
    demo_idx = selector.select(query)
    demos = [selector.candidates[i] for i in demo_idx]
    preds.append(inference.predict(demos, query, task)["prediction"])

print(compute_metrics(preds, test_labels))
print(bootstrap_ci([p == y for p, y in zip(preds, test_labels)]))
```

## Running the benchmark

Everything is driven by `configs/experiments.yaml` (benchmarks, defaults,
method hyper-parameters) and `configs/models.yaml` (model registry).

```bash
# All 8 methods on SST-5 with the dummy model: seconds, no GPU or API key.
python experiments/run_benchmark.py --benchmark smoke

# A real run on a local GPT-2 (downloads ~500 MB the first time).
python experiments/run_benchmark.py --benchmark quick_test

# The full grid: 3 datasets x 3 models x 8 methods x 3 seeds. Override anything from the CLI.
python experiments/run_benchmark.py --benchmark full_benchmark --models llama-3.2-3b --seeds 0 1 2 --resume

# k-shot ablation (writes to results/raw/k{1,3,5,8}/).
python experiments/run_benchmark.py --benchmark ablation_k

# Tables + significance tests, then plots.
python experiments/aggregate_results.py --baseline random
python experiments/plot_results.py
```

`--dry-run` lists the runs a benchmark expands to; `--resume` skips runs
whose result file already exists, so a long grid can be restarted.

Each run writes `results/raw/<dataset>__<model>__<method>__seed<seed>.json` with:

- `records`: per test example, the selected pool indices and their labels,
  the prediction, the gold label and the raw model output;
- `metrics`: accuracy, macro-F1, precision, recall, bootstrap 95% CI,
  parse-failure rate (and `ids_majority_vote_accuracy` for IDS);
- `cost`: fit / selection / inference wall time, model calls and tokens
  split into fit-time and evaluation-time usage, scorer forward passes;
- `selector`, `model`, `task`, `config`: every hyper-parameter of the run;
- `manifest`: git commit (and whether the tree was dirty), Python and
  package versions, CUDA device.

`aggregate_results.py` produces `results/processed/runs.csv`, `summary.csv`,
`significance.csv` and `summary.md` (mean ± std over seeds per dataset,
model and method, plus Δ accuracy vs the baseline with a paired-bootstrap CI
and an exact McNemar p-value pooled over seeds).

## Evaluation protocol

- **Splits.** The demonstration pool is sampled from the training split
  and is disjoint from the validation slice (used only by `influence`) and
  from the test set. CommonsenseQA's labeled validation split serves as its
  test set because the official test split is unlabeled.
- **Prompts.** One template per task (`src/datasets/tasks.py`):
  instruction, then `k` demonstrations, then the query.
- **Prediction.** Local models rank the label verbalizers by
  `log p(label | prompt)` (`prediction_mode: score`); API models generate
  and the text is mapped onto the label set by a documented decoding rule
  (`prediction_mode: generate`). Unparseable outputs count as wrong and are
  reported separately.
- **Uncertainty.** Bootstrap CIs over test examples within a run; mean ±
  sample std and a t-interval over seeds; paired McNemar and bootstrap
  tests between methods evaluated on the same examples.
- **Reproducibility.** `set_seed` seeds Python, NumPy and torch; every
  stochastic selector owns a seeded generator; result files carry the
  commit hash and library versions.

## Results

No benchmark table is published from the current code yet. Producing one is
a single command (`--benchmark full_benchmark` followed by
`aggregate_results.py`), and this README will be updated with the resulting
`summary.md` once a full grid has been run on the target models.

The accuracies obtained by the original notebook experiments (2024) are kept
in [docs/legacy_results.md](docs/legacy_results.md) together with the figures
in `Figures/`. They were produced by the archived notebooks, not by `src/`,
with different prompts, data slices and, in places, flawed scoring, so they are
not comparable to numbers produced by this benchmark; [docs/methods.md](docs/methods.md)
lists what changed in each method and why.

## Project structure

```
configs/
  experiments.yaml      benchmarks, defaults, method hyper-parameters
  models.yaml           model registry (type, checkpoint, kwargs)
  datasets.yaml         dataset metadata (informational; templates live in src/datasets/tasks.py)
experiments/
  run_benchmark.py      config-driven runner (one JSON per run)
  aggregate_results.py  tables + significance tests
  plot_results.py       accuracy bars with CIs, cost-vs-accuracy scatter
src/
  datasets/             loaders (SST-5, AG News, CSQA) and the Task registry
  models/               BaseModel + usage tracking, HFCausalModel, GPTModel, DummyModel, LMScorer
  selection/            BaseSelector, baselines, TopKCoNE, IDS, RDES, Se2, InfluenceSelection, registry
  prompting/            PromptBuilder, ICLInference.predict
  evaluation/           metrics, parsing, stats (bootstrap, McNemar, permutation)
  utils/                seeding, logging, shared Embedder, environment manifest
tests/                  pytest suite; all network boundaries mocked (no GPU / keys needed)
docs/                   methods.md (algorithms and deviations), legacy_results.md
notebooks_archive/      the original course-project notebooks (reference only)
Figures/, paper/        figures and report from the original project
```

## Extending the benchmark

**A new selector**: subclass `src.selection.BaseSelector`, implement
`fit`/`select`/`get_config`, register it in `src/selection/registry.py`
and add its hyper-parameters under `methods:` in `configs/experiments.yaml`.
If it needs the LM, take an `LMScorer` (log-likelihoods) or the IDS-style
callbacks (generation) in `__init__`; the registry wires them up.

**A new task**: add a loader returning `(texts, labels)` and a `Task`
entry (label names, instruction, input/output prefixes, split mapping) to
`src/datasets/tasks.py`.

**A new model**: add an entry to `configs/models.yaml`; any
`AutoModelForCausalLM` checkpoint works out of the box. Other backends
subclass `BaseModel` and implement `_generate` (and `_score_choices` if
they expose log-probabilities).

## Development

```bash
pytest                          # ~80 tests, a few seconds, no network
ruff check src tests experiments
black --check src tests experiments
pre-commit install              # runs both on every commit
python tests/verify_setup.py    # installation diagnostic
```

CI (`.github/workflows/tests.yml`) runs lint, the test suite on Python 3.10
and 3.11, and the smoke benchmark. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

```bibtex
@misc{praveen2024optimal,
  title  = {Optimal Demonstration Selection for In-Context Learning},
  author = {Praveen, Satvik and Tong, Jonathan and Kamisetty, Yamini Preethi and Bandi, Vinay Chandra},
  year   = {2024},
  note   = {Texas A\&M University. \url{https://github.com/SatvikPraveen/Optimal-Demo-Selection-ICL}}
}
```

A `CITATION.cff` is included for GitHub's "Cite this repository" button.

## License

MIT, see [LICENSE](LICENSE).
