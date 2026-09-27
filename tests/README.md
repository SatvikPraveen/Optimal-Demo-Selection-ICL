# Tests

`pytest` runs everything under this directory in a few seconds. No network
access, API key or GPU is needed: `conftest.py` provides fixtures that mock
the three network boundaries.

| Fixture | Replaces |
|---|---|
| `mock_sentence_transformer` | the sentence-transformers model behind `src.utils.Embedder` |
| `mock_gpt2_cone_backend` | `LMScorer.from_pretrained` (returns a deterministic `FakeScorer`) |
| `mock_hf_datasets` | the SST-5 / AG News / CommonsenseQA downloads |

| File | Covers |
|---|---|
| `test_functionality.py` | seeding, prompt builder, metrics, logger, IDS / TopK+ConE / RDES |
| `test_baselines.py` | Random / Top-K / BM25 selectors, `Embedder`, `cosine_top_k` |
| `test_scoring.py` | `LMScorer` against a manual log-softmax on a tiny config-built GPT-2 |
| `test_se2_influence.py` | Se² beam search (with a fake scorer), influence estimators and persistence |
| `test_tasks_parsing.py` | task templates, split handling, prediction parsing, `DummyModel`, `ICLInference.predict` |
| `test_stats.py` | bootstrap CIs, McNemar, permutation and paired bootstrap tests |
| `test_benchmark.py` | every method end to end through `run_benchmark.py`, aggregation and plotting |
| `test_project.py` | integration of loaders, selectors, prompts and metrics |
| `verify_setup.py` | standalone installation diagnostic (run directly, not via pytest) |

Tests that would load real weights or hit the network should be marked
`@pytest.mark.slow` (none currently) and are deselected with `-m "not slow"`.
