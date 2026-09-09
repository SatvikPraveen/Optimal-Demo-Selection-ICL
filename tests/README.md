# Tests

This directory contains test files for verifying the project setup and functionality.

`verify_setup.py` is a standalone diagnostic script (no `test_*` functions, meant
to be run directly). `test_functionality.py` and `test_project.py` are real
pytest test modules — `pytest tests/` discovers and runs the `test_*` functions
in both. `conftest.py` provides fixtures that mock the sentence-transformers
embedding model, the gpt2 CoNE backend, and the SST-5/AG News dataset downloads,
so the pytest suite needs no network access, `OPENAI_API_KEY`, or GPU — this is
exactly what `.github/workflows/tests.yml` runs in CI.

## Test Files

### `verify_setup.py`
**Purpose:** Installation and setup verification
**What it checks:**
- Virtual environment is active (informational only — not required in CI)
- ✅ Directory structure is correct
- ✅ Configuration files exist
- ✅ All dependencies are installed
- ✅ Module imports work properly

**Run:** `python tests/verify_setup.py`

### `test_functionality.py`
**Purpose:** Unit tests for core functionality
**What it tests:**
- ✅ Module imports
- ✅ Seed control and reproducibility
- ✅ Prompt builder
- ✅ Evaluation metrics (accuracy, F1)
- ✅ Logger configuration
- ✅ Selection algorithm initialization (TopK+CoNE, IDS) — with mocked embedding/CoNE backends

**Run:** `pytest tests/test_functionality.py`

### `test_project.py`
**Purpose:** Integration test for end-to-end functionality
**What it tests:**
- ✅ Dataset loading (SST-5, AG News) — against mocked HF downloads
- ✅ Selection algorithm initialization
- ✅ Embedding computation
- ✅ Prompt building
- ✅ Evaluation metrics
- ✅ Similarity-based retrieval

**Run:** `pytest tests/test_project.py`

## Running All Tests

```bash
# Activate environment
source venv/bin/activate

# Run setup verification
python tests/verify_setup.py

# Run the pytest suite (functional + integration tests)
pytest tests/test_functionality.py tests/test_project.py -v
```

## Expected Output

`verify_setup.py` prints ✅/❌ indicators and exits non-zero on failure.
`pytest` reports PASSED/FAILED per test. If anything fails, check:
1. Dependencies are installed: `pip install -r requirements.txt` (or
   `requirements-lock.txt` for the exact versions CI is verified against)
2. Package is installed: `pip install -e .`

## Adding New Tests

When adding new tests:
1. Follow the naming convention: `test_*.py`
2. Import from `src` package
3. Include clear test descriptions
4. Use assertions or explicit pass/fail messages
