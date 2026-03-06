# Tests

This directory contains test files for verifying the project setup and functionality.

## Test Files

### `verify_setup.py`
**Purpose:** Installation and setup verification  
**What it checks:**
- ✅ Virtual environment is active
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
- ✅ Selection algorithm initialization (TopK+CoNE, IDS)

**Run:** `python tests/test_functionality.py`

### `test_project.py`
**Purpose:** Integration test for end-to-end functionality  
**What it tests:**
- ✅ Dataset loading (SST-5, AG News)
- ✅ Selection algorithm initialization
- ✅ Embedding computation
- ✅ Prompt building
- ✅ Evaluation metrics
- ✅ Similarity-based retrieval

**Run:** `python tests/test_project.py`

## Running All Tests

```bash
# Activate environment
source venv/bin/activate

# Run setup verification
python tests/verify_setup.py

# Run functional tests
python tests/test_functionality.py

# Run integration test
python tests/test_project.py
```

## Expected Output

All tests should pass with ✅ indicators. If any test fails, check:
1. Virtual environment is activated
2. Dependencies are installed: `pip install -r requirements.txt`
3. Package is installed: `pip install -e .`

## Adding New Tests

When adding new tests:
1. Follow the naming convention: `test_*.py`
2. Import from `src` package
3. Include clear test descriptions
4. Use assertions or explicit pass/fail messages
