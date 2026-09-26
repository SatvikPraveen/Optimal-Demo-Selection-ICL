#!/bin/bash
# ============================================================================
# Environment setup for Optimal Demo Selection ICL
#
# Creates ./venv, installs the package in editable mode with the dev extras
# (pytest, black, ruff, mypy, matplotlib) and installs the pre-commit hooks.
# Pass --lock to install the exact pinned versions from requirements-lock.txt
# instead of the loose lower bounds in pyproject.toml.
# ============================================================================

set -euo pipefail

PYTHON=${PYTHON:-python3}
USE_LOCK=0
for arg in "$@"; do
  case "$arg" in
    --lock) USE_LOCK=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

echo "Creating virtual environment in ./venv with $PYTHON ..."
"$PYTHON" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip

if [ "$USE_LOCK" = "1" ]; then
  echo "Installing pinned dependencies from requirements-lock.txt ..."
  pip install -r requirements-lock.txt
fi

echo "Installing package (editable) with dev extras ..."
pip install -e ".[dev]"

if command -v pre-commit >/dev/null 2>&1; then
  pre-commit install >/dev/null && echo "pre-commit hooks installed."
fi

echo ""
echo "Setup complete. Activate with:  source venv/bin/activate"
echo "Run the test-suite with:        pytest"
