#!/bin/bash
# ============================================================================
# Environment Setup Script for Optimal Demo Selection ICL
# ============================================================================

set -e  # Exit on error

echo "======================================================================"
echo "Setting up virtual environment for Optimal Demo Selection ICL"
echo "======================================================================"

# Create virtual environment
echo "Creating virtual environment in ./venv..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "======================================================================"
echo "✓ Setup complete!"
echo "======================================================================"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
