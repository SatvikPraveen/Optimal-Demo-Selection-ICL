#!/usr/bin/env python3
"""
Verification script to test the installation and setup
"""

import sys
from pathlib import Path

def check_imports():
    """Test if all modules can be imported"""
    print("🔍 Checking module imports...")
    
    try:
        from src import datasets, models, selection, evaluation, utils
        print("  ✅ Core modules imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import core modules: {e}")
        return False
    
    try:
        from src.datasets import load_sst5, load_agnews, load_commonsense_qa
        print("  ✅ Dataset loaders imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import dataset loaders: {e}")
        return False
    
    try:
        from src.models import BaseModel, GPTModel, LlamaModel, GemmaModel
        print("  ✅ Model interfaces imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import model interfaces: {e}")
        return False
    
    try:
        from src.selection import TopKCoNE, IDS
        print("  ✅ Selection algorithms imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import selection algorithms: {e}")
        return False
    
    try:
        from src.evaluation import compute_metrics
        print("  ✅ Evaluation metrics imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import evaluation metrics: {e}")
        return False
    
    try:
        from src.utils import set_seed, setup_logger
        print("  ✅ Utilities imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import utilities: {e}")
        return False
    
    return True

def check_dependencies():
    """Check if key dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('datasets', 'datasets'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('sentence-transformers', 'sentence_transformers'),
        ('openai', 'openai'),
        ('tqdm', 'tqdm'),
    ]
    
    all_installed = True
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_structure():
    """Check if directory structure is correct"""
    print("\n🔍 Checking directory structure...")
    
    required_dirs = [
        'src',
        'src/datasets',
        'src/models',
        'src/selection',
        'src/evaluation',
        'src/prompting',
        'src/utils',
        'configs',
        'experiments',
        'results',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ - MISSING")
            all_exist = False
    
    return all_exist

def check_configs():
    """Check if config files exist"""
    print("\n🔍 Checking configuration files...")
    
    config_files = [
        'configs/datasets.yaml',
        'configs/models.yaml',
        'configs/experiments.yaml',
        'requirements.txt',
        'setup.py',
        '.gitignore',
    ]
    
    all_exist = True
    for file_path in config_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def check_venv():
    """Check if running in virtual environment"""
    print("\n🔍 Checking virtual environment...")
    
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print(f"  ✅ Running in virtual environment")
        print(f"     Python: {sys.executable}")
        return True
    else:
        print(f"  ⚠️  NOT running in virtual environment")
        print(f"     Python: {sys.executable}")
        print(f"     Consider activating venv: source venv/bin/activate")
        return False

def main():
    """Run all checks"""
    print("=" * 70)
    print("🧪 Optimal Demo Selection ICL - Installation Verification")
    print("=" * 70)
    
    checks = {
        "Virtual Environment": check_venv(),
        "Directory Structure": check_structure(),
        "Configuration Files": check_configs(),
        "Dependencies": check_dependencies(),
        "Module Imports": check_imports(),
    }
    
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    
    for check_name, passed in checks.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(checks.values())
    
    print("=" * 70)
    if all_passed:
        print("🎉 All checks passed! Your installation is ready.")
        print("\n📝 Next steps:")
        print("   1. Set up API keys: cp .env.example .env")
        print("   2. Read docs/QUICK_START.md for usage examples")
        print("   3. Run a test experiment: python experiments/run_ids.py --help")
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you activated the virtual environment:")
        print("      source venv/bin/activate")
        print("   2. Install the package:")
        print("      pip install -e .")
        print("   3. Install missing dependencies:")
        print("      pip install -r requirements.txt")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
