# Optimal Demonstration Selection for In-Context Learning

![MIT License](https://img.shields.io/github/license/SatvikPraveen/Optimal-Demo-Selection-ICL)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **A modular, research-grade framework for benchmarking demonstration selection methods in few-shot in-context learning.**

This repository implements and evaluates multiple demonstration selection strategies for In-Context Learning (ICL) with large language models. We provide a clean, extensible codebase for reproducing and extending research on optimal example selection.

---

## 👥 Team Members

- **Kamisetty Yamini Preethi** • yamini_preethi_k@tamu.edu
- **Jonathan Tong** • tongjo@tamu.edu
- **Satvik Praveen** • satvikpraveen_164@tamu.edu
- **Vinay Chandra Bandi** • vinaychandra@tamu.edu

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Selection Strategies](#selection-strategies)
- [Models & Datasets](#models--datasets)
- [Running Experiments](#running-experiments)
- [Results](#results)
- [Development Guide](#development-guide)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

In-context learning enables LLMs to perform tasks using only a few demonstration examples in the prompt. However, **which demonstrations to select** dramatically impacts performance. This project benchmarks five state-of-the-art selection algorithms:

### Key Research Questions

1. **Which selection method performs best?** Across different tasks and models
2. **How does computational cost trade off with accuracy?** 
3. **What factors make demonstrations effective?** Relevance, diversity, ordering

### Methodology

- **3 tasks**: Sentiment (SST-5), Topic Classification (AG News), Commonsense Reasoning (CSQA)
- **3+ models**: GPT-4o, LLaMA-3.2-3B, Gemma-2B, GPT-2
- **5 algorithms**: TopK+CoNE, IDS, RDES, Se², Influence-based
- **Rigorous evaluation**: Multiple seeds, statistical testing, comprehensive metrics

---

## ✨ Features

### 🏗️ Research-Grade Architecture

- **Modular design**: Clean separation of concerns (datasets, models, selection, evaluation)
- **Reproducible**: Seed control, detailed logging, configuration management
- **Extensible**: Add new methods, datasets, or models with minimal code changes
- **Type-safe**: Type hints throughout for better code quality

### 🧪 Comprehensive Benchmarking

- Multiple baseline methods (Random, BM25, SBERT)
- Statistical significance testing
- Confidence intervals via bootstrapping
- Computational cost tracking
- Visualization tools

### 🚀 Production-Ready Code

- Unit tests and CI/CD ready
- Proper dependency management
- Virtual environment support
- API key management
- Detailed documentation

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible GPU recommended for local models
- **API Keys**: 
  - OpenAI API key (for GPT models)
  - HuggingFace token (for gated models like LLaMA)

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/SatvikPraveen/Optimal-Demo-Selection-ICL.git
cd Optimal-Demo-Selection-ICL

# 2. Run automated setup (creates venv and installs dependencies)
chmod +x setup_env.sh
./setup_env.sh

# 3. Activate virtual environment
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# 4. Set up API keys
cp .env.example .env
# Edit .env and add your API keys (OPENAI_API_KEY, HF_TOKEN)

# 5. Install package
pip install -e .
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

---

## 🎮 Quick Start

### Example: Run IDS on SST-5

```python
from src.datasets import load_sst5
from src.models import GPTModel
from src.selection import IDS
from src.prompting import ICLInference, PromptBuilder
from src.evaluation import compute_metrics
from src.utils import set_seed

# Set seed for reproducibility
set_seed(42)

# Load dataset
train_texts, train_labels = load_sst5(split="train", num_samples=1000)
test_texts, test_labels = load_sst5(split="test", num_samples=100)

# Initialize components
model = GPTModel(model_name="gpt-4o-mini")
selector = IDS(k=5, embedding_model="all-MiniLM-L6-v2")
inference = ICLInference(model=model)

# Run ICL with selected demonstrations
predictions = []
for test_text in test_texts:
    # Select demonstrations
    demo_indices = selector.select_demonstrations(
        test_text, 
        train_texts,
        zero_shot_cot_fn=inference.run_zero_shot_cot,
        icl_fn=lambda q, demos: inference.run_icl(demos, q)
    )
    
    # Get predictions
    demos = [train_texts[i] for i in demo_indices]
    pred = inference.run_icl(demos, test_text)
    predictions.append(pred)

# Evaluate
metrics = compute_metrics(predictions, test_labels)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1: {metrics['f1']:.3f}")
```

### Quick Benchmark Test

```bash
# Run a quick test with small dataset
python experiments/run_benchmark.py \\
    --config configs/experiments.yaml \\
    --benchmark quick_test \\
    --output results/quick_test.json
```

---

## 📁 Project Structure

```
optimal-demo-selection-icl/
│
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package installation configuration
├── 📄 .gitignore                   # Git ignore rules (includes venv/)
├── 📄 setup_env.sh                # Automated environment setup
│
├── 📁 venv/                        # Virtual environment (in repo, but gitignored)
│
├── 📁 configs/                     # YAML configuration files
│   ├── datasets.yaml              # Dataset configurations
│   ├── models.yaml                # Model configurations  
│   └── experiments.yaml           # Experiment configurations
│
├── 📁 src/                         # 🔥 Main source code (modular)
│   ├── __init__.py
│   │
│   ├── 📁 datasets/               # Dataset loaders
│   │   ├── __init__.py
│   │   ├── load_sst5.py          # SST-5 sentiment dataset
│   │   ├── load_agnews.py        # AG News topic classification
│   │   └── load_csqa.py          # CommonsenseQA
│   │
│   ├── 📁 models/                 # LLM model interfaces
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract base class
│   │   ├── gpt.py                # OpenAI GPT models
│   │   ├── llama.py              # Meta LLaMA models
│   │   └── gemma.py              # Google Gemma models
│   │
│   ├── 📁 selection/              # 🧠 Demonstration selection algorithms
│   │   ├── __init__.py
│   │   ├── topk_cone.py          # TopK + CoNE
│   │   ├── ids.py                # Iterative Demonstration Selection
│   │   ├── rdes.py               # RDES
│   │   ├── se2.py                # Se²
│   │   └── influence.py          # Influence-based selection
│   │
│   ├── 📁 prompting/              # Prompt construction & inference
│   │   ├── __init__.py
│   │   ├── prompt_builder.py     # Prompt templates
│   │   └── inference.py          # ICL inference engine
│   │
│   ├── 📁 evaluation/             # Metrics & benchmarking
│   │   ├── __init__.py
│   │   └── metrics.py            # Accuracy, F1, confidence intervals
│   │
│   └── 📁 utils/                  # Utilities
│       ├── __init__.py
│       ├── seed.py               # Reproducibility utilities
│       └── logging.py            # Logging configuration
│
├── 📁 experiments/                # 🧪 Experiment scripts
│   ├── run_topk_cone.py
│   ├── run_ids.py
│   ├── run_rdes.py
│   └── run_benchmark.py          # Main benchmarking script
│
├── 📁 notebooks/                  # 📊 Jupyter notebooks for analysis
│   └── analysis.ipynb
│
├── 📁 results/                    # 📈 Experiment results
│   ├── raw/                      # Raw predictions (.gitignored)
│   ├── processed/                # Aggregated metrics (.gitignored)
│   └── plots/                    # Visualizations (.gitignored)
│
├── 📁 Figures/                    # Paper figures
├── 📁 paper/                      # Research paper (PDF)
│
└── 📁 Old Notebooks/              # Legacy notebooks (for reference)
    ├── TopK+CoNE/
    ├── IDS/
    ├── RDES/
    ├── SE2/
    └── ICINF/
```

### Key Design Principles

✅ **Separation of Concerns**: Each module has a single responsibility  
✅ **Configuration Over Code**: YAML configs for easy experimentation  
✅ **Reproducibility First**: Seed control, logging, version tracking  
✅ **Easy Extension**: Add new methods via inheritance, not modification

---

## 🧠 Selection Strategies

### Implemented Methods

| Method | Description | Key Idea | Complexity |
|--------|-------------|----------|------------|
| **TopK + CoNE** | Embedding retrieval + Cross-entropy refinement | Information gain quantification | O(n·k) |
| **IDS** | Iterative refinement with CoT | Align demos with reasoning path | O(q·n·k) |
| **RDES** | RL-based selection | Balance relevance & diversity | O(n²) |
| **Se²** | Sequential beam search | Order-aware selection | O(k²·b) |
| **Influence** | Influence function scoring | Gradient-based importance | O(n·p) |

### Baseline Methods

- **Random**: Random k examples (lower bound)
- **BM25**: Classical IR retrieval
- **SBERT**: Semantic similarity via sentence embeddings
- **kNN**: k-Nearest neighbors in embedding space

### References

- **TopK + CoNE**: Peng et al. "CoNE: Information-Theoretic Context Selection"
- **IDS**: Qin et al. "Iterative Demonstration Selection using CoT"
- **RDES**: Wang et al. "RL for Demonstration Selection"
- **Se²**: Lu et al. "Sequential Example Selection"
- **Influence**: Nguyen & Wong "Influence-based Selection"

---

## 🤖 Models & Datasets

### Supported Models

| Model | Type | Provider | Context | Cost | Access |
|-------|------|----------|---------|------|--------|
| **GPT-4o-mini** | API | OpenAI | 128K | $0.15/1M input | API key |
| **GPT-3.5-turbo** | API | OpenAI | 16K | $0.50/1M input | API key |
| **LLaMA-3.2-3B** | Local | Meta | 128K | Free* | HF token |
| **Gemma-2B** | Local | Google | 8K | Free* | HF token |
| **GPT-2-medium** | Local | OpenAI | 1K | Free | Public |

*Requires GPU for inference

### Datasets

| Dataset | Task | Classes | Train | Test | Metric |
|---------|------|---------|-------|------|--------|
| **SST-5** | Sentiment Classification | 5 | 8,544 | 2,210 | Acc, F1 |
| **AG News** | Topic Classification | 4 | 120K | 7,600 | Acc, F1 |
| **CommonsenseQA** | Multiple-Choice QA | 5 | 9,741 | 1,221 | Acc |

---

## 🔬 Running Experiments

### 1. Single Method Evaluation

```bash
# Run IDS on SST-5 with GPT-4o-mini
python experiments/run_ids.py \\
    --dataset sst5 \\
    --model gpt-4o-mini \\
    --k 5 \\
    --num_test 500 \\
    --num_train 2000 \\
    --seed 42
```

### 2. Full Benchmark (All Methods × Models × Datasets)

```bash
python experiments/run_benchmark.py \\
    --config configs/experiments.yaml \\
    --benchmark full_benchmark \\
    --output results/full_benchmark.json \\
    --n_runs 3
```

This runs:
- 5 methods × 3 models × 3 datasets = 45 experiments
- 3 random seeds for statistical confidence
- Saves detailed results + summary statistics

### 3. Ablation Study (Vary k-shot)

```bash
python experiments/run_benchmark.py \\
    --config configs/experiments.yaml \\
    --benchmark ablation_study \\
    --output results/ablation.json
```

### 4. Custom Experiment

Create `my_experiment.yaml`:

```yaml
datasets: ["sst5", "agnews"]
models: ["gpt-4o-mini"]
methods: ["ids", "topk_cone", "random"]
k: 5
num_test_samples: 1000
num_train_samples: 5000
num_runs: 5
seed: 42
```

Run:

```bash
python experiments/run_benchmark.py --config my_experiment.yaml
```

---

## 📊 Results

### Result Files

After running experiments, results are saved in:

```
results/
├── raw/
│   └── ids_gpt4o_sst5_20240306.json       # Raw predictions
├── processed/
│   └── benchmark_summary.csv              # Aggregated metrics
└── plots/
    ├── accuracy_heatmap.png               # Method × Dataset heatmap
    ├── method_comparison_boxplot.png      # Statistical comparison
    └── cost_vs_accuracy.png               # Efficiency analysis
```

### Sample Results

| Method | SST-5 | AG News | CSQA | Average | Rank |
|--------|-------|---------|------|---------|------|
| **IDS** | 0.82±0.02 | 0.76±0.01 | 0.68±0.03 | **0.75** | 1 |
| **TopK+CoNE** | 0.79±0.02 | 0.78±0.02 | 0.65±0.02 | **0.74** | 2 |
| **Se²** | 0.77±0.03 | 0.75±0.02 | 0.67±0.02 | **0.73** | 3 |
| **SBERT** | 0.76±0.02 | 0.74±0.01 | 0.70±0.02 | **0.73** | 3 |
| **Random** | 0.65±0.04 | 0.62±0.03 | 0.58±0.04 | **0.62** | 5 |

*Results are Accuracy ± 95% CI across 3 runs*

### Key Findings

1. **IDS performs best on average** but requires multiple inference passes
2. **TopK+CoNE offers best speed/accuracy tradeoff**
3. **Task-specific winners**: SBERT excels on CSQA (semantic similarity matters)
4. **Model dependency**: Selection matters more for smaller models
5. **k-shot sensitivity**: Performance plateaus after k=5 for most methods

---

## 🛠️ Development Guide

### Adding a New Selection Method

```python
# 1. Create src/selection/my_method.py
from typing import List
import numpy as np

class MyMethod:
    def __init__(self, k: int = 5):
        self.k = k
    
    def select_demonstrations(
        self, 
        query: str,
        candidates: List[str]
    ) -> List[int]:
        # Your selection logic here
        selected_indices = ...
        return selected_indices

# 2. Add to src/selection/__init__.py
from .my_method import MyMethod
__all__ = [..., "MyMethod"]

# 3. Create experiment script
# experiments/run_my_method.py

# 4. Add config
# configs/experiments.yaml
```

### Adding a New Dataset

```python
# 1. Create src/datasets/load_mydataset.py
from datasets import load_dataset
from typing import List, Tuple

def load_mydataset(
    split: str = "train",
    num_samples: int = None
) -> Tuple[List[str], List[str]]:
    dataset = load_dataset("your/dataset", split=split)
    # ... processing
    return texts, labels

# 2. Add to configs/datasets.yaml
mydataset:
  name: "your/dataset"
  task_type: "classification"
  num_classes: 3
  ...
```

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Lint code
black src/
flake8 src/
mypy src/
```

### Code Quality

We use:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pytest** for testing

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{praveen2024optimal,
  title={Optimal Demonstration Selection for In-Context Learning},
  author={Praveen, Satvik and Tong, Jonathan and 
          Kamisetty, Yamini Preethi and Bandi, Vinay Chandra},
  journal={Texas A&M University},
  year={2024}
}
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow code style (run `black src/`)
4. Add tests for new functionality
5. Update documentation
6. Submit a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **HuggingFace** for Transformers and Datasets libraries
- **OpenAI** for GPT API access
- **Meta AI** for LLaMA models
- **Google** for Gemma models
- Research groups whose papers inspired this work

---

## 📧 Contact

For questions or collaboration:

- **Issues**: Open a GitHub issue
- **Email**: satvikpraveen_164@tamu.edu
- **Project**: https://github.com/SatvikPraveen/Optimal-Demo-Selection-ICL

---

## 🔗 Related Resources

### Papers
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [In-Context Learning Survey](https://arxiv.org/abs/2301.00234)
- [Learning to Retrieve Prompts](https://arxiv.org/abs/2112.08633)

### Repositories
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)
- [OpenICL](https://github.com/Shark-NLP/OpenICL)

---

**⭐ If you find this project useful, please consider giving it a star! ⭐**

