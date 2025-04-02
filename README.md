# Optimal-Demo-Selection-ICL
Implements and benchmarks optimal demonstration selection strategies in In-Context Learning using LLMs. Implements IDS, RDES, Se², TopK+ConE, and Influence-based methods to analyze the role of relevance, diversity, and ordering across various tasks and datasets.

# Optimal Demonstration Selection Techniques in In-Context Learning

This repository supports research on demonstration selection strategies for In-Context Learning (ICL) using Large Language Models (LLMs). It implements and benchmarks diverse strategies to study how relevance, diversity, and example ordering affect model performance across classification, reasoning, and QA tasks.

## 🧠 Selection Strategies Implemented

- **TopK + ConE** – Measures how each demo improves model understanding (Peng et al.)
- **IDS (Iterative Demonstration Selection)** – Uses Zero-Shot CoT reasoning to iteratively refine selection (Qin et al.)
- **RDES (Relevance-Diversity Enhanced Selection)** – Reinforcement learning-based strategy optimizing diversity and relevance (Wang et al.)
- **Se² (Sequential-aware Selection)** – Dynamically constructs prompt sequences to improve contextual alignment (Lu et al.)
- **Influence-based Selection** – Selects examples based on their influence on task performance (Nguyen & Wong, 2023)

## 🎯 Project Goals

- Evaluate LLM performance across multiple selection strategies
- Analyze the trade-offs between similarity, diversity, and order in demonstration selection
- Understand task-model-strategy interactions on sentiment, reasoning, and classification benchmarks

## 🤖 Models Used

- **LLaMA 3.2 8B** – Meta AI's open-source model
- **GPT-2** – Baseline GPT family model for comparison
- **Deepseek-LLM 7B Base** – Recently released model for broader evaluation

## 📊 Datasets Used

- **SST-5** – Sentiment analysis
- **AGNews / CR / Subj / MNLI / QNLI** – Topic classification & natural language inference
- **CommonsenseQA / GSM8K / LogiQA / BoolQ** – Commonsense, logical, and math reasoning tasks

## 📁 Folder Structure

```bash
Optimal-Demo-Selection-ICL/
│
├── README.md                 # Project overview and instructions
├── requirements.txt          # Python dependencies
├── setup.sh                  # Optional: environment setup script
├── LICENSE                   # License information
│
├── data/
│   ├── raw/                  # Original datasets (e.g., SST-5, AGNews, BoolQ)
│   ├── processed/            # Preprocessed/tokenized datasets
│   └── download_scripts/     # Scripts to download and prepare datasets
│
├── models/
│   ├── llama/                # LLaMA 3.2 loading and inference utilities
│   ├── gpt2/                 # GPT-2 loading and inference utilities
│   ├── deepseek/             # Deepseek-LLM model wrapper and setup
│   └── base_model.py         # Common interface for all models
│
├── strategies/
│   ├── topk_cone/            # Implementation of TopK + ConE (Peng et al.)
│   ├── ids/                  # Iterative Demonstration Selection (Qin et al.)
│   ├── rdes/                 # Relevance-Diversity Enhanced Selection (Wang et al.)
│   ├── se2/                  # Sequential Example Selection (Lu et al.)
│   ├── influence/            # Influence-based selection (Nguyen & Wong)
│   └── utils.py              # Common utilities: TF-IDF, cosine similarity, CoT tools
│
├── evaluations/
│   ├── classification/
│   │   └── run_eval_classification.py   # Accuracy, F1, etc.
│   ├── reasoning/
│   │   └── run_eval_reasoning.py        # Evaluation for GSM8K, CommonsenseQA, etc.
│   └── metrics.py                       # Shared evaluation metrics and helpers
│
├── notebooks/
│   ├── exploratory.ipynb               # Dataset/model exploration
│   ├── strategy_comparison.ipynb       # Compare strategies across tasks
│   └── influence_analysis.ipynb        # Influence score visualization and insight
│
└── experiments/
    ├── configs/
    │   └── experiment_config.yaml      # Task + model + strategy parameters
    ├── logs/                           # Save outputs and evaluation logs
    ├── run_experiment.py               # Script to run individual experiments
    └── run_all.sh                      # Bash script to run all experiments sequentially
```

## 📦 Installation

```bash
pip install -r requirements.txt
```

▶️ Running Experiments
```bash
bash experiments/run_all.sh
```
