Certainly! Here’s an updated README incorporating your experiment results tables and figures. Feel free to adjust paths or styling as needed.

---

# Optimal-Demo-Selection-ICL

Implements and benchmarks optimal demonstration selection strategies in In-Context Learning (ICL) using large language models (LLMs). Strategies covered: IDS, RDES, Se², TopK+ConE, and Influence-based methods, evaluated across classification, reasoning, and QA tasks.

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Selection Strategies](#selection-strategies)  
3. [Models & Datasets](#models--datasets)  
4. [Experiment Results](#experiment-results)  
   - Se² Across Models  
   - Se² with LLaMA-3.2-3B (CommonsenseQA, AG News, SST-5)  
5. [Usage](#usage)  
6. [Repository Structure](#repository-structure)  

---

## Project Overview

This repository supports research on demonstration selection for ICL. We investigate how relevance, diversity, and ordering of examples affect LLM performance.

---

## Selection Strategies

- **TopK + ConE** — Quantifies each example’s information gain (Peng et al.).  
- **IDS** — Iterative refinement via zero-shot Chain-of-Thought (Qin et al.).  
- **RDES** — Reinforcement-learning to balance relevance & diversity (Wang et al.).  
- **Se²** — Sequential-aware selection with beam search (Lu et al.).  
- **Influence** — Selects examples by their influence score on model outputs (Nguyen & Wong, 2023).  

---

## Models & Datasets

**Models**  
- LLaMA-3.2-8B  
- GPT-2 (medium)  
- Deepseek-LLM-7B  

**Datasets**  
- **SST-5**: 5-class sentiment classification  
- **AG News**: 4-way topic classification  
- **CommonsenseQA**: multiple-choice QA  
- (Also: CR, Subj, MNLI, QNLI, GSM8K, LogiQA, BoolQ)

---

## Experiment Results

### 1. Se² Performance Across Model Architectures

| Model         | CommonsenseQA | AG News | SST-5 |
|---------------|--------------:|--------:|------:|
| GPT-Neo-1.3B   |        0.223  |   0.698 | 0.394 |
| GEMMA-2B      |        0.211  |   0.825 | 0.258 |
| GPT-2-medium  |        0.196  |   0.581 | 0.263 |

![Se2 Across Models](strategies/se2/plots/Se2_Other_models.png)

*Table:* Average Se² accuracy over three 200-example splits.  
*Figure:* Bar chart of Se² performance on CommonsenseQA, AG News, and SST-5.

---

### 2. LLaMA-3.2-3B Few-shot Results under Se²

#### CommonsenseQA

| Shot \& Beam | 1     | 2     | 3     |
|-------------:|------:|------:|------:|
| 1-shot       | 0.108 | 0.098 | 0.091 |
| 2-shot       | 0.108 | 0.140 | 0.179 |
| 3-shot       | 0.079 | 0.060 | 0.080 |

![CommonsenseQA](strategies/se2/plots/Se2_llama_CommonsenseQA.png)

#### AG News

| Shot \& Beam |   1   |   2   |   3   |
|-------------:|------:|------:|------:|
| 1-shot       | 0.748 | 0.729 | 0.724 |
| 2-shot       | 0.731 | 0.748 | 0.759 |
| 3-shot       | 0.727 | 0.765 | 0.786 |

![AG News](strategies/se2/plots/Se2_llama_AG_News.png)

#### SST-5

| Shot \& Beam |  1    |  2    |  3    |
|-------------:|------:|------:|------:|
| 1-shot       | 0.339 | 0.363 | 0.358 |
| 2-shot       | 0.361 | 0.387 | 0.382 |
| 3-shot       | 0.394 | 0.352 | 0.396 |

![SST-5](strategies/se2/plots/Se2_llama_SST5.png)

*All values averaged over three random splits.*

---

## Usage

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
2. **Run all experiments**  
   ```bash
   bash experiments/run_all.sh
   ```
3. **Visualize specific strategy**  
   ```bash
   python notebooks/strategy_comparison.ipynb
   ```

---

## Repository Structure

```
Optimal-Demo-Selection-ICL/
├── README.md
├── requirements.txt
├── data/                        # raw & processed datasets
├── models/                      # model wrappers (LLAMA, GPT2, Deepseek)
├── strategies/
│   ├── ids/
│   ├── rdes/
│   ├── se2/                     # includes plots/ with Se² figures
│   ├── topk_cone/
│   └── influence/
├── evaluations/                 # evaluation scripts & metrics
├── notebooks/                   # exploratory and comparison notebooks
└── experiments/
    ├── configs/
    ├── logs/
    ├── run_experiment.py
    └── run_all.sh
```
