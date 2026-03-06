"""
AG News Dataset Loader
"""

import pandas as pd
from typing import List, Tuple, Dict, Optional


def load_agnews(
    split: str = "train",
    num_samples: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Load AG News topic classification dataset.
    
    Args:
        split: Dataset split ('train', 'test')
        num_samples: Number of samples to load (None = all)
        seed: Random seed for sampling
        
    Returns:
        texts: List of news texts
        labels: List of topic labels
    """
    label_dict = {0: "World", 1: "Sports", 2: "Business", 3: "Technology"}
    
    splits = {
        'train': 'data/train-00000-of-00001.parquet',
        'test': 'data/test-00000-of-00001.parquet'
    }
    
    df = pd.read_parquet(f"hf://datasets/wangrongsheng/ag_news/{splits[split]}")
    
    if num_samples:
        df = df.sample(n=min(num_samples, len(df)), random_state=seed).reset_index(drop=True)
    
    texts = df['text'].tolist()
    labels = [label_dict[label] for label in df['label'].tolist()]
    
    return texts, labels


def format_agnews_prompt(text: str, label: Optional[str] = None) -> str:
    """
    Format AG News sample as prompt.
    
    Args:
        text: Input news text
        label: Ground truth topic label (optional)
        
    Returns:
        Formatted prompt string
    """
    prompt = f"Text: {text}\nTopic:"
    if label:
        prompt += f" {label}"
    return prompt
