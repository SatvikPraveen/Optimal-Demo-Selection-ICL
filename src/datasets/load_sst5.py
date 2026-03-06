"""
SST-5 Dataset Loader
"""

from datasets import load_dataset
from typing import List, Tuple, Dict, Optional


def load_sst5(
    split: str = "train",
    num_samples: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Load SST-5 sentiment dataset.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        num_samples: Number of samples to load (None = all)
        seed: Random seed for sampling
        
    Returns:
        texts: List of text samples
        labels: List of sentiment labels
    """
    dataset = load_dataset("SetFit/sst5", "default", split=split)
    dataset = dataset.filter(lambda x: x["label_text"] is not None)
    
    if num_samples:
        dataset = dataset.shuffle(seed=seed).select(range(min(num_samples, len(dataset))))
    
    texts = []
    labels = []
    
    for sample in dataset:
        text = sample["text"].strip().replace("\n", " ")
        label_text = sample["label_text"]
        texts.append(text)
        labels.append(label_text)
    
    return texts, labels


def format_sst5_prompt(text: str, label: Optional[str] = None) -> str:
    """
    Format SST-5 sample as prompt.
    
    Args:
        text: Input text
        label: Ground truth label (optional)
        
    Returns:
        Formatted prompt string
    """
    prompt = f"Text: {text}\nSentiment:"
    if label:
        prompt += f" {label}"
    return prompt
