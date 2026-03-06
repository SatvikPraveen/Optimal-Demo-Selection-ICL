"""
CommonsenseQA Dataset Loader
"""

from datasets import load_dataset
from typing import List, Tuple, Dict, Optional


def load_commonsense_qa(
    split: str = "train",
    num_samples: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Load CommonsenseQA dataset.
    
    Args:
        split: Dataset split ('train', 'validation')
        num_samples: Number of samples to load (None = all)
        seed: Random seed for sampling
        
    Returns:
        texts: List of formatted question + choices
        labels: List of correct answers
    """
    if num_samples:
        ds = load_dataset("tau/commonsense_qa", split=f"{split}[:{num_samples}]")
    else:
        ds = load_dataset("tau/commonsense_qa", split=split)
    
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    
    questions = ds["question"]
    choices_all = ds["choices"]
    answer_keys = ds["answerKey"]
    
    texts = []
    labels = []
    
    for q, choice_dict, key in zip(questions, choices_all, answer_keys):
        # Format question with choices
        prompt = "Question: " + q + "\n" + "\n".join(
            f"{lbl}. {txt}"
            for lbl, txt in zip(choice_dict["label"], choice_dict["text"])
        )
        texts.append(prompt)
        
        # Get correct answer text
        idx = choice_dict["label"].index(key)
        labels.append(choice_dict["text"][idx])
    
    return texts, labels


def format_csqa_prompt(question: str, choices: Dict, answer: Optional[str] = None) -> str:
    """
    Format CommonsenseQA sample as prompt.
    
    Args:
        question: Question text
        choices: Dictionary with 'label' and 'text' lists
        answer: Correct answer letter (optional)
        
    Returns:
        Formatted prompt string
    """
    prompt = f"Question: {question}\n"
    prompt += "\n".join(
        f"{lbl}. {txt}"
        for lbl, txt in zip(choices["label"], choices["text"])
    )
    prompt += "\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt
