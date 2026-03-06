"""
Example experiment: Run IDS on SST-5 dataset
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

from src.datasets import load_sst5
from src.models import GPTModel, LlamaModel, GemmaModel
from src.selection import IDS
from src.prompting import ICLInference, PromptBuilder
from src.evaluation import compute_metrics
from src.utils import set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run IDS experiment")
    parser.add_argument("--dataset", type=str, default="sst5", help="Dataset name")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--k", type=int, default=5, help="Number of demonstrations")
    parser.add_argument("--q", type=int, default=3, help="Number of IDS iterations")
    parser.add_argument("--num_test", type=int, default=100, help="Number of test samples")
    parser.add_argument("--num_train", type=int, default=1000, help="Number of train samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/raw/ids_experiment.json")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    logger = setup_logger("ids_experiment")
    
    logger.info(f"Starting IDS experiment with k={args.k}, q={args.q}")
    logger.info(f"Dataset: {args.dataset}, Model: {args.model}")
    
    # Load data
    logger.info(f"Loading {args.dataset} dataset...")
    train_texts, train_labels = load_sst5(split="train", num_samples=args.num_train, seed=args.seed)
    test_texts, test_labels = load_sst5(split="test", num_samples=args.num_test, seed=args.seed)
    
    # Initialize model
    logger.info(f"Initializing model: {args.model}")
    if "gpt" in args.model.lower():
        model = GPTModel(model_name=args.model)
    elif "llama" in args.model.lower():
        model = LlamaModel(model_name="meta-llama/Llama-3.2-3B-Instruct")
    elif "gemma" in args.model.lower():
        model = GemmaModel(model_name="google/gemma-2b")
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Initialize IDS
    logger.info("Initializing IDS selector...")
    selector = IDS(k=args.k, q=args.q)
    inference = ICLInference(model=model)
    
    # Run experiments
    logger.info("Running IDS selection and inference...")
    predictions = []
    
    for i, (test_text, test_label) in enumerate(zip(test_texts, test_labels)):
        if i % 10 == 0:
            logger.info(f"Processing sample {i}/{len(test_texts)}")
        
        try:
            # Select demonstrations
            demo_indices = selector.select_demonstrations(
                test_text,
                train_texts,
                zero_shot_cot_fn=lambda x: inference.run_zero_shot_cot(x),
                icl_fn=lambda q, demos: inference.run_icl(demos, q)
            )
            
            # Run ICL
            demos = [train_texts[idx] for idx in demo_indices]
            pred = inference.run_icl(demos, test_text)
            predictions.append(pred)
            
        except Exception as e:
            logger.error(f"Error on sample {i}: {e}")
            predictions.append("unknown")
    
    # Evaluate
    logger.info("Computing metrics...")
    metrics = compute_metrics(predictions, test_labels)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "experiment": "ids",
        "dataset": args.dataset,
        "model": args.model,
        "k": args.k,
        "q": args.q,
        "num_test": args.num_test,
        "num_train": args.num_train,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "predictions": predictions,
        "labels": test_labels
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"F1: {metrics['f1']:.3f}")
    logger.info("Experiment completed!")


if __name__ == "__main__":
    main()
