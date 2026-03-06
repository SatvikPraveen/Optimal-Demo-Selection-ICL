"""
Example experiment: Run TopK+CoNE on a dataset
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

from src.datasets import load_sst5, load_agnews
from src.models import GPTModel
from src.selection import TopKCoNE
from src.prompting import ICLInference
from src.evaluation import compute_metrics
from src.utils import set_seed, setup_logger
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Run TopK+CoNE experiment")
    parser.add_argument("--dataset", type=str, default="sst5", 
                       choices=["sst5", "agnews"], help="Dataset name")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--k", type=int, default=5, help="Number of demonstrations")
    parser.add_argument("--retrieve_k", type=int, default=30, help="Candidates to retrieve")
    parser.add_argument("--num_test", type=int, default=100, help="Number of test samples")
    parser.add_argument("--num_train", type=int, default=1000, help="Number of train samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/raw/topk_cone_experiment.json")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    logger = setup_logger("topk_cone_experiment")
    
    logger.info(f"Starting TopK+CoNE experiment with k={args.k}")
    logger.info(f"Dataset: {args.dataset}, Model: {args.model}")
    
    # Load data
    logger.info(f"Loading {args.dataset} dataset...")
    if args.dataset == "sst5":
        train_texts, train_labels = load_sst5(
            split="train", num_samples=args.num_train, seed=args.seed
        )
        test_texts, test_labels = load_sst5(
            split="test", num_samples=args.num_test, seed=args.seed
        )
    elif args.dataset == "agnews":
        train_texts, train_labels = load_agnews(
            split="train", num_samples=args.num_train, seed=args.seed
        )
        test_texts, test_labels = load_agnews(
            split="test", num_samples=args.num_test, seed=args.seed
        )
    
    # Compute embeddings
    logger.info("Computing embeddings for training set...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    train_embeddings = embedder.encode(train_texts, show_progress_bar=True)
    
    # Initialize model
    logger.info(f"Initializing model: {args.model}")
    model = GPTModel(model_name=args.model)
    
    # Initialize TopK+CoNE
    logger.info("Initializing TopK+CoNE selector...")
    selector = TopKCoNE(
        embeddings=train_embeddings,
        raw_texts=train_texts,
        k=args.k,
        retrieve_k=args.retrieve_k
    )
    inference = ICLInference(model=model)
    
    # Run experiments
    logger.info("Running TopK+CoNE selection...")
    predictions = []
    
    for i, (test_text, test_label) in enumerate(zip(test_texts, test_labels)):
        if i % 10 == 0:
            logger.info(f"Processing sample {i}/{len(test_texts)}")
        
        try:
            # Compute query embedding
            query_embedding = embedder.encode([test_text])[0]
            
            # Select demonstrations
            demo_indices = selector.select_demonstrations(query_embedding, test_text)
            
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
        "experiment": "topk_cone",
        "dataset": args.dataset,
        "model": args.model,
        "k": args.k,
        "retrieve_k": args.retrieve_k,
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
