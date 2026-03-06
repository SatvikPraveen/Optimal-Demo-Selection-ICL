#!/usr/bin/env python3
"""
Simple functional test without API keys
Tests core functionality of the package
"""

import sys
from pathlib import Path

print("=" * 70)
print("🧪 Running Functional Tests")
print("=" * 70)

# Test 1: Import all modules
print("\n✅ Test 1: Importing modules...")
try:
    from src.datasets import load_sst5, load_agnews
    from src.selection import TopKCoNE, IDS
    from src.evaluation import compute_metrics, compute_accuracy
    from src.utils import set_seed, setup_logger
    from src.prompting import PromptBuilder
    print("   ✅ All modules imported successfully")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Seed control
print("\n✅ Test 2: Testing seed control...")
try:
    set_seed(42)
    import numpy as np
    import random
    
    # Generate some random numbers
    np1 = np.random.rand(5)
    r1 = random.random()
    
    # Reset seed
    set_seed(42)
    np2 = np.random.rand(5)
    r2 = random.random()
    
    assert np.allclose(np1, np2), "NumPy random state not reproducible"
    assert r1 == r2, "Python random state not reproducible"
    print("   ✅ Seed control works correctly")
except Exception as e:
    print(f"   ❌ Seed test failed: {e}")
    sys.exit(1)

# Test 3: Prompt builder
print("\n✅ Test 3: Testing prompt builder...")
try:
    builder = PromptBuilder(task_instruction="Classify sentiment")
    
    demos = [
        "Text: Great movie! Sentiment: positive",
        "Text: Terrible film. Sentiment: negative"
    ]
    query = "Text: Amazing experience! Sentiment:"
    
    prompt = builder.build_prompt(demos, query)
    assert "Classify sentiment" in prompt
    assert "Great movie" in prompt
    assert "Amazing experience" in prompt
    print("   ✅ Prompt builder works correctly")
except Exception as e:
    print(f"   ❌ Prompt builder test failed: {e}")
    sys.exit(1)

# Test 4: Evaluation metrics
print("\n✅ Test 4: Testing evaluation metrics...")
try:
    predictions = ["positive", "negative", "positive", "neutral"]
    labels = ["positive", "positive", "positive", "neutral"]
    
    accuracy = compute_accuracy(predictions, labels)
    metrics = compute_metrics(predictions, labels)
    
    assert 0.0 <= accuracy <= 1.0, "Accuracy out of range"
    assert accuracy == 0.75, f"Expected 0.75, got {accuracy}"
    assert 'accuracy' in metrics
    assert 'f1' in metrics
    print(f"   ✅ Metrics computed correctly (accuracy: {accuracy:.2f})")
except Exception as e:
    print(f"   ❌ Metrics test failed: {e}")
    sys.exit(1)

# Test 5: Logger
print("\n✅ Test 5: Testing logger...")
try:
    logger = setup_logger("test", level=30)  # WARNING level
    logger.info("This should not appear")
    logger.warning("This warning should appear")
    print("   ✅ Logger configured correctly")
except Exception as e:
    print(f"   ❌ Logger test failed: {e}")
    sys.exit(1)

# Test 6: TopK+CoNE (basic initialization)
print("\n✅ Test 6: Testing TopK+CoNE initialization...")
try:
    import numpy as np
    
    # Create dummy data
    embeddings = np.random.rand(10, 384)  # 10 samples, 384-dim embeddings
    texts = [f"Sample text {i}" for i in range(10)]
    
    selector = TopKCoNE(
        embeddings=embeddings,
        raw_texts=texts,
        k=3,
        retrieve_k=5
    )
    
    print(f"   ✅ TopK+CoNE initialized (k={selector.k}, retrieve_k={selector.retrieve_k})")
except Exception as e:
    print(f"   ❌ TopK+CoNE test failed: {e}")
    sys.exit(1)

# Test 7: IDS initialization
print("\n✅ Test 7: Testing IDS initialization...")
try:
    selector = IDS(k=5, q=3)
    assert selector.k == 5
    assert selector.q == 3
    print(f"   ✅ IDS initialized (k={selector.k}, q={selector.q})")
except Exception as e:
    print(f"   ❌ IDS test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("🎉 All functional tests passed!")
print("=" * 70)
print("\n📝 Next steps:")
print("   1. Set up API keys in .env file")
print("   2. Try loading a real dataset:")
print("      python -c \"from src.datasets import load_sst5; texts, labels = load_sst5('test', 10); print(f'Loaded {len(texts)} samples')\"")
print("   3. Run a full experiment (requires API keys):")
print("      python experiments/run_ids.py --help")
print("=" * 70)
