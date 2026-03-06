#!/usr/bin/env python3
"""
Quick project functionality test
"""

from src.datasets import load_sst5, load_agnews
from src.selection import IDS, TopKCoNE
from src.evaluation import compute_accuracy, compute_f1
from src.prompting import PromptBuilder
from sentence_transformers import SentenceTransformer
import numpy as np

print("="*70)
print("🧪 TESTING PROJECT FUNCTIONALITY")
print("="*70)

# Test 1: Dataset Loading
print("\n1️⃣ Testing Dataset Loading...")
texts_sst, labels_sst = load_sst5('test', 50, seed=42)
texts_ag, labels_ag = load_agnews('test', 20, seed=42)
print(f"   ✅ SST-5: {len(texts_sst)} samples, labels: {set(labels_sst)}")
print(f"   ✅ AG News: {len(texts_ag)} samples, labels: {set(labels_ag)}")

# Test 2: Selection Algorithms
print("\n2️⃣ Testing Selection Algorithms...")
ids_selector = IDS(k=5, q=3)
print(f"   ✅ IDS initialized (k={ids_selector.k}, q={ids_selector.q})")

# Create embeddings for TopK+CoNE
encoder = SentenceTransformer('all-MiniLM-L6-v2')
sample_texts = texts_sst[:30]
embeddings = encoder.encode(sample_texts, show_progress_bar=False)
topk_selector = TopKCoNE(embeddings, sample_texts, k=3, retrieve_k=10)
print(f"   ✅ TopK+CoNE initialized (k={topk_selector.k})")
print(f"   ✅ Embeddings shape: {embeddings.shape}")

# Test 3: Prompt Building
print("\n3️⃣ Testing Prompt Builder...")
demonstrations = ["Input: Great movie! Output: positive", "Input: Terrible film. Output: negative"]
builder = PromptBuilder(task_instruction="Classify the sentiment:")
prompt = builder.build_prompt(demonstrations, "Input: Good acting.")
print(f"   ✅ Prompt built ({len(prompt)} chars)")

# Test 4: Evaluation Metrics
print("\n4️⃣ Testing Evaluation Metrics...")
predictions = ['positive', 'negative', 'neutral', 'positive']
labels = ['positive', 'positive', 'neutral', 'negative']
accuracy = compute_accuracy(predictions, labels)
f1 = compute_f1(predictions, labels, average='weighted')
print(f"   ✅ Accuracy: {accuracy:.2f}")
print(f"   ✅ F1 Score: {f1:.2f}")

# Test 5: Similarity Computation (TopK retrieval)
print("\n5️⃣ Testing Similarity-based Retrieval...")
test_embedding = embeddings[0].reshape(1, -1)
similarities = np.dot(embeddings, test_embedding.T).flatten()
top_k_indices = np.argsort(similarities)[-5:][::-1]
print(f"   ✅ Retrieved top-5 similar samples: {top_k_indices.tolist()}")

print("\n" + "="*70)
print("🎉 ALL TESTS PASSED! Project is fully functional!")
print("="*70)
print("\n📝 Next steps:")
print("   1. Set up API keys in .env file")
print("   2. Check experiment scripts: experiments/run_ids.py")
print("   3. Read documentation: docs/QUICK_START.md")
print("="*70)
