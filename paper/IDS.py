import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Initialize SentenceBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to encode text using SentenceBERT
def encode_text(text):
    return model.encode(text)

# Function to select top-k similar examples
def select_top_k(query_embedding, candidate_embeddings, k):
    similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return top_k_indices

# Function to apply Zero-shot-CoT
def zero_shot_cot(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}\n\nLet's think step by step."}
        ]
    )
    return response.choices[0].message['content']

# Main IDS function
def iterative_demonstration_selection(test_sample, train_samples, k=4, q=3):
    # Encode training samples
    train_embeddings = np.array([encode_text(sample) for sample in train_samples])
    
    answers = []
    reasoning_path = zero_shot_cot(test_sample)
    
    for _ in range(q):
        # Select demonstrations
        query_embedding = encode_text(reasoning_path)
        selected_indices = select_top_k(query_embedding, train_embeddings, k)
        demonstrations = [train_samples[i] for i in selected_indices]
        
        # Perform ICL
        prompt = f"Examples:\n" + "\n".join(demonstrations) + f"\n\nNow, answer this:\n{test_sample}\n\nLet's think step by step."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message['content']
        answers.append(answer)
        
        # Extract new reasoning path
        reasoning_path = answer.split("Therefore,")[0] if "Therefore," in answer else answer
    
    # Majority voting
    final_answer = max(set(answers), key=answers.count)
    return final_answer

# Example usage
test_sample = "What is the capital of France?"
train_samples = ["The capital of Germany is Berlin.", "London is the capital of the UK.", "Washington D.C. is the capital of the USA."]

result = iterative_demonstration_selection(test_sample, train_samples)
print(result)
