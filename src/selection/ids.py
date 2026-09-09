"""
Iterative Demonstration Selection (IDS)

Reference: Based on the IDS algorithm for demonstration selection
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Callable, Optional


class IDS:
    """
    Iterative Demonstration Selection
    
    Selects demonstrations by iteratively refining based on reasoning paths.
    """
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        k: int = 4,
        q: int = 3,
        device: Optional[str] = None
    ):
        """
        Initialize IDS selector.
        
        Args:
            embedding_model: Name of sentence transformer model
            k: Number of demonstrations to select
            q: Number of iterations
            device: Device for embeddings ('cuda', 'cpu', or None for auto)
        """
        self.k = k
        self.q = q
        self.model = SentenceTransformer(embedding_model, device=device)
        self._cached_train_samples: Optional[List[str]] = None
        self._cached_train_embeddings: Optional[np.ndarray] = None

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using SentenceBERT"""
        return self.model.encode(text)

    def precompute_train_embeddings(self, train_samples: List[str]) -> np.ndarray:
        """
        Compute and cache embeddings for a training pool once, so repeated
        calls to select_demonstrations() over the same pool (e.g. once per
        test example) don't re-embed it every time.
        """
        self._cached_train_embeddings = np.array(
            [self.encode_text(sample) for sample in train_samples]
        )
        self._cached_train_samples = train_samples
        return self._cached_train_embeddings
    
    def select_top_k(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        k: int
    ) -> np.ndarray:
        """Select top-k most similar examples"""
        similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices
    
    def select_demonstrations(
        self,
        test_sample: str,
        train_samples: List[str],
        zero_shot_cot_fn: Callable[[str], str],
        icl_fn: Callable[[str, List[str]], str],
        precomputed_train_embeddings: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Select demonstrations for a test sample using IDS.

        Args:
            test_sample: The test sample to select demonstrations for
            train_samples: Pool of training examples
            zero_shot_cot_fn: Function for zero-shot chain-of-thought
            icl_fn: Function for in-context learning
            precomputed_train_embeddings: Optional embeddings for train_samples
                from precompute_train_embeddings(), to avoid re-encoding the
                same pool on every call (e.g. once per test example in a loop)

        Returns:
            Indices of selected demonstrations
        """
        if precomputed_train_embeddings is not None:
            train_embeddings = precomputed_train_embeddings
        elif self._cached_train_samples is train_samples and self._cached_train_embeddings is not None:
            train_embeddings = self._cached_train_embeddings
        else:
            train_embeddings = self.precompute_train_embeddings(train_samples)

        # Initial reasoning path
        reasoning_path = zero_shot_cot_fn(test_sample)
        
        # Iteratively refine selection
        for _ in range(self.q):
            query_embedding = self.encode_text(reasoning_path)
            selected_indices = self.select_top_k(query_embedding, train_embeddings, self.k)
            demonstrations = [train_samples[i] for i in selected_indices]
            
            # Get new reasoning path
            reasoning_path = icl_fn(test_sample, demonstrations)
        
        return selected_indices.tolist()
