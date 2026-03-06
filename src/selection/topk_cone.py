"""
TopK + CoNE (Conditional Negative Log-Likelihood Evaluation)

Reference: Combining retrieval with cross-entropy based refinement
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional


class TopKCoNE:
    """
    TopK + CoNE demonstration selection
    
    Combines embedding-based retrieval with cross-entropy refinement.
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        raw_texts: List[str],
        k: int = 5,
        retrieve_k: int = 30,
        model_name: str = 'gpt2',
        device: Optional[str] = None
    ):
        """
        Initialize TopK+CoNE selector.
        
        Args:
            embeddings: Pre-computed embeddings for training samples
            raw_texts: Raw text of training samples
            k: Final number of demonstrations to select
            retrieve_k: Number of candidates to retrieve before CoNE
            model_name: Language model for CoNE scoring
            device: Device ('cuda', 'cpu', or None for auto)
        """
        self.embeddings = embeddings
        self.raw_texts = raw_texts
        self.k = k
        self.retrieve_k = retrieve_k
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def get_topk(self, query_embedding: np.ndarray) -> np.ndarray:
        """Retrieve top-k similar examples by embedding"""
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        topk_indices = np.argsort(similarities)[-self.retrieve_k:][::-1]
        return topk_indices
    
    def compute_cross_entropy(self, text: str) -> float:
        """Compute cross-entropy for text"""
        encodings = self.tokenizer(text, return_tensors='pt').to(self.device)
        input_ids = encodings['input_ids']
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
        
        loss = outputs.loss.item()
        return loss * input_ids.size(1)
    
    def apply_cone(self, query_text: str, topk_indices: np.ndarray) -> List[int]:
        """Apply CoNE to refine candidates"""
        candidate_scores = []
        
        for idx in topk_indices:
            demo = self.raw_texts[idx]
            prompt_with_query = demo + "\n" + query_text
            
            # Conditional cross-entropy
            H_xc = self.compute_cross_entropy(prompt_with_query)
            H_c = self.compute_cross_entropy(demo)
            H_cond = H_xc - H_c
            
            candidate_scores.append((idx, H_cond))
        
        # Select k demonstrations with lowest conditional entropy
        sorted_indices = [idx for idx, _ in sorted(candidate_scores, key=lambda x: x[1])]
        return sorted_indices[:self.k]
    
    def select_demonstrations(
        self,
        query_embedding: np.ndarray,
        query_text: str
    ) -> List[int]:
        """
        Select demonstrations using TopK + CoNE.
        
        Args:
            query_embedding: Embedding of the query
            query_text: Raw text of the query
            
        Returns:
            Indices of selected demonstrations
        """
        topk_indices = self.get_topk(query_embedding)
        refined_indices = self.apply_cone(query_text, topk_indices)
        return refined_indices
