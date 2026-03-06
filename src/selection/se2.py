"""
Se² (Selective Annotation with Entropy-based Selection)

Placeholder for Se² implementation
"""

from typing import List
import numpy as np


class Se2:
    """
    Se² Selection Algorithm
    
    TODO: Implement Se² algorithm based on research paper
    """
    
    def __init__(
        self,
        k: int = 5,
        retrieve_k: int = 20,
        beam_size: int = 3
    ):
        """
        Initialize Se² selector.
        
        Args:
            k: Number of demonstrations to select
            retrieve_k: Number of candidates to retrieve
            beam_size: Beam size for selection
        """
        self.k = k
        self.retrieve_k = retrieve_k
        self.beam_size = beam_size
    
    def select_demonstrations(
        self,
        query: str,
        candidates: List[str]
    ) -> List[int]:
        """
        Select demonstrations using Se².
        
        Args:
            query: Test sample
            candidates: Pool of training examples
            
        Returns:
            Indices of selected demonstrations
        """
        # TODO: Implement Se² algorithm
        raise NotImplementedError("Se² algorithm needs to be implemented")
