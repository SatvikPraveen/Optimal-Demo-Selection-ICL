"""
Base model interface
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseModel(ABC):
    """
    Abstract base class for language models.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize model.
        
        Args:
            model_name: Name/path of the model
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated texts
        """
        pass
