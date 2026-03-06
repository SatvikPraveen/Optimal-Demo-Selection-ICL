"""
ICL inference utilities
"""

from typing import List, Dict, Optional
from ..models import BaseModel
from .prompt_builder import PromptBuilder


class ICLInference:
    """
    In-context learning inference engine
    """
    
    def __init__(
        self,
        model: BaseModel,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        """
        Initialize ICL inference.
        
        Args:
            model: Language model for inference
            prompt_builder: Prompt builder (default: basic builder)
        """
        self.model = model
        self.prompt_builder = prompt_builder or PromptBuilder()
    
    def run_icl(
        self,
        demonstrations: List[str],
        query: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Run in-context learning inference.
        
        Args:
            demonstrations: List of demonstration examples
            query: Query to answer
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
            
        Returns:
            Model prediction
        """
        prompt = self.prompt_builder.build_prompt(demonstrations, query)
        prediction = self.model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return prediction
    
    def run_zero_shot_cot(
        self,
        query: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Run zero-shot chain-of-thought.
        
        Args:
            query: Query to answer
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
            
        Returns:
            Model reasoning and prediction
        """
        prompt = self.prompt_builder.build_zero_shot_cot_prompt(query)
        response = self.model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response
