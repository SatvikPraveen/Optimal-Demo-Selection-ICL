"""
OpenAI GPT model interface
"""

from typing import List, Optional
from .base import BaseModel
from openai import OpenAI


class GPTModel(BaseModel):
    """
    Interface for OpenAI GPT models (GPT-3.5, GPT-4, etc.)
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize GPT model.
        
        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            **kwargs: Additional arguments
        """
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(api_key=api_key)
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text using GPT.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional OpenAI API arguments
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content
    
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
            **kwargs: Additional arguments
            
        Returns:
            List of generated texts
        """
        return [
            self.generate(prompt, max_tokens, temperature, **kwargs)
            for prompt in prompts
        ]
