"""
Gemma model interface
"""

from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseModel


class GemmaModel(BaseModel):
    """
    Interface for Google Gemma models
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-2b",
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Gemma model.
        
        Args:
            model_name: HuggingFace model name
            device: Device ('cuda', 'cpu', or None for auto)
            **kwargs: Additional arguments for model loading
        """
        super().__init__(model_name, **kwargs)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            **kwargs
        )
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text using Gemma.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove input prompt from output
        generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
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
        # Simple sequential generation (can optimize with batching later)
        return [
            self.generate(prompt, max_tokens, temperature, **kwargs)
            for prompt in prompts
        ]
