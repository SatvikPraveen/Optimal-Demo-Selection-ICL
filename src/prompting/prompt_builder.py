"""
Prompt construction utilities
"""

from typing import List, Optional


class PromptBuilder:
    """
    Builder for ICL prompts
    """
    
    def __init__(
        self,
        task_instruction: Optional[str] = None,
        separator: str = "\n\n"
    ):
        """
        Initialize prompt builder.
        
        Args:
            task_instruction: Optional task instruction/description
            separator: Separator between demonstrations
        """
        self.task_instruction = task_instruction
        self.separator = separator
    
    def build_prompt(
        self,
        demonstrations: List[str],
        query: str,
        include_instruction: bool = True
    ) -> str:
        """
        Build ICL prompt from demonstrations and query.
        
        Args:
            demonstrations: List of demonstration examples
            query: Query/test example
            include_instruction: Whether to include task instruction
            
        Returns:
            Formatted prompt string
        """
        parts = []
        
        if include_instruction and self.task_instruction:
            parts.append(self.task_instruction)
        
        if demonstrations:
            parts.append(self.separator.join(demonstrations))
        
        parts.append(query)
        
        return self.separator.join(parts)
    
    def build_zero_shot_cot_prompt(self, query: str) -> str:
        """
        Build zero-shot chain-of-thought prompt.
        
        Args:
            query: Query/test example
            
        Returns:
            Zero-shot CoT prompt
        """
        cot_suffix = "Let's think step by step."
        return f"{query}\n\n{cot_suffix}"
