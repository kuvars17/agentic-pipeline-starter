"""Base LLM interface for the agentic pipeline.

This module defines the abstract base class that all LLM implementations
must inherit from to ensure consistent interface across different providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseLLM(ABC):
    """Abstract base class for LLM implementations.
    
    This class defines the interface that all LLM implementations must follow,
    ensuring consistency across different providers (OpenAI, Anthropic, local models, etc.).
    """
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response from the LLM.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters specific to the implementation
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM service is available and responsive.
        
        Returns:
            True if the LLM is available, False otherwise
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model information (name, version, capabilities, etc.)
        """
        return {
            "name": getattr(self, "model_name", "unknown"),
            "type": self.__class__.__name__,
            "capabilities": getattr(self, "capabilities", [])
        }