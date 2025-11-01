"""LLM Factory for creating appropriate LLM instances.

This module provides a factory pattern for creating LLM instances
based on configuration and available services.
"""

from typing import Optional
from .base import BaseLLM
from .mock_llm import MockLLM


class LLMFactory:
    """Factory for creating appropriate LLM instances."""
    
    @staticmethod
    def create_llm(provider: str = "mock", **kwargs) -> BaseLLM:
        """Create an LLM instance based on the provider.
        
        Args:
            provider: LLM provider ("mock", "ollama", etc.)
            **kwargs: Additional arguments for the LLM
            
        Returns:
            LLM instance
            
        Raises:
            ValueError: If provider is not supported
        """
        if provider.lower() == "mock":
            return MockLLM(**kwargs)
        elif provider.lower() == "ollama":
            # Import here to avoid circular imports
            try:
                from .ollama_client import OllamaLLM
                return OllamaLLM(**kwargs)
            except ImportError:
                # Fallback to MockLLM if OllamaLLM is not available
                return MockLLM(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> list[str]:
        """Get list of available LLM providers.
        
        Returns:
            List of provider names
        """
        providers = ["mock"]
        
        # Check if Ollama is available
        try:
            from .ollama_client import OllamaLLM
            providers.append("ollama")
        except ImportError:
            pass
            
        return providers