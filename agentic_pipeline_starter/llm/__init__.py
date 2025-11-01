"""LLM abstraction layer for Ollama and Mock implementations."""

from .mock_llm import BaseLLM, MockLLM
from .ollama_client import OllamaLLM, LLMFactory

__all__ = [
    "BaseLLM",
    "MockLLM", 
    "OllamaLLM",
    "LLMFactory",
]