"""Ollama LLM client for local LLM integration.

This module provides OllamaLLM class for connecting to local Ollama models
for production use of the agentic pipeline.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
import httpx

from .mock_llm import BaseLLM
from ..config import get_settings


logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Ollama LLM client for local model integration.
    
    Connects to a local Ollama instance to generate responses using
    locally running language models.
    """
    
    def __init__(self, 
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout: Optional[int] = None):
        """Initialize the Ollama LLM client.
        
        Args:
            base_url: Ollama server URL (defaults to config)
            model: Model name to use (defaults to config)
            timeout: Request timeout in seconds (defaults to config)
        """
        self.settings = get_settings()
        self.base_url = base_url or self.settings.ollama_base_url
        self.model = model or self.settings.ollama_model
        self.timeout = timeout or self.settings.ollama_timeout
        
        # Ensure base_url has correct format
        if not self.base_url.endswith('/api'):
            self.base_url = f"{self.base_url.rstrip('/')}/api"
        
        logger.info(f"OllamaLLM initialized: {self.base_url}, model: {self.model}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama model.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response from Ollama
            
        Raises:
            RuntimeError: If Ollama request fails
            ConnectionError: If cannot connect to Ollama server
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    **kwargs
                }
                
                logger.debug(f"Sending request to Ollama: {self.base_url}/generate")
                
                response = await client.post(
                    f"{self.base_url}/generate",
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                if "response" not in result:
                    raise RuntimeError(f"Invalid Ollama response format: {result}")
                
                generated_text = result["response"].strip()
                
                if not generated_text:
                    raise RuntimeError("Ollama returned empty response")
                
                logger.debug(f"Ollama response length: {len(generated_text)} chars")
                return generated_text
                
        except httpx.ConnectError as e:
            error_msg = f"Cannot connect to Ollama server at {self.base_url}: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
            
        except httpx.TimeoutException as e:
            error_msg = f"Ollama request timed out after {self.timeout}s: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        except httpx.HTTPStatusError as e:
            error_msg = f"Ollama HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error calling Ollama: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
    
    async def health_check(self) -> bool:
        """Check if Ollama server is accessible and model is available.
        
        Returns:
            True if Ollama is healthy and model is available
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Check server health
                response = await client.get(f"{self.base_url}/tags")
                response.raise_for_status()
                
                # Check if our model is available
                models_data = response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
                
                if self.model not in available_models:
                    logger.warning(f"Model {self.model} not found in available models: {available_models}")
                    return False
                
                logger.info(f"Ollama health check passed for model {self.model}")
                return True
                
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            return False


class LLMFactory:
    """Factory class for creating LLM instances based on configuration."""
    
    @staticmethod
    def create_llm() -> BaseLLM:
        """Create LLM instance based on current settings.
        
        Returns:
            Configured LLM instance (MockLLM or OllamaLLM)
        """
        settings = get_settings()
        
        if settings.llm_mode.value == "mock":
            logger.info("Creating MockLLM for testing/development")
            from .mock_llm import MockLLM
            return MockLLM()
        
        elif settings.llm_mode.value == "ollama":
            logger.info("Creating OllamaLLM for production use")
            return OllamaLLM()
        
        else:
            logger.warning(f"Unknown LLM mode: {settings.llm_mode}, falling back to MockLLM")
            from .mock_llm import MockLLM
            return MockLLM()
    
    @staticmethod
    async def create_llm_with_health_check() -> BaseLLM:
        """Create LLM instance with health check for Ollama.
        
        Returns:
            Configured LLM instance, falls back to MockLLM if Ollama unhealthy
        """
        settings = get_settings()
        
        if settings.llm_mode.value == "ollama":
            ollama_llm = OllamaLLM()
            
            # Check if Ollama is healthy
            if await ollama_llm.health_check():
                logger.info("Ollama health check passed, using OllamaLLM")
                return ollama_llm
            else:
                logger.warning("Ollama health check failed, falling back to MockLLM")
                from .mock_llm import MockLLM
                return MockLLM()
        
        else:
            return LLMFactory.create_llm()