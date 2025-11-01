"""Mock LLM implementation for deterministic testing.

This module provides a MockLLM class that returns predefined responses
for testing and development without requiring external LLM services.
"""

import asyncio
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response from the LLM.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass


class MockLLM(BaseLLM):
    """Mock LLM for deterministic testing and development.
    
    Returns predefined responses based on prompt content for consistent testing.
    """
    
    def __init__(self):
        """Initialize the mock LLM with predefined responses."""
        self._responses = self._get_mock_responses()
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response based on prompt content.
        
        Args:
            prompt: Input prompt text
            **kwargs: Ignored for mock implementation
            
        Returns:
            Predefined mock response based on prompt content
        """
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Return response based on prompt content
        prompt_lower = prompt.lower()
        
        if "plan" in prompt_lower and "query" in prompt_lower:
            return self._get_planning_response(prompt)
        elif "judge" in prompt_lower or "evaluate" in prompt_lower:
            return "High confidence: The information appears accurate and well-sourced."
        elif "report" in prompt_lower or "summarize" in prompt_lower:
            return "Based on the analysis, here is a comprehensive summary of the findings."
        else:
            return "I understand your request and will provide a helpful response."
    
    def _get_planning_response(self, prompt: str) -> str:
        """Generate planning response based on query content.
        
        Args:
            prompt: The planning prompt
            
        Returns:
            Structured planning response
        """
        # Extract query from prompt for context-aware responses
        if "capital" in prompt.lower():
            return """1. Search for official information about the country's capital city
2. Verify the information from reliable government or educational sources
3. Confirm the current status (as capitals can change historically)
4. Provide the definitive answer with confidence"""
        
        elif "math" in prompt.lower() or "calculate" in prompt.lower():
            return """1. Parse the mathematical expression or problem
2. Identify the required mathematical operations
3. Perform the calculations step by step
4. Verify the result and provide the answer"""
        
        elif "weather" in prompt.lower():
            return """1. Identify the specific location requested
2. Access current weather data from reliable meteorological sources
3. Gather relevant weather metrics (temperature, conditions, etc.)
4. Present the weather information clearly"""
        
        else:
            return """1. Analyze the user's question to understand the core request
2. Identify the key information needed to provide a complete answer
3. Gather relevant data from appropriate sources
4. Synthesize the information into a comprehensive response"""
    
    def _get_mock_responses(self) -> Dict[str, str]:
        """Get predefined responses for different scenarios.
        
        Returns:
            Dictionary of scenario -> response mappings
        """
        return {
            "default_plan": """1. Analyze the user's question to understand the core request
2. Identify the key information needed to provide a complete answer
3. Gather relevant data from appropriate sources
4. Synthesize the information into a comprehensive response""",
            
            "capital_query": """1. Search for official information about the country's capital city
2. Verify the information from reliable government or educational sources
3. Confirm the current status (as capitals can change historically)
4. Provide the definitive answer with confidence""",
            
            "math_query": """1. Parse the mathematical expression or problem
2. Identify the required mathematical operations
3. Perform the calculations step by step
4. Verify the result and provide the answer""",
            
            "error_response": "I apologize, but I'm unable to generate a plan for this query at the moment.",
        }