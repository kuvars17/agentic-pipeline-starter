"""Mock LLM implementation for deterministic testing.

This module provides a MockLLM class that returns predefined responses
for testing and development without requiring external LLM services.
"""

import asyncio
import hashlib
import re
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
    
    Returns deterministic responses based on prompt content for consistent testing.
    Integrates with the DeterministicMockResponses system for realistic plan generation.
    """
    
    def __init__(self, enable_deterministic_responses: bool = True):
        """Initialize the mock LLM.
        
        Args:
            enable_deterministic_responses: Whether to use deterministic response system
        """
        self.enable_deterministic_responses = enable_deterministic_responses
        self.model_name = "mock-llm-v1.0"
        self._fallback_responses = self._get_fallback_responses()
        
        # Import here to avoid circular imports
        if enable_deterministic_responses:
            try:
                from ..nodes.mock_responses import mock_responses
                self.mock_response_system = mock_responses
            except ImportError:
                self.mock_response_system = None
                self.enable_deterministic_responses = False
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response based on prompt content.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters (mock_delay, mock_error)
            
        Returns:
            Deterministic mock response based on prompt content
        """
        # Simulate processing time (configurable for testing)
        delay = kwargs.get('mock_delay', 0.1)
        await asyncio.sleep(delay)
        
        # Allow testing error scenarios
        if kwargs.get('mock_error'):
            raise RuntimeError(f"Mock error: {kwargs['mock_error']}")
        
        # Extract query from prompt for deterministic response
        query = self._extract_query_from_prompt(prompt)
        
        if self.enable_deterministic_responses and self.mock_response_system:
            # Use deterministic response system
            plan_steps = self.mock_response_system.get_mock_response_for_query(query)
            return self.mock_response_system.generate_mock_llm_response(query, plan_steps)
        else:
            # Use simple fallback system
            return self._get_simple_response(prompt)
    
    def _extract_query_from_prompt(self, prompt: str) -> str:
        """Extract the actual user query from the formatted prompt.
        
        Args:
            prompt: Full formatted prompt
            
        Returns:
            Extracted user query
        """
        # Look for common patterns in prompts
        patterns = [
            r"Query:\s*(.+?)(?:\n|$)",
            r"Question:\s*(.+?)(?:\n|$)",
            r"User query:\s*(.+?)(?:\n|$)",
            r"The user asks?:?\s*(.+?)(?:\n|$)",
            r"Please analyze this query:\s*(.+?)(?:\n|$)",
            r'"([^"]+)"',  # Quoted text
            r"'([^']+)'",  # Single quoted text
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
            if match:
                query = match.group(1).strip()
                # Clean up the query
                query = re.sub(r'\s+', ' ', query)  # Normalize whitespace
                if query and len(query) > 3:  # Reasonable length
                    return query
        
        # If no pattern matches, use a hash of the prompt for consistency
        # This ensures deterministic responses even for unexpected prompt formats
        prompt_hash = hashlib.md5(prompt.lower().encode()).hexdigest()
        return f"generic_query_{prompt_hash[:8]}"
    
    def _get_simple_response(self, prompt: str) -> str:
        """Get simple response when deterministic system is disabled.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Simple mock response
        """
        prompt_lower = prompt.lower()
        
        # Determine response type based on prompt content
        if any(word in prompt_lower for word in ["math", "calculate", "solve", "equation"]):
            return self._fallback_responses["mathematical"]
        elif any(word in prompt_lower for word in ["compare", "versus", "vs", "difference"]):
            return self._fallback_responses["comparison"]
        elif any(word in prompt_lower for word in ["how to", "steps", "process", "method"]):
            return self._fallback_responses["procedural"]
        elif any(word in prompt_lower for word in ["analyze", "examine", "evaluate"]):
            return self._fallback_responses["analysis"]
        else:
            return self._fallback_responses["general"]
    
    def _get_fallback_responses(self) -> Dict[str, str]:
        """Get fallback responses for when deterministic system is unavailable.
        
        Returns:
            Dictionary of fallback responses by type
        """
        return {
            "mathematical": """Here's a plan to solve this mathematical problem:

1. Parse and understand the mathematical expression or problem
2. Identify the required mathematical operations and concepts
3. Break down complex calculations into manageable steps
4. Perform the calculations with appropriate precision
5. Verify the result and check for reasonableness
6. Present the solution clearly with step-by-step working""",
            
            "comparison": """Here's a plan to compare the requested items:

1. Identify all items or concepts that need to be compared
2. Determine the relevant criteria and dimensions for comparison
3. Research key characteristics and features of each option
4. Analyze similarities and differences systematically
5. Evaluate pros and cons based on specific use cases
6. Present a balanced comparison with clear conclusions""",
            
            "procedural": """Here's a step-by-step plan to address your request:

1. Break down the process into logical, sequential steps
2. Research best practices and standard procedures
3. Identify required tools, materials, or prerequisites
4. Organize steps in chronological order with clear transitions
5. Include safety considerations and potential pitfalls
6. Provide clear, actionable instructions with examples""",
            
            "analysis": """Here's a plan to analyze the topic thoroughly:

1. Define the scope and objectives of the analysis
2. Gather relevant data and information from reliable sources
3. Apply appropriate analytical frameworks and methods
4. Identify patterns, trends, and key insights
5. Draw evidence-based conclusions and recommendations
6. Present findings in a structured, actionable format""",
            
                        "general": """Here is a structured plan to address your request:

1. Understand and analyze the user's question or request
2. Identify the key information needed to provide a complete answer
3. Research and gather relevant data from appropriate sources
4. Process and organize the information in a logical structure
5. Provide a comprehensive and helpful response""",
        }
    
    async def health_check(self) -> bool:
        """Check if the mock LLM is available.
        
        Returns:
            Always True for mock implementation
        """
        return True
    
    def get_deterministic_test_data(self) -> Dict[str, any]:
        """Get test data for verifying deterministic behavior.
        
        Returns:
            Dictionary with test queries and expected responses
        """
        if self.enable_deterministic_responses and self.mock_response_system:
            return {
                "test_scenarios": self.mock_response_system.get_test_scenarios(),
                "statistics": self.mock_response_system.get_statistics(),
                "deterministic_verification": {
                    "test_query": "What is machine learning?",
                    "is_deterministic": self.mock_response_system.is_deterministic("What is machine learning?")
                }
            }
        else:
            return {
                "fallback_mode": True,
                "available_types": list(self._fallback_responses.keys())
            }