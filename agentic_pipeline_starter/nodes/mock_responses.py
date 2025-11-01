"""Deterministic mock responses for testing the Planner node.

This module provides deterministic, realistic mock responses for testing
the planner functionality. It includes various scenario types and ensures
consistent, reproducible test results.
"""

import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import re


class MockScenario(Enum):
    """Different types of mock scenarios for testing."""
    FACTUAL_QUERY = "factual_query"
    MATHEMATICAL_PROBLEM = "mathematical_problem"
    COMPARISON_REQUEST = "comparison_request"
    PROCEDURAL_QUESTION = "procedural_question"
    CREATIVE_TASK = "creative_task"
    ANALYSIS_REQUEST = "analysis_request"
    TROUBLESHOOTING = "troubleshooting"
    RESEARCH_TASK = "research_task"
    PLANNING_REQUEST = "planning_request"
    CODING_TASK = "coding_task"


@dataclass
class MockResponse:
    """Mock response data structure.
    
    Attributes:
        scenario: Type of scenario this response is for
        query_pattern: Regex pattern that matches this response
        plan_steps: List of plan steps to return
        expected_quality: Expected quality level for validation
        metadata: Additional metadata about the response
    """
    scenario: MockScenario
    query_pattern: str
    plan_steps: List[str]
    expected_quality: str
    metadata: Dict[str, Any]


class DeterministicMockResponses:
    """Provides deterministic mock responses for testing.
    
    This class generates consistent, realistic mock responses based on
    query content and patterns. Responses are deterministic based on
    query hash to ensure reproducible testing.
    """
    
    def __init__(self):
        """Initialize the mock response generator."""
        self.mock_responses = self._create_mock_responses()
        self.fallback_responses = self._create_fallback_responses()
        
    def _create_mock_responses(self) -> List[MockResponse]:
        """Create a comprehensive set of mock responses.
        
        Returns:
            List of mock responses covering various scenarios
        """
        return [
            # Factual queries
            MockResponse(
                scenario=MockScenario.FACTUAL_QUERY,
                query_pattern=r"what is|who is|where is|when did|define",
                plan_steps=[
                    "Identify the specific information being requested",
                    "Search for authoritative sources on the topic",
                    "Verify information accuracy from multiple sources", 
                    "Compile findings into a comprehensive answer",
                    "Present the information in a clear, structured format"
                ],
                expected_quality="GOOD",
                metadata={"complexity": "medium", "steps": 5}
            ),
            
            # Mathematical problems
            MockResponse(
                scenario=MockScenario.MATHEMATICAL_PROBLEM,
                query_pattern=r"calculate|solve|math|equation|formula|compute",
                plan_steps=[
                    "Parse the mathematical expression or problem statement",
                    "Identify the mathematical operations and concepts required",
                    "Break down complex problems into manageable steps",
                    "Perform calculations with appropriate precision",
                    "Verify the result and check for reasonableness",
                    "Present the solution with clear step-by-step working"
                ],
                expected_quality="EXCELLENT",
                metadata={"complexity": "high", "steps": 6}
            ),
            
            # Comparison requests
            MockResponse(
                scenario=MockScenario.COMPARISON_REQUEST,
                query_pattern=r"compare|versus|vs|difference|better|contrast",
                plan_steps=[
                    "Identify all items or concepts to be compared",
                    "Determine relevant comparison criteria and dimensions",
                    "Research key characteristics and features of each option",
                    "Analyze similarities and differences systematically",
                    "Evaluate pros and cons based on specific use cases",
                    "Present a balanced comparison with clear conclusions"
                ],
                expected_quality="GOOD",
                metadata={"complexity": "high", "steps": 6}
            ),
            
            # Procedural questions
            MockResponse(
                scenario=MockScenario.PROCEDURAL_QUESTION,
                query_pattern=r"how to|steps|process|method|procedure|guide",
                plan_steps=[
                    "Break down the process into logical, sequential steps",
                    "Research best practices and standard procedures",
                    "Identify required tools, materials, or prerequisites",
                    "Organize steps in chronological order with clear transitions",
                    "Include safety considerations and potential pitfalls",
                    "Provide clear, actionable instructions with examples"
                ],
                expected_quality="EXCELLENT",
                metadata={"complexity": "medium", "steps": 6}
            ),
            
            # Creative tasks
            MockResponse(
                scenario=MockScenario.CREATIVE_TASK,
                query_pattern=r"create|design|write|compose|generate|brainstorm",
                plan_steps=[
                    "Understand the creative requirements and constraints",
                    "Brainstorm initial concepts and ideas",
                    "Research relevant styles, techniques, or examples",
                    "Develop and refine the most promising concepts",
                    "Create the final output with attention to quality",
                    "Review and iterate based on the original requirements"
                ],
                expected_quality="GOOD",
                metadata={"complexity": "high", "steps": 6}
            ),
            
            # Analysis requests
            MockResponse(
                scenario=MockScenario.ANALYSIS_REQUEST,
                query_pattern=r"analyze|examine|evaluate|assess|review|study",
                plan_steps=[
                    "Define the scope and objectives of the analysis",
                    "Gather relevant data and information sources",
                    "Apply appropriate analytical frameworks and methods",
                    "Identify patterns, trends, and key insights",
                    "Draw evidence-based conclusions and recommendations",
                    "Present findings in a structured, actionable format"
                ],
                expected_quality="EXCELLENT",
                metadata={"complexity": "high", "steps": 6}
            ),
            
            # Troubleshooting
            MockResponse(
                scenario=MockScenario.TROUBLESHOOTING,
                query_pattern=r"fix|debug|troubleshoot|error|problem|issue|not working",
                plan_steps=[
                    "Reproduce and clearly define the problem or error",
                    "Gather detailed information about the system and context",
                    "Identify potential causes through systematic investigation",
                    "Test solutions starting with the most likely causes",
                    "Implement the effective solution with proper verification",
                    "Document the resolution for future reference"
                ],
                expected_quality="EXCELLENT",
                metadata={"complexity": "high", "steps": 6}
            ),
            
            # Research tasks
            MockResponse(
                scenario=MockScenario.RESEARCH_TASK,
                query_pattern=r"research|investigate|explore|find out|learn about",
                plan_steps=[
                    "Define research objectives and key questions",
                    "Identify authoritative and relevant information sources",
                    "Conduct systematic information gathering and review",
                    "Evaluate source credibility and information quality",
                    "Synthesize findings and identify knowledge gaps",
                    "Present comprehensive research summary with citations"
                ],
                expected_quality="GOOD",
                metadata={"complexity": "high", "steps": 6}
            ),
            
            # Planning requests
            MockResponse(
                scenario=MockScenario.PLANNING_REQUEST,
                query_pattern=r"plan|strategy|roadmap|schedule|organize|prepare",
                plan_steps=[
                    "Define clear goals and success criteria",
                    "Assess current resources and constraints",
                    "Break down objectives into manageable tasks",
                    "Sequence tasks logically with dependencies",
                    "Allocate timeframes and assign responsibilities",
                    "Create monitoring checkpoints and contingency plans"
                ],
                expected_quality="EXCELLENT",
                metadata={"complexity": "high", "steps": 6}
            ),
            
            # Coding tasks
            MockResponse(
                scenario=MockScenario.CODING_TASK,
                query_pattern=r"code|program|develop|implement|build|script",
                plan_steps=[
                    "Understand the functional requirements and specifications",
                    "Design the overall architecture and data structures",
                    "Break down implementation into modular components",
                    "Write clean, well-documented code following best practices",
                    "Test the implementation thoroughly with various scenarios",
                    "Refactor and optimize the code for maintainability"
                ],
                expected_quality="EXCELLENT",
                metadata={"complexity": "high", "steps": 6}
            )
        ]
    
    def _create_fallback_responses(self) -> Dict[str, List[str]]:
        """Create fallback responses for different complexity levels.
        
        Returns:
            Dictionary of fallback responses by complexity
        """
        return {
            "simple": [
                "Understand the user's request clearly",
                "Gather the necessary information to respond",
                "Provide a helpful and accurate answer"
            ],
            "medium": [
                "Analyze the user's question to identify key requirements",
                "Research and gather relevant information from reliable sources",
                "Process and organize the information logically",
                "Provide a comprehensive and well-structured response"
            ],
            "complex": [
                "Break down the complex request into manageable components",
                "Conduct thorough research on each component",
                "Analyze relationships and dependencies between components",
                "Synthesize findings into coherent insights",
                "Present a comprehensive solution with clear reasoning",
                "Validate the approach and suggest improvements"
            ]
        }
    
    def get_mock_response_for_query(self, query: str) -> List[str]:
        """Get deterministic mock response for a query.
        
        Args:
            query: User query to generate response for
            
        Returns:
            List of plan steps for the query
        """
        # Create deterministic hash of the query
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        
        # Find matching response pattern
        for response in self.mock_responses:
            if re.search(response.query_pattern, query.lower()):
                # Use hash to slightly vary the response
                steps = response.plan_steps.copy()
                
                # Deterministically modify based on hash
                hash_int = int(query_hash[:8], 16)
                
                # Occasionally add an extra step
                if hash_int % 7 == 0 and len(steps) < 8:
                    extra_steps = [
                        "Validate assumptions and prerequisites",
                        "Consider alternative approaches or perspectives",
                        "Review and refine the approach based on best practices",
                        "Ensure all edge cases are properly addressed"
                    ]
                    extra_step = extra_steps[hash_int % len(extra_steps)]
                    # Insert before the last step
                    steps.insert(-1, extra_step)
                
                # Occasionally simplify by removing a step
                elif hash_int % 13 == 0 and len(steps) > 3:
                    # Remove a middle step deterministically
                    remove_index = 1 + (hash_int % (len(steps) - 2))
                    steps.pop(remove_index)
                
                return steps
        
        # No pattern match - use hash-based fallback
        hash_int = int(query_hash[:8], 16)
        complexity_level = ["simple", "medium", "complex"][hash_int % 3]
        
        fallback_steps = self.fallback_responses[complexity_level].copy()
        
        # Add query-specific customization
        if "math" in query.lower() or "calculate" in query.lower():
            fallback_steps[1] = "Identify the mathematical concepts and operations needed"
        elif "compare" in query.lower():
            fallback_steps[1] = "Gather information about each option being compared"
        elif "how to" in query.lower():
            fallback_steps[1] = "Research the step-by-step process or methodology"
        
        return fallback_steps
    
    def get_expected_quality_for_query(self, query: str) -> str:
        """Get expected quality level for a query.
        
        Args:
            query: User query
            
        Returns:
            Expected quality level (EXCELLENT, GOOD, ACCEPTABLE, etc.)
        """
        # Find matching response pattern
        for response in self.mock_responses:
            if re.search(response.query_pattern, query.lower()):
                return response.expected_quality
        
        # Default quality based on query complexity
        query_lower = query.lower()
        if any(word in query_lower for word in ["analyze", "compare", "calculate", "solve"]):
            return "GOOD"
        elif len(query.split()) > 10:
            return "GOOD"
        else:
            return "ACCEPTABLE"
    
    def get_mock_metadata_for_query(self, query: str) -> Dict[str, Any]:
        """Get mock metadata for a query.
        
        Args:
            query: User query
            
        Returns:
            Metadata dictionary with mock information
        """
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        hash_int = int(query_hash[:8], 16)
        
        # Find matching response for base metadata
        base_metadata = {"complexity": "medium", "steps": 4}
        for response in self.mock_responses:
            if re.search(response.query_pattern, query.lower()):
                base_metadata = response.metadata.copy()
                break
        
        # Add deterministic mock data
        base_metadata.update({
            "mock_generation_time": 50 + (hash_int % 200),  # 50-250ms
            "mock_llm_tokens": 100 + (hash_int % 300),      # 100-400 tokens
            "mock_confidence": 0.7 + ((hash_int % 30) / 100),  # 0.7-0.99
            "mock_query_hash": query_hash[:8],
            "mock_scenario": self._detect_scenario(query).value
        })
        
        return base_metadata
    
    def _detect_scenario(self, query: str) -> MockScenario:
        """Detect the most likely scenario for a query.
        
        Args:
            query: User query
            
        Returns:
            Most likely MockScenario
        """
        query_lower = query.lower()
        
        # Pattern matching for scenarios
        scenario_patterns = {
            MockScenario.MATHEMATICAL_PROBLEM: r"calculate|solve|math|equation|formula|compute",
            MockScenario.COMPARISON_REQUEST: r"compare|versus|vs|difference|better|contrast",
            MockScenario.PROCEDURAL_QUESTION: r"how to|steps|process|method|procedure|guide",
            MockScenario.CREATIVE_TASK: r"create|design|write|compose|generate|brainstorm",
            MockScenario.ANALYSIS_REQUEST: r"analyze|examine|evaluate|assess|review|study",
            MockScenario.TROUBLESHOOTING: r"fix|debug|troubleshoot|error|problem|issue|not working",
            MockScenario.RESEARCH_TASK: r"research|investigate|explore|find out|learn about",
            MockScenario.PLANNING_REQUEST: r"plan|strategy|roadmap|schedule|organize|prepare",
            MockScenario.CODING_TASK: r"code|program|develop|implement|build|script",
            MockScenario.FACTUAL_QUERY: r"what is|who is|where is|when did|define"
        }
        
        for scenario, pattern in scenario_patterns.items():
            if re.search(pattern, query_lower):
                return scenario
        
        # Default to factual query
        return MockScenario.FACTUAL_QUERY
    
    def generate_mock_llm_response(self, query: str, plan_steps: Optional[List[str]] = None) -> str:
        """Generate a mock LLM response that would produce the given plan steps.
        
        Args:
            query: Original user query
            plan_steps: Plan steps to format as LLM response
            
        Returns:
            Formatted mock LLM response
        """
        if plan_steps is None:
            plan_steps = self.get_mock_response_for_query(query)
        
        # Create realistic LLM response format
        response_parts = ["Here is a structured plan to address your request:\n"]
        
        for i, step in enumerate(plan_steps, 1):
            response_parts.append(f"{i}. {step}")
        
        response_parts.append("\nThis plan provides a systematic approach to achieve the desired outcome.")
        
        return "\n".join(response_parts)
    
    def get_response_variations(self, query: str, count: int = 3) -> List[List[str]]:
        """Get multiple deterministic variations of responses for testing.
        
        Args:
            query: User query
            count: Number of variations to generate
            
        Returns:
            List of response variations
        """
        base_response = self.get_mock_response_for_query(query)
        variations = [base_response]
        
        # Generate variations by modifying the query slightly
        for i in range(1, count):
            modified_query = f"{query} variation_{i}"
            variation = self.get_mock_response_for_query(modified_query)
            variations.append(variation)
        
        return variations
    
    def get_test_scenarios(self) -> List[Tuple[str, List[str], str]]:
        """Get predefined test scenarios for comprehensive testing.
        
        Returns:
            List of tuples (query, expected_steps, expected_quality)
        """
        return [
            (
                "What is artificial intelligence?",
                self.get_mock_response_for_query("What is artificial intelligence?"),
                "GOOD"
            ),
            (
                "How to bake a chocolate cake?",
                self.get_mock_response_for_query("How to bake a chocolate cake?"),
                "EXCELLENT"
            ),
            (
                "Calculate the area of a circle with radius 5",
                self.get_mock_response_for_query("Calculate the area of a circle with radius 5"),
                "EXCELLENT"
            ),
            (
                "Compare Python and Java programming languages",
                self.get_mock_response_for_query("Compare Python and Java programming languages"),
                "GOOD"
            ),
            (
                "Analyze the impact of climate change on polar bears",
                self.get_mock_response_for_query("Analyze the impact of climate change on polar bears"),
                "EXCELLENT"
            ),
            (
                "Debug a segmentation fault in C++",
                self.get_mock_response_for_query("Debug a segmentation fault in C++"),
                "EXCELLENT"
            ),
            (
                "Create a marketing strategy for a new product",
                self.get_mock_response_for_query("Create a marketing strategy for a new product"),
                "GOOD"
            ),
            (
                "Simple question",
                self.get_mock_response_for_query("Simple question"),
                "ACCEPTABLE"
            )
        ]
    
    def is_deterministic(self, query: str, iterations: int = 5) -> bool:
        """Verify that responses are deterministic for a query.
        
        Args:
            query: Query to test
            iterations: Number of iterations to test
            
        Returns:
            True if all responses are identical
        """
        first_response = self.get_mock_response_for_query(query)
        
        for _ in range(iterations - 1):
            response = self.get_mock_response_for_query(query)
            if response != first_response:
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the mock response system.
        
        Returns:
            Statistics about available responses and scenarios
        """
        scenario_counts = {}
        total_responses = len(self.mock_responses)
        
        for response in self.mock_responses:
            scenario = response.scenario.value
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        avg_steps = sum(len(r.plan_steps) for r in self.mock_responses) / total_responses
        
        quality_distribution = {}
        for response in self.mock_responses:
            quality = response.expected_quality
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        return {
            "total_responses": total_responses,
            "scenario_distribution": scenario_counts,
            "average_steps_per_response": avg_steps,
            "quality_distribution": quality_distribution,
            "fallback_levels": list(self.fallback_responses.keys()),
            "supported_scenarios": [s.value for s in MockScenario]
        }


# Global instance for easy access
mock_responses = DeterministicMockResponses()