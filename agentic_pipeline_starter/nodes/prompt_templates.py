"""Prompt templates for the Planner node.

This module contains structured prompt templates used by the PlannerNode
to generate high-quality, context-aware plans for different types of queries.
"""

from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass


class QueryType(Enum):
    """Types of queries that can be handled by the planner."""
    GENERAL = "general"
    FACTUAL = "factual"
    MATHEMATICAL = "mathematical"
    RESEARCH = "research"
    COMPARISON = "comparison"
    PROCEDURAL = "procedural"
    CREATIVE = "creative"


@dataclass
class PromptTemplate:
    """Structured prompt template for plan generation.
    
    Attributes:
        system_prompt: System-level instructions for the LLM
        user_template: Template for the user query portion
        examples: Optional examples to guide the LLM
        constraints: Specific constraints for this template type
    """
    system_prompt: str
    user_template: str
    examples: Optional[str] = None
    constraints: Optional[str] = None
    
    def format(self, query: str, **kwargs) -> str:
        """Format the complete prompt with the given query.
        
        Args:
            query: User query to incorporate
            **kwargs: Additional template variables
            
        Returns:
            Formatted prompt ready for LLM
        """
        prompt_parts = [self.system_prompt]
        
        if self.examples:
            prompt_parts.append(f"\n{self.examples}")
        
        if self.constraints:
            prompt_parts.append(f"\n{self.constraints}")
        
        user_prompt = self.user_template.format(query=query, **kwargs)
        prompt_parts.append(f"\n{user_prompt}")
        
        return "\n".join(prompt_parts)


class PlanningPrompts:
    """Collection of prompt templates for different query types."""
    
    # Base system prompt used across all templates
    BASE_SYSTEM_PROMPT = """You are an expert AI planning assistant. Your task is to create clear, actionable plans to answer user queries effectively.

Your plans should be:
- Specific and actionable steps
- Logically ordered and sequential
- Comprehensive but concise
- Focused on gathering accurate information
- Designed to provide thorough, helpful answers

Format your response as a numbered list of steps. Each step should be a clear action that can be executed to gather information or solve the problem."""
    
    @classmethod
    def get_templates(cls) -> Dict[QueryType, PromptTemplate]:
        """Get all available prompt templates.
        
        Returns:
            Dictionary mapping query types to their templates
        """
        return {
            QueryType.GENERAL: cls._create_general_template(),
            QueryType.FACTUAL: cls._create_factual_template(),
            QueryType.MATHEMATICAL: cls._create_mathematical_template(),
            QueryType.RESEARCH: cls._create_research_template(),
            QueryType.COMPARISON: cls._create_comparison_template(),
            QueryType.PROCEDURAL: cls._create_procedural_template(),
            QueryType.CREATIVE: cls._create_creative_template(),
        }
    
    @classmethod
    def _create_general_template(cls) -> PromptTemplate:
        """Create template for general queries."""
        return PromptTemplate(
            system_prompt=cls.BASE_SYSTEM_PROMPT,
            user_template="""User Query: "{query}"

Please create a step-by-step plan to answer this query comprehensively.

Plan:""",
            examples="""Example:
Query: "What is the weather like today?"
Plan:
1. Identify the user's current location or requested location
2. Access current weather data from reliable meteorological sources
3. Gather key weather metrics (temperature, conditions, humidity, wind)
4. Present the weather information in a clear, understandable format""",
            constraints="Keep the plan focused and avoid unnecessary steps. Aim for 3-6 actionable steps."
        )
    
    @classmethod
    def _create_factual_template(cls) -> PromptTemplate:
        """Create template for factual information queries."""
        return PromptTemplate(
            system_prompt=cls.BASE_SYSTEM_PROMPT,
            user_template="""User Query: "{query}"

This appears to be a factual information request. Create a plan to research and verify the information needed.

Plan:""",
            examples="""Example:
Query: "What is the capital of Germany?"
Plan:
1. Search for official information about Germany's capital city
2. Verify the information from reliable government or educational sources
3. Confirm the current status (as capitals can change historically)
4. Provide the definitive answer with confidence level""",
            constraints="Focus on authoritative sources and verification. Include fact-checking steps."
        )
    
    @classmethod
    def _create_mathematical_template(cls) -> PromptTemplate:
        """Create template for mathematical queries."""
        return PromptTemplate(
            system_prompt=cls.BASE_SYSTEM_PROMPT,
            user_template="""User Query: "{query}"

This appears to be a mathematical problem or calculation. Create a plan to solve it systematically.

Plan:""",
            examples="""Example:
Query: "What is 15% of 250?"
Plan:
1. Parse the mathematical expression to identify the operation needed
2. Convert the percentage to a decimal (15% = 0.15)
3. Multiply 250 by 0.15 to calculate the result
4. Verify the calculation and present the answer clearly""",
            constraints="Break down complex calculations into clear steps. Show the mathematical reasoning."
        )
    
    @classmethod
    def _create_research_template(cls) -> PromptTemplate:
        """Create template for research queries."""
        return PromptTemplate(
            system_prompt=cls.BASE_SYSTEM_PROMPT,
            user_template="""User Query: "{query}"

This requires research across multiple sources. Create a comprehensive research plan.

Plan:""",
            examples="""Example:
Query: "What are the benefits of renewable energy?"
Plan:
1. Define the scope of renewable energy sources to research
2. Gather information from academic and scientific sources
3. Research economic benefits and cost comparisons
4. Investigate environmental impact studies
5. Compile and synthesize findings into a comprehensive overview""",
            constraints="Include diverse, credible sources. Plan for synthesis and analysis of findings."
        )
    
    @classmethod
    def _create_comparison_template(cls) -> PromptTemplate:
        """Create template for comparison queries."""
        return PromptTemplate(
            system_prompt=cls.BASE_SYSTEM_PROMPT,
            user_template="""User Query: "{query}"

This requires comparing different options or alternatives. Create a structured comparison plan.

Plan:""",
            examples="""Example:
Query: "Python vs JavaScript for web development"
Plan:
1. Identify key comparison criteria (performance, ease of learning, ecosystem)
2. Research Python's capabilities and frameworks for web development
3. Research JavaScript's features and frameworks for web development
4. Compare each language across the identified criteria
5. Summarize strengths and weaknesses with practical recommendations""",
            constraints="Ensure balanced comparison with clear criteria. Avoid bias toward any option."
        )
    
    @classmethod
    def _create_procedural_template(cls) -> PromptTemplate:
        """Create template for how-to or procedural queries."""
        return PromptTemplate(
            system_prompt=cls.BASE_SYSTEM_PROMPT,
            user_template="""User Query: "{query}"

This appears to be a procedural question requiring step-by-step instructions. Create a plan to provide comprehensive guidance.

Plan:""",
            examples="""Example:
Query: "How to bake chocolate chip cookies?"
Plan:
1. Research reliable recipe sources and ingredients list
2. Identify the required equipment and tools
3. Outline the step-by-step baking process
4. Include important tips for success and common mistakes to avoid
5. Provide timing and temperature guidelines""",
            constraints="Focus on clear, sequential instructions. Include safety considerations and tips."
        )
    
    @classmethod
    def _create_creative_template(cls) -> PromptTemplate:
        """Create template for creative or open-ended queries."""
        return PromptTemplate(
            system_prompt=cls.BASE_SYSTEM_PROMPT,
            user_template="""User Query: "{query}"

This is a creative or open-ended request. Create a plan to provide thoughtful, imaginative assistance.

Plan:""",
            examples="""Example:
Query: "Write a short story about space exploration"
Plan:
1. Brainstorm creative themes and concepts related to space exploration
2. Develop interesting characters (astronauts, scientists, aliens)
3. Create a compelling plot structure with conflict and resolution
4. Research accurate space-related details for authenticity
5. Write and refine the story with engaging narrative flow""",
            constraints="Encourage creativity while maintaining structure. Balance imagination with helpful guidance."
        )


class QueryClassifier:
    """Classifier to determine the most appropriate prompt template for a query."""
    
    # Keywords that indicate different query types
    CLASSIFICATION_KEYWORDS = {
        QueryType.FACTUAL: [
            "what is", "who is", "when did", "where is", "which", "capital", 
            "definition", "meaning", "fact", "information about"
        ],
        QueryType.MATHEMATICAL: [
            "calculate", "compute", "solve", "equation", "formula", "math",
            "percentage", "ratio", "sum", "product", "divide", "multiply"
        ],
        QueryType.RESEARCH: [
            "research", "study", "analyze", "investigate", "explore", "examine",
            "benefits of", "effects of", "impact of", "history of"
        ],
        QueryType.COMPARISON: [
            "vs", "versus", "compare", "difference", "better", "best", "choose",
            "pros and cons", "advantages", "disadvantages"
        ],
        QueryType.PROCEDURAL: [
            "how to", "how do", "steps to", "guide", "tutorial", "instructions",
            "process", "method", "way to"
        ],
        QueryType.CREATIVE: [
            "write", "create", "design", "imagine", "story", "poem", "idea",
            "brainstorm", "creative", "artistic"
        ]
    }
    
    @classmethod
    def classify_query(cls, query: str) -> QueryType:
        """Classify a query to determine the best prompt template.
        
        Args:
            query: User query to classify
            
        Returns:
            The most appropriate QueryType for the query
        """
        query_lower = query.lower()
        
        # Count matches for each category
        scores = {}
        for query_type, keywords in cls.CLASSIFICATION_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[query_type] = score
        
        # Return the type with the highest score, default to GENERAL
        if scores:
            return max(scores, key=scores.get)
        
        return QueryType.GENERAL
    
    @classmethod
    def get_template_for_query(cls, query: str) -> PromptTemplate:
        """Get the most appropriate template for a given query.
        
        Args:
            query: User query to get template for
            
        Returns:
            The most appropriate PromptTemplate for the query
        """
        query_type = cls.classify_query(query)
        templates = PlanningPrompts.get_templates()
        return templates[query_type]