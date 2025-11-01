"""Planner node for generating multi-step plans using LLM.

This module implements the PlannerNode that serves as the strategic thinking
component of the agentic system, analyzing user queries and generating
structured, executable plans.
"""

import logging
from typing import Optional

from ..config import get_settings
from ..state import ConversationState
from ..llm import BaseLLM, LLMFactory
from .prompt_templates import QueryClassifier
from .plan_validator import PlanValidator, PlanQuality


logger = logging.getLogger(__name__)


class PlannerNode:
    """Planner node that generates multi-step plans for user queries.
    
    This node serves as the strategic thinking component of the agentic system,
    analyzing user queries and generating structured, executable plans that
    guide the subsequent processing steps.
    
    Attributes:
        llm: The LLM instance for plan generation
        settings: Application settings
    """
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize the Planner node.
        
        Args:
            llm: Optional LLM instance. If None, will create based on settings.
        """
        self.settings = get_settings()
        self.llm = llm or self._create_llm()
        self.validator = PlanValidator()
        
        logger.info(f"PlannerNode initialized with {type(self.llm).__name__}")
    
    def _create_llm(self) -> BaseLLM:
        """Create LLM instance based on configuration.
        
        Returns:
            Configured LLM instance
        """
        return LLMFactory.create_llm()
    
    async def execute(self, state: ConversationState) -> ConversationState:
        """Execute the planning process for the given conversation state.
        
        Args:
            state: Current conversation state containing the user query
            
        Returns:
            Updated conversation state with generated plan
            
        Raises:
            ValueError: If the query is empty or invalid
            RuntimeError: If plan generation fails critically
        """
        try:
            logger.info(f"Starting plan generation for conversation {state.conversation_id}")
            
            # Validate input
            if not state.query or not state.query.strip():
                raise ValueError("Query cannot be empty for plan generation")
            
            # Generate plan using LLM
            plan_text = await self._generate_plan(state.query)
            
            # Parse and validate the plan
            plan_steps = self._parse_plan(plan_text)
            validation_result = self.validator.validate_plan(plan_steps, state.query)
            
            # Use validated and structured steps
            final_steps = validation_result.structured_steps
            
            # Handle validation results
            if not validation_result.is_valid:
                logger.warning(f"Plan validation failed: {validation_result.issues}")
                state.add_error(f"Plan validation failed: {'; '.join(validation_result.issues)}")
                # Use fallback plan if validation fails completely
                final_steps = self._create_fallback_plan(state.query)
            elif validation_result.quality == PlanQuality.POOR:
                logger.warning(f"Plan quality is poor (score: {validation_result.score})")
                state.update_metadata("plan_quality_warning", True)
            
            # Update state with generated plan
            for step in final_steps:
                state.add_plan_step(step)
            
            # Add comprehensive metadata
            state.update_metadata("planner_executed", True)
            state.update_metadata("plan_steps_count", len(final_steps))
            state.update_metadata("plan_quality_score", validation_result.score)
            state.update_metadata("plan_quality_level", validation_result.quality.value)
            
            if validation_result.issues:
                state.update_metadata("plan_validation_issues", validation_result.issues)
            if validation_result.suggestions:
                state.update_metadata("plan_improvement_suggestions", validation_result.suggestions)
            
            logger.info(f"Successfully generated {len(final_steps)} plan steps with quality score {validation_result.score}")
            return state
            
        except ValueError as e:
            error_msg = f"Validation error in planner: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
            
        except Exception as e:
            error_msg = f"Unexpected error in planner: {str(e)}"
            logger.error(error_msg, exc_info=True)
            state.add_error(error_msg)
            return state
    
    async def _generate_plan(self, query: str) -> str:
        """Generate plan using the LLM.
        
        Args:
            query: User query to generate plan for
            
        Returns:
            Generated plan text from LLM
            
        Raises:
            RuntimeError: If LLM generation fails
        """
        try:
            # Create planning prompt
            prompt = self._create_planning_prompt(query)
            
            # Generate plan using LLM
            plan_text = await self.llm.generate(prompt)
            
            if not plan_text or not plan_text.strip():
                raise RuntimeError("LLM returned empty plan")
            
            return plan_text.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate plan: {str(e)}")
            raise RuntimeError(f"Plan generation failed: {str(e)}")
    
    def _create_planning_prompt(self, query: str) -> str:
        """Create planning prompt for the LLM using intelligent templates.
        
        Args:
            query: User query to create prompt for
            
        Returns:
            Formatted planning prompt optimized for the query type
        """
        # Use intelligent template selection
        template = QueryClassifier.get_template_for_query(query)
        return template.format(query=query)
    
    def _parse_plan(self, plan_text: str) -> list[str]:
        """Parse plan text into structured steps.
        
        Args:
            plan_text: Raw plan text from LLM
            
        Returns:
            List of parsed plan steps
        """
        steps = []
        
        # Split by lines and process each
        lines = plan_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering (1., 2., etc.) and clean up
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove leading numbering/bullets
                cleaned = line
                for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.',
                              '-', '*', '•']:
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):].strip()
                        break
                
                if cleaned:
                    steps.append(cleaned)
            elif line and not any(skip in line.lower() for skip in ['plan:', 'steps:', 'response:']):
                # Include lines that don't start with numbers but seem like steps
                steps.append(line)
        
        # Ensure we have at least one step
        if not steps:
            steps = ["Analyze the query and provide a comprehensive response"]
        
        logger.debug(f"Parsed {len(steps)} steps from plan")
        return steps
    
    def _create_fallback_plan(self, query: str) -> List[str]:
        """Create a fallback plan when validation fails.
        
        Args:
            query: User query to create fallback plan for
            
        Returns:
            Basic but valid fallback plan steps
        """
        logger.info("Creating fallback plan due to validation failure")
        
        fallback_steps = [
            "Analyze the user's question to understand the core request.",
            "Identify the key information needed to provide a complete answer.",
            "Gather relevant data from appropriate and reliable sources.",
            "Synthesize the information into a comprehensive response."
        ]
        
        # Customize based on query type if possible
        query_lower = query.lower()
        if "math" in query_lower or "calculate" in query_lower:
            fallback_steps[2] = "Perform the necessary calculations step by step."
        elif "compare" in query_lower:
            fallback_steps[2] = "Gather information about each option being compared."
        elif "how to" in query_lower:
            fallback_steps[2] = "Research the step-by-step process or instructions."
        
        return fallback_steps