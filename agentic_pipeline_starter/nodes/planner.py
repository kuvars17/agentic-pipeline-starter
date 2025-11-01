"""Planner node for the agentic reasoning pipeline.

This module implements the planning component that generates structured,
actionable plans based on user queries. It integrates with the LLM layer
to create intelligent, context-aware planning.
"""

import asyncio
from typing import List, Optional, Dict, Any
import logging

from ..state.conversation_state import ConversationState
from ..llm.base import BaseLLM
from ..llm.factory import LLMFactory
from .prompt_templates import PromptTemplateManager, QueryType
from .plan_validator import PlanValidator, ValidationResult, PlanQuality
from .error_handler import PlannerErrorHandler, CircuitBreaker, ErrorType


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
        self.llm = llm or self._create_llm()
        self.prompt_manager = PromptTemplateManager()
        self.validator = PlanValidator()
        
        # Initialize error handling and circuit breaker
        self.error_handler = PlannerErrorHandler()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        logger.info(f"PlannerNode initialized with {type(self.llm).__name__} and error handling")
    
    def _create_llm(self) -> BaseLLM:
        """Create LLM instance based on configuration.
        
        Returns:
            Configured LLM instance
        """
        return LLMFactory.create_llm()
    
    async def execute(self, state: ConversationState) -> ConversationState:
        """Execute the planning phase with comprehensive error handling.
        
        This method generates a structured plan for the user's query,
        validates the plan, and updates the conversation state.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state with plan
        """
        start_time = asyncio.get_event_loop().time()
        query = state.messages[-1].content if state.messages else ""
        
        logger.info(f"Starting plan generation for query: {query[:100]}...")
        
        try:
            # Use error handler for plan generation with fallback
            success, result = await self.error_handler.handle_with_fallback(
                self._generate_plan_with_circuit_breaker,
                query,
                query
            )
            
            if success:
                plan_steps = result
                logger.info(f"Plan generated successfully with {len(plan_steps)} steps")
            else:
                # Result is the fallback plan
                plan_steps = result
                logger.warning("Using fallback plan due to generation failures")
                state.add_error(f"Plan generation failed, using fallback: {type(result).__name__}")
            
            # Validate the plan (whether generated or fallback)
            validation_result = await self.validator.validate_plan(plan_steps, query)
            
            # Update state with plan and validation results
            state.plan = plan_steps
            state.add_metadata("plan_validation", {
                "quality": validation_result.quality.value,
                "score": validation_result.score,
                "issues": validation_result.issues,
                "suggestions": validation_result.suggestions,
                "is_fallback": not success,
                "generation_time": asyncio.get_event_loop().time() - start_time
            })
            
            # Log quality assessment
            if validation_result.quality in [PlanQuality.EXCELLENT, PlanQuality.GOOD]:
                logger.info(f"Plan quality: {validation_result.quality.value} (score: {validation_result.score:.2f})")
            else:
                logger.warning(f"Plan quality: {validation_result.quality.value} (score: {validation_result.score:.2f})")
                if validation_result.issues:
                    logger.warning(f"Plan issues: {', '.join(validation_result.issues)}")
            
            # Add error history summary to metadata for debugging
            error_summary = self.error_handler.get_error_summary()
            if error_summary["total_errors"] > 0:
                state.add_metadata("error_summary", error_summary)
            
            return state
            
        except Exception as e:
            logger.error(f"Critical error in plan execution: {str(e)}", exc_info=True)
            
            # Add error to state
            state.add_error(f"Critical planning error: {str(e)}")
            
            # Create emergency fallback plan
            emergency_plan = [
                "Acknowledge the user's request",
                "Attempt to provide helpful information based on available resources"
            ]
            
            state.plan = emergency_plan
            state.add_metadata("plan_validation", {
                "quality": "EMERGENCY",
                "score": 0.1,
                "issues": ["Emergency fallback used due to critical error"],
                "suggestions": ["Manual intervention may be required"],
                "is_fallback": True,
                "generation_time": asyncio.get_event_loop().time() - start_time
            })
            
            return state
    
    async def _generate_plan_with_circuit_breaker(self, query: str) -> List[str]:
        """Generate plan using circuit breaker protection.
        
        Args:
            query: User query to generate plan for
            
        Returns:
            Generated plan steps
        """
        return await self.circuit_breaker.call(self._generate_plan_internal, query)
    
    async def _generate_plan_internal(self, query: str) -> List[str]:
        """Internal plan generation method.
        
        Args:
            query: User query to generate plan for
            
        Returns:
            Generated plan steps
            
        Raises:
            Various exceptions based on failure type
        """
        # Generate prompt for the query
        template = self.prompt_manager.get_template_for_query(query)
        
        if not template:
            raise ValueError(f"No suitable template found for query type")
        
        # Generate plan using LLM
        try:
            plan_response = await self.llm.generate(template.format_prompt(query))
            
            if not plan_response or not plan_response.strip():
                raise ValueError("LLM returned empty response")
            
            # Parse the response into steps
            plan_steps = self._parse_plan_response(plan_response)
            
            if not plan_steps:
                raise ValueError("No valid plan steps extracted from LLM response")
            
            return plan_steps
            
        except Exception as e:
            logger.error(f"LLM plan generation failed: {str(e)}")
            raise e
    
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
        # Generate prompt for the query
        template = self.prompt_manager.get_template_for_query(query)
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