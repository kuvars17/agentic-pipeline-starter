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
from .planner_logger import PlannerLogger, LogContext, ComponentType, performance_monitor


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
        # Initialize comprehensive logging
        self.logger = PlannerLogger("PlannerNode")
        
        # Log initialization start
        self.logger.push_context(LogContext(
            component=ComponentType.PLANNER,
            operation="initialization"
        ))
        
        try:
            self.llm = llm or self._create_llm()
            self.prompt_manager = PromptTemplateManager()
            self.validator = PlanValidator()
            
            # Initialize error handling and circuit breaker
            self.error_handler = PlannerErrorHandler()
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0
            )
            
            self.logger.info(f"PlannerNode initialized successfully")
            self.logger.audit("planner_initialization", {
                "llm_type": type(self.llm).__name__,
                "error_handler_enabled": True,
                "circuit_breaker_enabled": True,
                "validation_enabled": True
            })
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PlannerNode: {str(e)}", exc_info=True)
            raise
        finally:
            self.logger.pop_context()
    
    def _create_llm(self) -> BaseLLM:
        """Create LLM instance based on configuration.
        
        Returns:
            Configured LLM instance
        """
        return LLMFactory.create_llm()
    
    @performance_monitor("plan_execution")
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
        
        # Set up logging context for this execution
        execution_context = LogContext(
            component=ComponentType.PLANNER,
            operation="execute",
            conversation_id=str(state.conversation_id),
            query_id=f"query_{len(state.messages)}",
            metadata={"query_length": len(query)}
        )
        self.logger.push_context(execution_context)
        
        try:
            self.logger.info(f"Starting plan generation", 
                           query_preview=query[:100] + "..." if len(query) > 100 else query,
                           query_length=len(query))
            
            # Use error handler for plan generation with fallback
            success, result = await self.error_handler.handle_with_fallback(
                self._generate_plan_with_circuit_breaker,
                query,
                query
            )
            
            if success:
                plan_steps = result
                self.logger.info(f"Plan generated successfully", 
                               step_count=len(plan_steps),
                               generation_method="llm")
            else:
                # Result is the fallback plan
                plan_steps = result
                self.logger.warning("Using fallback plan due to generation failures",
                                  step_count=len(plan_steps),
                                  generation_method="fallback")
                state.add_error(f"Plan generation failed, using fallback")
            
            # Validate the plan (whether generated or fallback)
            validation_start = asyncio.get_event_loop().time()
            validation_result = await self.validator.validate_plan(plan_steps, query)
            validation_duration = (asyncio.get_event_loop().time() - validation_start) * 1000
            
            # Log validation results
            self.logger.log_validation_result(
                plan_steps=plan_steps,
                quality_score=validation_result.score,
                quality_level=validation_result.quality.value,
                issues=validation_result.issues,
                suggestions=validation_result.suggestions,
                duration_ms=validation_duration
            )
            
            # Update state with plan and validation results
            state.plan = plan_steps
            total_duration = (asyncio.get_event_loop().time() - start_time) * 1000
            
            validation_metadata = {
                "quality": validation_result.quality.value,
                "score": validation_result.score,
                "issues": validation_result.issues,
                "suggestions": validation_result.suggestions,
                "is_fallback": not success,
                "generation_time": total_duration,
                "validation_time": validation_duration
            }
            state.add_metadata("plan_validation", validation_metadata)
            
            # Log quality assessment
            if validation_result.quality in [PlanQuality.EXCELLENT, PlanQuality.GOOD]:
                self.logger.info(f"High quality plan generated",
                               quality=validation_result.quality.value,
                               score=validation_result.score)
            else:
                self.logger.warning(f"Low quality plan generated",
                                  quality=validation_result.quality.value,
                                  score=validation_result.score,
                                  issues=validation_result.issues)
            
            # Add error history summary to metadata for debugging
            error_summary = self.error_handler.get_error_summary()
            if error_summary["total_errors"] > 0:
                state.add_metadata("error_summary", error_summary)
                self.logger.debug("Error summary added to state",
                                error_count=error_summary["total_errors"])
            
            # Log successful completion
            self.logger.audit("plan_execution_completed", {
                "success": True,
                "plan_steps": len(plan_steps),
                "quality": validation_result.quality.value,
                "duration_ms": total_duration,
                "fallback_used": not success
            })
            
            return state
            
        except Exception as e:
            self.logger.critical(f"Critical error in plan execution: {str(e)}", exc_info=True)
            
            # Add error to state
            state.add_error(f"Critical planning error: {str(e)}")
            
            # Create emergency fallback plan
            emergency_plan = [
                "Acknowledge the user's request",
                "Attempt to provide helpful information based on available resources"
            ]
            
            state.plan = emergency_plan
            total_duration = (asyncio.get_event_loop().time() - start_time) * 1000
            
            state.add_metadata("plan_validation", {
                "quality": "EMERGENCY",
                "score": 0.1,
                "issues": ["Emergency fallback used due to critical error"],
                "suggestions": ["Manual intervention may be required"],
                "is_fallback": True,
                "generation_time": total_duration
            })
            
            # Log critical failure
            self.logger.audit("plan_execution_failed", {
                "success": False,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "duration_ms": total_duration,
                "emergency_fallback_used": True
            })
            
            return state
            
        finally:
            self.logger.pop_context()
    
    @performance_monitor("llm_plan_generation")
    async def _generate_plan_with_circuit_breaker(self, query: str) -> List[str]:
        """Generate plan using circuit breaker protection.
        
        Args:
            query: User query to generate plan for
            
        Returns:
            Generated plan steps
        """
        self.logger.push_context(LogContext(
            component=ComponentType.CIRCUIT_BREAKER,
            operation="protected_call"
        ))
        
        try:
            self.logger.debug("Executing plan generation with circuit breaker protection")
            result = await self.circuit_breaker.call(self._generate_plan_internal, query)
            self.logger.debug("Circuit breaker call successful")
            return result
        except Exception as e:
            self.logger.error(f"Circuit breaker call failed: {str(e)}")
            raise
        finally:
            self.logger.pop_context()
    
    @performance_monitor("internal_plan_generation")
    async def _generate_plan_internal(self, query: str) -> List[str]:
        """Internal plan generation method.
        
        Args:
            query: User query to generate plan for
            
        Returns:
            Generated plan steps
            
        Raises:
            Various exceptions based on failure type
        """
        self.logger.push_context(LogContext(
            component=ComponentType.PLANNER,
            operation="generate_plan_internal",
            metadata={"query_length": len(query)}
        ))
        
        try:
            # Generate prompt for the query
            self.logger.debug("Selecting prompt template for query")
            template = self.prompt_manager.get_template_for_query(query)
            
            if not template:
                error_msg = "No suitable template found for query type"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.logger.debug(f"Selected template type: {template.query_type.value}")
            
            # Generate plan using LLM
            try:
                prompt = template.format_prompt(query)
                self.logger.debug("Formatted prompt for LLM",
                                prompt_length=len(prompt))
                
                llm_start = asyncio.get_event_loop().time()
                plan_response = await self.llm.generate(prompt)
                llm_duration = (asyncio.get_event_loop().time() - llm_start) * 1000
                
                if not plan_response or not plan_response.strip():
                    error_msg = "LLM returned empty response"
                    self.logger.error(error_msg)
                    
                    # Log the LLM interaction failure
                    self.logger.log_llm_interaction(
                        operation="plan_generation",
                        prompt_length=len(prompt),
                        response_length=0,
                        duration_ms=llm_duration,
                        model_name=getattr(self.llm, 'model_name', 'unknown'),
                        success=False,
                        error=error_msg
                    )
                    
                    raise ValueError(error_msg)
                
                # Log successful LLM interaction
                self.logger.log_llm_interaction(
                    operation="plan_generation",
                    prompt_length=len(prompt),
                    response_length=len(plan_response),
                    duration_ms=llm_duration,
                    model_name=getattr(self.llm, 'model_name', 'unknown'),
                    success=True
                )
                
                # Parse the response into steps
                self.logger.debug("Parsing LLM response into plan steps",
                                response_length=len(plan_response))
                
                plan_steps = self._parse_plan_response(plan_response)
                
                if not plan_steps:
                    error_msg = "No valid plan steps extracted from LLM response"
                    self.logger.error(error_msg,
                                    response_preview=plan_response[:200] + "..." if len(plan_response) > 200 else plan_response)
                    raise ValueError(error_msg)
                
                self.logger.info(f"Successfully generated plan",
                               step_count=len(plan_steps),
                               llm_duration_ms=llm_duration)
                
                return plan_steps
                
            except Exception as e:
                self.logger.error(f"LLM plan generation failed: {str(e)}")
                raise e
                
        finally:
            self.logger.pop_context()
    
    def _parse_plan_response(self, response: str) -> List[str]:
        """Parse LLM response into plan steps with detailed logging.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of plan steps
        """
        self.logger.debug("Parsing plan response",
                        response_length=len(response),
                        response_preview=response[:100] + "..." if len(response) > 100 else response)
        
        steps = []
        lines = response.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and common headers
            if not line or line.lower() in ['plan:', 'steps:', 'action plan:', 'here is the plan:']:
                continue
            
            # Remove numbering and bullet points
            import re
            cleaned_line = re.sub(r'^[\d\.\-\*\+\s]+', '', line).strip()
            
            if cleaned_line and len(cleaned_line) > 5:  # Meaningful step
                steps.append(cleaned_line)
                self.logger.debug(f"Extracted step {len(steps)}",
                                line_number=line_num,
                                step_content=cleaned_line[:50] + "..." if len(cleaned_line) > 50 else cleaned_line)
        
        self.logger.info(f"Plan parsing completed",
                       total_lines=len(lines),
                       extracted_steps=len(steps),
                       success=len(steps) > 0)
        
        return steps
    
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