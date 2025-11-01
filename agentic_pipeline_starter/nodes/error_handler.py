"""Error handling and fallback strategies for the Planner node.

This module provides comprehensive error handling, recovery mechanisms,
and fallback strategies to ensure robust plan generation even when
facing various failure scenarios.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur during planning."""
    LLM_CONNECTION_ERROR = "llm_connection_error"
    LLM_TIMEOUT_ERROR = "llm_timeout_error"
    LLM_RESPONSE_ERROR = "llm_response_error"
    VALIDATION_ERROR = "validation_error"
    PROMPT_ERROR = "prompt_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNEXPECTED_ERROR = "unexpected_error"


class FallbackStrategy(Enum):
    """Available fallback strategies for error recovery."""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    SWITCH_LLM_BACKEND = "switch_llm_backend"
    USE_TEMPLATE_PLAN = "use_template_plan"
    USE_SIMPLE_PLAN = "use_simple_plan"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class ErrorContext:
    """Context information for error handling.
    
    Attributes:
        error_type: Classification of the error
        original_error: The original exception that occurred
        query: User query being processed
        attempt_count: Number of attempts made
        timestamp: When the error occurred
        metadata: Additional context information
    """
    error_type: ErrorType
    original_error: Exception
    query: str
    attempt_count: int
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class RetryConfig:
    """Configuration for retry behavior.
    
    Attributes:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff
        timeout: Timeout for individual attempts
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    backoff_multiplier: float = 2.0
    timeout: float = 30.0


class PlannerErrorHandler:
    """Comprehensive error handler for the Planner node."""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """Initialize the error handler.
        
        Args:
            retry_config: Configuration for retry behavior
        """
        self.retry_config = retry_config or RetryConfig()
        self.error_history: List[ErrorContext] = []
        self.fallback_templates = self._create_fallback_templates()
        
        logger.info("PlannerErrorHandler initialized")
    
    async def handle_with_fallback(self,
                                 operation: Callable,
                                 query: str,
                                 *args,
                                 **kwargs) -> tuple[bool, Union[str, List[str], Exception]]:
        """Execute an operation with comprehensive error handling and fallback.
        
        Args:
            operation: The async operation to execute
            query: User query being processed
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Tuple of (success: bool, result: Union[str, List[str], Exception])
        """
        attempt_count = 0
        last_error = None
        
        while attempt_count < self.retry_config.max_attempts:
            attempt_count += 1
            
            try:
                # Execute the operation with timeout
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=self.retry_config.timeout
                )
                
                # Log successful recovery if this wasn't the first attempt
                if attempt_count > 1:
                    logger.info(f"Operation succeeded on attempt {attempt_count}")
                
                return True, result
                
            except asyncio.TimeoutError as e:
                error_type = ErrorType.LLM_TIMEOUT_ERROR
                last_error = e
                logger.warning(f"Timeout on attempt {attempt_count}: {str(e)}")
                
            except ConnectionError as e:
                error_type = ErrorType.LLM_CONNECTION_ERROR
                last_error = e
                logger.warning(f"Connection error on attempt {attempt_count}: {str(e)}")
                
            except ValueError as e:
                error_type = ErrorType.VALIDATION_ERROR
                last_error = e
                logger.warning(f"Validation error on attempt {attempt_count}: {str(e)}")
                
            except Exception as e:
                error_type = ErrorType.UNEXPECTED_ERROR
                last_error = e
                logger.error(f"Unexpected error on attempt {attempt_count}: {str(e)}", exc_info=True)
            
            # Record the error
            error_context = ErrorContext(
                error_type=error_type,
                original_error=last_error,
                query=query,
                attempt_count=attempt_count,
                timestamp=datetime.utcnow(),
                metadata={"operation": operation.__name__ if hasattr(operation, '__name__') else str(operation)}
            )
            self.error_history.append(error_context)
            
            # Determine if we should retry
            if not self._should_retry(error_type, attempt_count):
                break
            
            # Wait before retry with exponential backoff
            if attempt_count < self.retry_config.max_attempts:
                delay = self._calculate_retry_delay(attempt_count)
                logger.info(f"Waiting {delay:.1f}s before retry {attempt_count + 1}")
                await asyncio.sleep(delay)
        
        # All attempts failed, use fallback strategy
        logger.error(f"All {attempt_count} attempts failed for operation")
        fallback_result = await self._execute_fallback_strategy(query, error_type, last_error)
        
        return False, fallback_result
    
    def _should_retry(self, error_type: ErrorType, attempt_count: int) -> bool:
        """Determine if an error should trigger a retry.
        
        Args:
            error_type: Type of error that occurred
            attempt_count: Current attempt number
            
        Returns:
            True if retry should be attempted
        """
        if attempt_count >= self.retry_config.max_attempts:
            return False
        
        # Don't retry validation errors - they likely won't improve
        if error_type == ErrorType.VALIDATION_ERROR:
            return False
        
        # Don't retry configuration errors
        if error_type == ErrorType.CONFIGURATION_ERROR:
            return False
        
        # Retry connection and timeout errors
        if error_type in [ErrorType.LLM_CONNECTION_ERROR, ErrorType.LLM_TIMEOUT_ERROR]:
            return True
        
        # Retry LLM response errors (might be transient)
        if error_type == ErrorType.LLM_RESPONSE_ERROR:
            return True
        
        # Retry unexpected errors once more
        if error_type == ErrorType.UNEXPECTED_ERROR:
            return attempt_count == 1
        
        return False
    
    def _calculate_retry_delay(self, attempt_count: int) -> float:
        """Calculate delay for retry with exponential backoff.
        
        Args:
            attempt_count: Current attempt number
            
        Returns:
            Delay in seconds
        """
        delay = self.retry_config.base_delay * (self.retry_config.backoff_multiplier ** (attempt_count - 1))
        return min(delay, self.retry_config.max_delay)
    
    async def _execute_fallback_strategy(self,
                                       query: str,
                                       error_type: ErrorType,
                                       last_error: Exception) -> List[str]:
        """Execute appropriate fallback strategy based on error type.
        
        Args:
            query: User query being processed
            error_type: Type of error that occurred
            last_error: The last error that occurred
            
        Returns:
            Fallback plan steps
        """
        logger.info(f"Executing fallback strategy for {error_type.value}")
        
        try:
            # Determine best fallback strategy
            strategy = self._select_fallback_strategy(error_type)
            
            if strategy == FallbackStrategy.USE_TEMPLATE_PLAN:
                return self._get_template_plan(query)
            elif strategy == FallbackStrategy.USE_SIMPLE_PLAN:
                return self._get_simple_plan(query)
            elif strategy == FallbackStrategy.GRACEFUL_DEGRADATION:
                return self._get_minimal_plan()
            else:
                # Default to simple plan
                return self._get_simple_plan(query)
                
        except Exception as e:
            logger.error(f"Fallback strategy failed: {str(e)}", exc_info=True)
            return self._get_emergency_plan()
    
    def _select_fallback_strategy(self, error_type: ErrorType) -> FallbackStrategy:
        """Select the most appropriate fallback strategy.
        
        Args:
            error_type: Type of error that occurred
            
        Returns:
            Best fallback strategy for the error type
        """
        strategy_mapping = {
            ErrorType.LLM_CONNECTION_ERROR: FallbackStrategy.USE_TEMPLATE_PLAN,
            ErrorType.LLM_TIMEOUT_ERROR: FallbackStrategy.USE_TEMPLATE_PLAN,
            ErrorType.LLM_RESPONSE_ERROR: FallbackStrategy.USE_SIMPLE_PLAN,
            ErrorType.VALIDATION_ERROR: FallbackStrategy.USE_SIMPLE_PLAN,
            ErrorType.PROMPT_ERROR: FallbackStrategy.USE_TEMPLATE_PLAN,
            ErrorType.CONFIGURATION_ERROR: FallbackStrategy.GRACEFUL_DEGRADATION,
            ErrorType.UNEXPECTED_ERROR: FallbackStrategy.GRACEFUL_DEGRADATION,
        }
        
        return strategy_mapping.get(error_type, FallbackStrategy.USE_SIMPLE_PLAN)
    
    def _create_fallback_templates(self) -> Dict[str, List[str]]:
        """Create templates for different types of queries.
        
        Returns:
            Dictionary of query type -> plan template mappings
        """
        return {
            "factual": [
                "Search for reliable information sources about the topic.",
                "Verify information from multiple authoritative sources.",
                "Cross-reference facts to ensure accuracy.",
                "Present the verified information clearly."
            ],
            "mathematical": [
                "Parse the mathematical problem or expression.",
                "Identify the required mathematical operations.",
                "Perform calculations step by step.",
                "Verify the result and present the answer."
            ],
            "comparison": [
                "Identify the items or concepts to be compared.",
                "Research key characteristics of each option.",
                "Analyze differences and similarities systematically.",
                "Present a balanced comparison with conclusions."
            ],
            "procedural": [
                "Break down the process into logical steps.",
                "Research best practices and requirements.",
                "Organize steps in chronological order.",
                "Provide clear, actionable instructions."
            ],
            "general": [
                "Analyze the question to understand requirements.",
                "Research relevant information from reliable sources.",
                "Organize findings in a logical structure.",
                "Present a comprehensive and helpful response."
            ]
        }
    
    def _get_template_plan(self, query: str) -> List[str]:
        """Get a template-based plan for the query.
        
        Args:
            query: User query to create plan for
            
        Returns:
            Template-based plan steps
        """
        query_lower = query.lower()
        
        # Determine query type and return appropriate template
        if any(word in query_lower for word in ["what is", "who is", "where is", "when"]):
            return self.fallback_templates["factual"]
        elif any(word in query_lower for word in ["calculate", "solve", "math", "equation"]):
            return self.fallback_templates["mathematical"]
        elif any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            return self.fallback_templates["comparison"]
        elif any(word in query_lower for word in ["how to", "steps", "process", "method"]):
            return self.fallback_templates["procedural"]
        else:
            return self.fallback_templates["general"]
    
    def _get_simple_plan(self, query: str) -> List[str]:
        """Get a simple, generic plan.
        
        Args:
            query: User query to create plan for
            
        Returns:
            Simple plan steps
        """
        return [
            "Understand and analyze the user's question.",
            "Gather necessary information to answer the question.",
            "Process and organize the information logically.",
            "Provide a clear and helpful response."
        ]
    
    def _get_minimal_plan(self) -> List[str]:
        """Get the most basic plan possible.
        
        Returns:
            Minimal plan steps
        """
        return [
            "Process the user's request.",
            "Provide the best possible response."
        ]
    
    def _get_emergency_plan(self) -> List[str]:
        """Get emergency plan when all else fails.
        
        Returns:
            Emergency plan steps
        """
        return [
            "Acknowledge the user's request.",
            "Attempt to provide helpful information based on available resources."
        ]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error history for debugging.
        
        Returns:
            Summary of error patterns and statistics
        """
        if not self.error_history:
            return {"total_errors": 0, "error_types": {}, "recent_errors": []}
        
        # Count error types
        error_counts = {}
        for error in self.error_history:
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Get recent errors (last 10)
        recent_errors = [
            {
                "type": error.error_type.value,
                "query": error.query[:50] + "..." if len(error.query) > 50 else error.query,
                "timestamp": error.timestamp.isoformat(),
                "attempt": error.attempt_count
            }
            for error in self.error_history[-10:]
        ]
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_counts,
            "recent_errors": recent_errors,
            "retry_config": {
                "max_attempts": self.retry_config.max_attempts,
                "base_delay": self.retry_config.base_delay,
                "timeout": self.retry_config.timeout
            }
        }
    
    def clear_error_history(self) -> None:
        """Clear the error history (useful for testing)."""
        self.error_history.clear()
        logger.info("Error history cleared")


class CircuitBreaker:
    """Circuit breaker pattern implementation for LLM calls."""
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type to monitor
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        logger.info(f"CircuitBreaker initialized with threshold {failure_threshold}")
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker state: HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state.
        
        Returns:
            Current state information
        """
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }