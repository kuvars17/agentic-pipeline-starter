"""Comprehensive logging configuration and utilities for the Planner node.

This module provides structured logging capabilities with performance metrics,
debug information, and observability features for the planning component.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
from functools import wraps
import sys


class LogLevel(Enum):
    """Enhanced log levels for different scenarios."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    PERFORMANCE = "PERFORMANCE"
    AUDIT = "AUDIT"


class ComponentType(Enum):
    """Different components for structured logging."""
    PLANNER = "PLANNER"
    LLM = "LLM"
    VALIDATOR = "VALIDATOR"
    ERROR_HANDLER = "ERROR_HANDLER"
    PROMPT_MANAGER = "PROMPT_MANAGER"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"


@dataclass
class LogContext:
    """Context information for structured logging.
    
    Attributes:
        component: Component generating the log
        operation: Current operation being performed
        conversation_id: Unique conversation identifier
        query_id: Unique query identifier
        user_id: User identifier (if available)
        session_id: Session identifier
        metadata: Additional context information
    """
    component: ComponentType
    operation: str
    conversation_id: Optional[str] = None
    query_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations.
    
    Attributes:
        operation_name: Name of the operation
        start_time: When the operation started
        end_time: When the operation completed
        duration_ms: Operation duration in milliseconds
        success: Whether operation succeeded
        error_type: Type of error if failed
        metadata: Additional performance data
    """
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, success: bool = True, error_type: Optional[str] = None) -> None:
        """Mark operation as complete and calculate duration.
        
        Args:
            success: Whether operation succeeded
            error_type: Type of error if failed
        """
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = success
        self.error_type = error_type


class PlannerLogger:
    """Comprehensive logger for the Planner node with structured logging."""
    
    def __init__(self, name: str = "PlannerNode"):
        """Initialize the planner logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.performance_metrics: List[PerformanceMetrics] = []
        self.context_stack: List[LogContext] = []
        
        # Configure structured logging
        self._configure_logger()
        
        self.logger.info("PlannerLogger initialized", extra={
            "component": ComponentType.PLANNER.value,
            "operation": "initialization"
        })
    
    def _configure_logger(self) -> None:
        """Configure the logger with structured formatting."""
        # Create formatter for structured logs
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(component)s | %(operation)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Check if handler already exists to avoid duplicates
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def push_context(self, context: LogContext) -> None:
        """Push a new logging context onto the stack.
        
        Args:
            context: Context to add
        """
        self.context_stack.append(context)
        self.logger.debug(f"Context pushed: {context.component.value}.{context.operation}", 
                         extra=self._get_log_extra(context))
    
    def pop_context(self) -> Optional[LogContext]:
        """Pop the current logging context from the stack.
        
        Returns:
            The popped context, if any
        """
        if self.context_stack:
            context = self.context_stack.pop()
            self.logger.debug(f"Context popped: {context.component.value}.{context.operation}",
                             extra=self._get_log_extra(context))
            return context
        return None
    
    def _get_current_context(self) -> Optional[LogContext]:
        """Get the current logging context.
        
        Returns:
            Current context or None
        """
        return self.context_stack[-1] if self.context_stack else None
    
    def _get_log_extra(self, context: Optional[LogContext] = None) -> Dict[str, Any]:
        """Get extra fields for logging.
        
        Args:
            context: Optional context to use, defaults to current
            
        Returns:
            Dictionary of extra fields
        """
        ctx = context or self._get_current_context()
        if not ctx:
            return {"component": "UNKNOWN", "operation": "unknown"}
        
        extra = {
            "component": ctx.component.value,
            "operation": ctx.operation
        }
        
        if ctx.conversation_id:
            extra["conversation_id"] = ctx.conversation_id
        if ctx.query_id:
            extra["query_id"] = ctx.query_id
        if ctx.user_id:
            extra["user_id"] = ctx.user_id
        if ctx.session_id:
            extra["session_id"] = ctx.session_id
        
        return extra
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context.
        
        Args:
            message: Log message
            **kwargs: Additional metadata
        """
        extra = self._get_log_extra()
        extra.update(kwargs)
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with context.
        
        Args:
            message: Log message
            **kwargs: Additional metadata
        """
        extra = self._get_log_extra()
        extra.update(kwargs)
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with context.
        
        Args:
            message: Log message
            **kwargs: Additional metadata
        """
        extra = self._get_log_extra()
        extra.update(kwargs)
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message with context.
        
        Args:
            message: Log message
            exc_info: Include exception information
            **kwargs: Additional metadata
        """
        extra = self._get_log_extra()
        extra.update(kwargs)
        self.logger.error(message, exc_info=exc_info, extra=extra)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs) -> None:
        """Log critical message with context.
        
        Args:
            message: Log message
            exc_info: Include exception information
            **kwargs: Additional metadata
        """
        extra = self._get_log_extra()
        extra.update(kwargs)
        self.logger.critical(message, exc_info=exc_info, extra=extra)
    
    def performance(self, message: str, duration_ms: float, **kwargs) -> None:
        """Log performance information.
        
        Args:
            message: Performance message
            duration_ms: Operation duration in milliseconds
            **kwargs: Additional performance metadata
        """
        extra = self._get_log_extra()
        extra.update({
            "duration_ms": duration_ms,
            "performance": True,
            **kwargs
        })
        self.logger.info(f"[PERFORMANCE] {message}", extra=extra)
    
    def audit(self, action: str, details: Dict[str, Any], **kwargs) -> None:
        """Log audit information for important actions.
        
        Args:
            action: Action being audited
            details: Details about the action
            **kwargs: Additional audit metadata
        """
        extra = self._get_log_extra()
        extra.update({
            "audit": True,
            "action": action,
            "details": json.dumps(details, default=str),
            **kwargs
        })
        self.logger.info(f"[AUDIT] {action}", extra=extra)
    
    def start_performance_tracking(self, operation_name: str, **metadata) -> PerformanceMetrics:
        """Start tracking performance for an operation.
        
        Args:
            operation_name: Name of the operation
            **metadata: Additional metadata
            
        Returns:
            Performance metrics object
        """
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            metadata=metadata
        )
        
        self.performance_metrics.append(metrics)
        
        self.debug(f"Started tracking: {operation_name}", 
                  operation_start=True, **metadata)
        
        return metrics
    
    def end_performance_tracking(self, metrics: PerformanceMetrics, 
                                success: bool = True, 
                                error_type: Optional[str] = None) -> None:
        """End performance tracking for an operation.
        
        Args:
            metrics: Performance metrics object
            success: Whether operation succeeded
            error_type: Type of error if failed
        """
        metrics.complete(success=success, error_type=error_type)
        
        log_func = self.performance if success else self.warning
        status = "completed" if success else f"failed ({error_type})"
        
        log_func(
            f"{metrics.operation_name} {status}",
            duration_ms=metrics.duration_ms,
            success=success,
            error_type=error_type,
            **metrics.metadata
        )
    
    def log_llm_interaction(self, 
                           operation: str,
                           prompt_length: int,
                           response_length: int,
                           duration_ms: float,
                           model_name: str,
                           success: bool = True,
                           error: Optional[str] = None) -> None:
        """Log LLM interaction details.
        
        Args:
            operation: Type of LLM operation
            prompt_length: Length of prompt in characters
            response_length: Length of response in characters
            duration_ms: Request duration
            model_name: Name of the model used
            success: Whether request succeeded
            error: Error message if failed
        """
        self.push_context(LogContext(
            component=ComponentType.LLM,
            operation=operation,
            conversation_id=self._get_current_context().conversation_id if self._get_current_context() else None
        ))
        
        try:
            if success:
                self.performance(
                    f"LLM {operation} successful",
                    duration_ms=duration_ms,
                    prompt_length=prompt_length,
                    response_length=response_length,
                    model_name=model_name,
                    tokens_per_second=response_length / (duration_ms / 1000) if duration_ms > 0 else 0
                )
            else:
                self.error(
                    f"LLM {operation} failed: {error}",
                    duration_ms=duration_ms,
                    prompt_length=prompt_length,
                    model_name=model_name,
                    error_message=error
                )
        finally:
            self.pop_context()
    
    def log_validation_result(self, 
                             plan_steps: List[str],
                             quality_score: float,
                             quality_level: str,
                             issues: List[str],
                             suggestions: List[str],
                             duration_ms: float) -> None:
        """Log plan validation results.
        
        Args:
            plan_steps: Plan steps that were validated
            quality_score: Calculated quality score
            quality_level: Quality level (EXCELLENT, GOOD, etc.)
            issues: List of identified issues
            suggestions: List of improvement suggestions
            duration_ms: Validation duration
        """
        self.push_context(LogContext(
            component=ComponentType.VALIDATOR,
            operation="validate_plan"
        ))
        
        try:
            self.info(
                f"Plan validation completed: {quality_level}",
                step_count=len(plan_steps),
                quality_score=quality_score,
                quality_level=quality_level,
                issue_count=len(issues),
                suggestion_count=len(suggestions),
                duration_ms=duration_ms
            )
            
            if issues:
                self.warning(
                    f"Plan validation issues detected",
                    issues=issues,
                    issue_count=len(issues)
                )
            
            if suggestions:
                self.debug(
                    f"Plan improvement suggestions available",
                    suggestions=suggestions,
                    suggestion_count=len(suggestions)
                )
                
        finally:
            self.pop_context()
    
    def log_error_handling(self,
                          error_type: str,
                          attempt_count: int,
                          fallback_strategy: str,
                          recovery_successful: bool,
                          duration_ms: float) -> None:
        """Log error handling and recovery attempts.
        
        Args:
            error_type: Type of error encountered
            attempt_count: Number of attempts made
            fallback_strategy: Strategy used for recovery
            recovery_successful: Whether recovery succeeded
            duration_ms: Total duration including retries
        """
        self.push_context(LogContext(
            component=ComponentType.ERROR_HANDLER,
            operation="handle_error"
        ))
        
        try:
            if recovery_successful:
                self.info(
                    f"Error recovery successful using {fallback_strategy}",
                    error_type=error_type,
                    attempt_count=attempt_count,
                    fallback_strategy=fallback_strategy,
                    duration_ms=duration_ms
                )
            else:
                self.error(
                    f"Error recovery failed after {attempt_count} attempts",
                    error_type=error_type,
                    attempt_count=attempt_count,
                    fallback_strategy=fallback_strategy,
                    duration_ms=duration_ms
                )
                
        finally:
            self.pop_context()
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours.
        
        Args:
            hours: Number of hours to include in summary
            
        Returns:
            Performance summary statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.performance_metrics 
            if m.start_time >= cutoff_time and m.end_time is not None
        ]
        
        if not recent_metrics:
            return {"message": f"No performance data in the last {hours} hours"}
        
        # Calculate statistics
        durations = [m.duration_ms for m in recent_metrics if m.duration_ms is not None]
        successful_ops = [m for m in recent_metrics if m.success]
        failed_ops = [m for m in recent_metrics if not m.success]
        
        # Group by operation type
        operation_stats = {}
        for metric in recent_metrics:
            op_name = metric.operation_name
            if op_name not in operation_stats:
                operation_stats[op_name] = {
                    "count": 0,
                    "successes": 0,
                    "failures": 0,
                    "avg_duration_ms": 0,
                    "min_duration_ms": float('inf'),
                    "max_duration_ms": 0
                }
            
            stats = operation_stats[op_name]
            stats["count"] += 1
            
            if metric.success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            
            if metric.duration_ms is not None:
                stats["avg_duration_ms"] = (stats["avg_duration_ms"] * (stats["count"] - 1) + metric.duration_ms) / stats["count"]
                stats["min_duration_ms"] = min(stats["min_duration_ms"], metric.duration_ms)
                stats["max_duration_ms"] = max(stats["max_duration_ms"], metric.duration_ms)
        
        return {
            "time_period_hours": hours,
            "total_operations": len(recent_metrics),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(recent_metrics) if recent_metrics else 0,
            "average_duration_ms": sum(durations) / len(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "operations_by_type": operation_stats
        }
    
    def clear_performance_history(self) -> None:
        """Clear performance history (useful for testing)."""
        self.performance_metrics.clear()
        self.info("Performance history cleared")


def performance_monitor(operation_name: str):
    """Decorator for automatic performance monitoring.
    
    Args:
        operation_name: Name of the operation to monitor
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Try to find a logger in the instance
            logger_instance = None
            if args and hasattr(args[0], 'logger') and isinstance(args[0].logger, PlannerLogger):
                logger_instance = args[0].logger
            
            if logger_instance:
                metrics = logger_instance.start_performance_tracking(operation_name)
                try:
                    result = await func(*args, **kwargs)
                    logger_instance.end_performance_tracking(metrics, success=True)
                    return result
                except Exception as e:
                    logger_instance.end_performance_tracking(
                        metrics, 
                        success=False, 
                        error_type=type(e).__name__
                    )
                    raise
            else:
                # No logger available, just execute function
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar logic for sync functions
            logger_instance = None
            if args and hasattr(args[0], 'logger') and isinstance(args[0].logger, PlannerLogger):
                logger_instance = args[0].logger
            
            if logger_instance:
                metrics = logger_instance.start_performance_tracking(operation_name)
                try:
                    result = func(*args, **kwargs)
                    logger_instance.end_performance_tracking(metrics, success=True)
                    return result
                except Exception as e:
                    logger_instance.end_performance_tracking(
                        metrics, 
                        success=False, 
                        error_type=type(e).__name__
                    )
                    raise
            else:
                # No logger available, just execute function
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator