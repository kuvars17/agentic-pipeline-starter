"""Async interface for LangGraph integration.

This module provides the async interface layer that enables the Planner node
to integrate seamlessly with LangGraph's async execution model and state management.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import time
from datetime import datetime

from ..state.conversation_state import ConversationState, Message, MessageRole
from .planner import PlannerNode
from .planner_logger import PlannerLogger, LogContext, ComponentType


T = TypeVar('T')


class NodeState(Enum):
    """States for async node execution."""
    IDLE = "idle"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionPriority(Enum):
    """Priority levels for async execution."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AsyncExecutionContext:
    """Context for async execution tracking.
    
    Attributes:
        execution_id: Unique identifier for this execution
        node_name: Name of the executing node
        priority: Execution priority level
        timeout: Maximum execution time in seconds
        started_at: When execution started
        state: Current execution state
        metadata: Additional context information
    """
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_name: str = "planner"
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    timeout: float = 60.0
    started_at: Optional[datetime] = None
    state: NodeState = NodeState.IDLE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AsyncExecutionResult:
    """Result of async execution.
    
    Attributes:
        success: Whether execution was successful
        state: Updated conversation state
        execution_time: Time taken in seconds
        error: Error information if failed
        metadata: Additional result information
    """
    success: bool
    state: ConversationState
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncPlannerInterface:
    """Async interface for PlannerNode integration with LangGraph.
    
    This class provides the async interface layer that makes the PlannerNode
    compatible with LangGraph's async execution model. It handles state
    management, execution tracking, and provides hooks for LangGraph integration.
    """
    
    def __init__(self, 
                 planner_node: Optional[PlannerNode] = None,
                 max_concurrent_executions: int = 5,
                 default_timeout: float = 60.0):
        """Initialize the async interface.
        
        Args:
            planner_node: PlannerNode instance to wrap
            max_concurrent_executions: Maximum concurrent executions allowed
            default_timeout: Default execution timeout in seconds
        """
        self.planner_node = planner_node or PlannerNode()
        self.max_concurrent_executions = max_concurrent_executions
        self.default_timeout = default_timeout
        
        # Execution tracking
        self.active_executions: Dict[str, AsyncExecutionContext] = {}
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self.execution_history: List[AsyncExecutionResult] = []
        
        # Logging
        self.logger = PlannerLogger("AsyncPlannerInterface")
        
        self.logger.info("AsyncPlannerInterface initialized",
                        max_concurrent=max_concurrent_executions,
                        default_timeout=default_timeout)
    
    async def __call__(self, state: ConversationState, **kwargs) -> ConversationState:
        """Make the interface callable for LangGraph compatibility.
        
        This method provides the standard LangGraph node interface.
        
        Args:
            state: Current conversation state
            **kwargs: Additional execution parameters
            
        Returns:
            Updated conversation state
        """
        result = await self.execute_async(state, **kwargs)
        if not result.success:
            state.add_error(f"Async execution failed: {result.error}")
        return result.state
    
    async def execute_async(self, 
                          state: ConversationState,
                          priority: ExecutionPriority = ExecutionPriority.NORMAL,
                          timeout: Optional[float] = None,
                          execution_id: Optional[str] = None,
                          **kwargs) -> AsyncExecutionResult:
        """Execute planner asynchronously with full tracking.
        
        Args:
            state: Current conversation state
            priority: Execution priority
            timeout: Execution timeout (uses default if None)
            execution_id: Optional custom execution ID
            **kwargs: Additional execution parameters
            
        Returns:
            Async execution result
        """
        # Create execution context
        context = AsyncExecutionContext(
            execution_id=execution_id or str(uuid.uuid4()),
            node_name="planner",
            priority=priority,
            timeout=timeout or self.default_timeout,
            started_at=datetime.utcnow(),
            state=NodeState.IDLE,
            metadata=kwargs
        )
        
        # Set up logging context
        log_context = LogContext(
            component=ComponentType.PLANNER,
            operation="async_execution",
            conversation_id=str(state.conversation_id),
            metadata={
                "execution_id": context.execution_id,
                "priority": priority.name,
                "timeout": context.timeout
            }
        )
        self.logger.push_context(log_context)
        
        try:
            self.logger.info("Starting async execution",
                           execution_id=context.execution_id,
                           priority=priority.name,
                           timeout=context.timeout)
            
            # Acquire semaphore for concurrency control
            async with self.execution_semaphore:
                # Track execution
                self.active_executions[context.execution_id] = context
                context.state = NodeState.RUNNING
                
                try:
                    # Execute with timeout
                    start_time = time.time()
                    
                    updated_state = await asyncio.wait_for(
                        self.planner_node.execute(state),
                        timeout=context.timeout
                    )
                    
                    execution_time = time.time() - start_time
                    context.state = NodeState.COMPLETED
                    
                    # Create successful result
                    result = AsyncExecutionResult(
                        success=True,
                        state=updated_state,
                        execution_time=execution_time,
                        metadata={
                            "execution_id": context.execution_id,
                            "priority": priority.name,
                            "plan_steps": len(updated_state.plan) if updated_state.plan else 0
                        }
                    )
                    
                    self.logger.info("Async execution completed successfully",
                                   execution_id=context.execution_id,
                                   execution_time=execution_time,
                                   plan_steps=len(updated_state.plan) if updated_state.plan else 0)
                    
                    return result
                    
                except asyncio.TimeoutError:
                    context.state = NodeState.FAILED
                    error_msg = f"Execution timed out after {context.timeout}s"
                    
                    self.logger.error("Async execution timeout",
                                    execution_id=context.execution_id,
                                    timeout=context.timeout)
                    
                    return AsyncExecutionResult(
                        success=False,
                        state=state,
                        execution_time=context.timeout,
                        error=error_msg,
                        metadata={"execution_id": context.execution_id, "error_type": "timeout"}
                    )
                    
                except asyncio.CancelledError:
                    context.state = NodeState.CANCELLED
                    error_msg = "Execution was cancelled"
                    
                    self.logger.warning("Async execution cancelled",
                                      execution_id=context.execution_id)
                    
                    return AsyncExecutionResult(
                        success=False,
                        state=state,
                        execution_time=time.time() - start_time if 'start_time' in locals() else 0,
                        error=error_msg,
                        metadata={"execution_id": context.execution_id, "error_type": "cancelled"}
                    )
                    
                except Exception as e:
                    context.state = NodeState.FAILED
                    error_msg = f"Execution failed: {str(e)}"
                    
                    self.logger.error("Async execution failed",
                                    execution_id=context.execution_id,
                                    error=str(e),
                                    exc_info=True)
                    
                    return AsyncExecutionResult(
                        success=False,
                        state=state,
                        execution_time=time.time() - start_time if 'start_time' in locals() else 0,
                        error=error_msg,
                        metadata={"execution_id": context.execution_id, "error_type": type(e).__name__}
                    )
                    
                finally:
                    # Clean up tracking
                    if context.execution_id in self.active_executions:
                        del self.active_executions[context.execution_id]
                    
                    # Add to history (keep last 100)
                    self.execution_history.append(result if 'result' in locals() else AsyncExecutionResult(
                        success=False,
                        state=state,
                        execution_time=0,
                        error="Unknown error during cleanup"
                    ))
                    if len(self.execution_history) > 100:
                        self.execution_history = self.execution_history[-100:]
        
        finally:
            self.logger.pop_context()
    
    async def execute_batch(self, 
                          states: List[ConversationState],
                          priority: ExecutionPriority = ExecutionPriority.NORMAL,
                          timeout: Optional[float] = None) -> List[AsyncExecutionResult]:
        """Execute multiple states in parallel.
        
        Args:
            states: List of conversation states to process
            priority: Execution priority for all executions
            timeout: Timeout for each individual execution
            
        Returns:
            List of execution results
        """
        self.logger.info(f"Starting batch execution",
                       batch_size=len(states),
                       priority=priority.name)
        
        # Create tasks for parallel execution
        tasks = [
            self.execute_async(state, priority=priority, timeout=timeout)
            for state in states
        ]
        
        # Execute all tasks and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(AsyncExecutionResult(
                    success=False,
                    state=states[i],
                    execution_time=0,
                    error=f"Batch execution exception: {str(result)}",
                    metadata={"batch_index": i, "error_type": type(result).__name__}
                ))
            else:
                processed_results.append(result)
        
        successful_count = sum(1 for r in processed_results if r.success)
        self.logger.info(f"Batch execution completed",
                       total=len(states),
                       successful=successful_count,
                       failed=len(states) - successful_count)
        
        return processed_results
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution.
        
        Args:
            execution_id: ID of execution to cancel
            
        Returns:
            True if cancellation was successful
        """
        if execution_id not in self.active_executions:
            self.logger.warning(f"Cannot cancel execution - not found",
                              execution_id=execution_id)
            return False
        
        # Note: In a real implementation, you'd need to track and cancel
        # the actual asyncio tasks. This is a simplified version.
        context = self.active_executions[execution_id]
        context.state = NodeState.CANCELLED
        
        self.logger.info(f"Execution cancelled",
                       execution_id=execution_id)
        
        return True
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a running execution.
        
        Args:
            execution_id: ID of execution to check
            
        Returns:
            Execution status information or None if not found
        """
        if execution_id not in self.active_executions:
            return None
        
        context = self.active_executions[execution_id]
        elapsed = (datetime.utcnow() - context.started_at).total_seconds() if context.started_at else 0
        
        return {
            "execution_id": context.execution_id,
            "node_name": context.node_name,
            "state": context.state.value,
            "priority": context.priority.name,
            "elapsed_time": elapsed,
            "timeout": context.timeout,
            "metadata": context.metadata
        }
    
    def get_all_execution_statuses(self) -> List[Dict[str, Any]]:
        """Get status of all active executions.
        
        Returns:
            List of execution status information
        """
        return [
            self.get_execution_status(execution_id)
            for execution_id in self.active_executions
        ]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics and performance metrics.
        
        Returns:
            Statistics about executions
        """
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0,
                "average_execution_time": 0,
                "active_executions": len(self.active_executions)
            }
        
        successful = [r for r in self.execution_history if r.success]
        failed = [r for r in self.execution_history if not r.success]
        
        execution_times = [r.execution_time for r in self.execution_history if r.execution_time > 0]
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Error type analysis
        error_types = {}
        for result in failed:
            if result.metadata and "error_type" in result.metadata:
                error_type = result.metadata["error_type"]
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful),
            "failed_executions": len(failed),
            "success_rate": len(successful) / len(self.execution_history),
            "average_execution_time": avg_time,
            "min_execution_time": min(execution_times) if execution_times else 0,
            "max_execution_time": max(execution_times) if execution_times else 0,
            "active_executions": len(self.active_executions),
            "error_types": error_types,
            "max_concurrent_limit": self.max_concurrent_executions,
            "semaphore_available": self.execution_semaphore._value
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the async interface.
        
        Returns:
            Health status information
        """
        start_time = time.time()
        
        try:
            # Create a simple test state
            test_state = ConversationState()
            test_state.add_message(Message(
                role=MessageRole.USER,
                content="Health check test query"
            ))
            
            # Execute a quick test
            result = await self.execute_async(
                test_state,
                timeout=10.0,
                priority=ExecutionPriority.LOW
            )
            
            health_time = time.time() - start_time
            
            return {
                "status": "healthy" if result.success else "degraded",
                "response_time": health_time,
                "planner_available": True,
                "execution_successful": result.success,
                "active_executions": len(self.active_executions),
                "max_concurrent": self.max_concurrent_executions,
                "semaphore_available": self.execution_semaphore._value,
                "error": result.error if not result.success else None
            }
            
        except Exception as e:
            health_time = time.time() - start_time
            
            return {
                "status": "unhealthy",
                "response_time": health_time,
                "planner_available": False,
                "execution_successful": False,
                "error": str(e),
                "active_executions": len(self.active_executions),
                "max_concurrent": self.max_concurrent_executions
            }
    
    def clear_execution_history(self) -> None:
        """Clear execution history (useful for testing)."""
        self.execution_history.clear()
        self.logger.info("Execution history cleared")


class LangGraphNodeWrapper:
    """Wrapper to make AsyncPlannerInterface compatible with LangGraph node signature.
    
    This wrapper ensures the interface works seamlessly with LangGraph's expected
    node signature and state management patterns.
    """
    
    def __init__(self, async_interface: Optional[AsyncPlannerInterface] = None):
        """Initialize the wrapper.
        
        Args:
            async_interface: AsyncPlannerInterface instance to wrap
        """
        self.async_interface = async_interface or AsyncPlannerInterface()
        self.logger = PlannerLogger("LangGraphNodeWrapper")
        
        self.logger.info("LangGraphNodeWrapper initialized")
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph-compatible node function.
        
        Args:
            state: LangGraph state dictionary
            
        Returns:
            Updated LangGraph state dictionary
        """
        try:
            # Convert LangGraph state to ConversationState
            conversation_state = self._dict_to_conversation_state(state)
            
            # Execute through async interface
            result = await self.async_interface.execute_async(conversation_state)
            
            if result.success:
                # Convert back to LangGraph state format
                updated_dict = self._conversation_state_to_dict(result.state)
                
                # Add execution metadata
                updated_dict["_execution_metadata"] = {
                    "success": True,
                    "execution_time": result.execution_time,
                    "execution_id": result.metadata.get("execution_id"),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                return updated_dict
            else:
                # Handle failure
                self.logger.error("LangGraph execution failed", error=result.error)
                
                # Return original state with error information
                error_state = state.copy()
                error_state["_execution_metadata"] = {
                    "success": False,
                    "error": result.error,
                    "execution_time": result.execution_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                return error_state
                
        except Exception as e:
            self.logger.error("Critical error in LangGraph wrapper", 
                            error=str(e), exc_info=True)
            
            # Return original state with error
            error_state = state.copy()
            error_state["_execution_metadata"] = {
                "success": False,
                "error": f"Wrapper error: {str(e)}",
                "execution_time": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return error_state
    
    def _dict_to_conversation_state(self, state_dict: Dict[str, Any]) -> ConversationState:
        """Convert LangGraph state dict to ConversationState.
        
        Args:
            state_dict: LangGraph state dictionary
            
        Returns:
            ConversationState instance
        """
        # Create new conversation state
        conversation_state = ConversationState()
        
        # Extract messages if present
        if "messages" in state_dict:
            for msg_data in state_dict["messages"]:
                if isinstance(msg_data, dict):
                    message = Message(
                        role=MessageRole(msg_data.get("role", "user")),
                        content=msg_data.get("content", ""),
                        metadata=msg_data.get("metadata", {})
                    )
                    conversation_state.add_message(message)
        
        # Extract other fields
        if "plan" in state_dict:
            conversation_state.plan = state_dict["plan"]
        
        if "metadata" in state_dict:
            conversation_state.metadata.update(state_dict["metadata"])
        
        if "errors" in state_dict:
            conversation_state.errors.extend(state_dict["errors"])
        
        return conversation_state
    
    def _conversation_state_to_dict(self, state: ConversationState) -> Dict[str, Any]:
        """Convert ConversationState to LangGraph state dict.
        
        Args:
            state: ConversationState instance
            
        Returns:
            LangGraph compatible state dictionary
        """
        return {
            "conversation_id": str(state.conversation_id),
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "metadata": msg.metadata,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in state.messages
            ],
            "plan": state.plan,
            "metadata": state.metadata,
            "errors": state.errors,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat()
        }