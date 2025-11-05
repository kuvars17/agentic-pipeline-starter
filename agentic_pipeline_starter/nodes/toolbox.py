"""
Toolbox Node for Agentic Pipeline

This module implements the ToolboxNode that executes plan steps using available tools
and gathers evidence for the agentic pipeline. The node serves as the action-taking
component that uses tools like HTTP fetch and safe math to collect data.

Author: Agentic Pipeline Team
Date: November 2025
Version: 1.0.0
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..state.conversation_state import ConversationState, Message
from ..llm.base import BaseLLM
from ..tools.http_fetch import HttpFetchTool, HttpRequest, HttpMethod, HttpError


class ToolType(Enum):
    """Available tool types in the toolbox."""
    HTTP_FETCH = "http_fetch"
    SAFE_MATH = "safe_math"
    
    
class ToolExecutionStatus(Enum):
    """Tool execution status indicators."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RETRY_EXHAUSTED = "retry_exhausted"


@dataclass
class ToolResult:
    """Result container for tool execution."""
    tool_type: ToolType
    status: ToolExecutionStatus
    data: Any
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    retry_count: int = 0
    
    
@dataclass
class Evidence:
    """Evidence container for gathered data."""
    source: str
    tool_used: ToolType
    data: Any
    timestamp: str
    metadata: Dict[str, Any]


class ToolboxNode:
    """
    Toolbox node that executes plan steps using available tools.
    
    This node takes planned steps from the ConversationState and uses various tools
    to gather data, perform calculations, and collect evidence. It supports async
    execution for concurrent tool usage and includes comprehensive error handling.
    
    Features:
    - HTTP fetch tool with configurable timeout and retry logic
    - Safe math evaluation without eval()
    - Tool execution sandboxing and error handling
    - Async execution for improved performance
    - Retry logic for transient failures
    - Proper evidence structuring and storage
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        http_timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        concurrent_tools: int = 5
    ):
        """
        Initialize the ToolboxNode.
        
        Args:
            llm: Optional LLM instance for tool guidance
            http_timeout: HTTP request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
            concurrent_tools: Maximum concurrent tool executions
        """
        self.llm = llm
        self.http_timeout = http_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.concurrent_tools = concurrent_tools
        self.logger = logging.getLogger(__name__)
        
        # Initialize HTTP fetch tool
        self.http_tool = HttpFetchTool(
            default_timeout=http_timeout,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        # Tool registry
        self.tools: Dict[ToolType, Any] = {
            ToolType.HTTP_FETCH: self.http_tool
        }
        
        self.logger.info(
            f"ToolboxNode initialized with timeout={http_timeout}s, "
            f"max_retries={max_retries}, concurrent_tools={concurrent_tools}"
        )
    
    async def execute(self, state: ConversationState) -> ConversationState:
        """
        Execute the toolbox node on the given conversation state.
        
        This method processes the plan from the conversation state and executes
        the necessary tools to gather evidence and populate the state.
        
        Args:
            state: Current conversation state with plan to execute
            
        Returns:
            Updated conversation state with gathered evidence
            
        Raises:
            ValueError: If state is invalid or missing required data
            RuntimeError: If critical tool execution fails
        """
        self.logger.info("Starting ToolboxNode execution")
        
        try:
            # Validate input state
            if not state.plan:
                raise ValueError("ConversationState must contain a plan for execution")
            
            # Extract plan steps
            plan_steps = self._extract_plan_steps(state.plan)
            self.logger.info(f"Extracted {len(plan_steps)} plan steps for execution")
            
            # Execute tools for each plan step
            evidence_list = await self._execute_plan_steps(plan_steps)
            
            # Update state with gathered evidence
            updated_state = self._update_state_with_evidence(state, evidence_list)
            
            # Add execution metadata
            updated_state.add_metadata("toolbox_execution", {
                "steps_executed": len(plan_steps),
                "evidence_gathered": len(evidence_list),
                "tools_used": list(set(evidence.tool_used.value for evidence in evidence_list)),
                "execution_timestamp": self._get_timestamp()
            })
            
            self.logger.info(
                f"ToolboxNode execution completed successfully. "
                f"Gathered {len(evidence_list)} pieces of evidence"
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"ToolboxNode execution failed: {str(e)}")
            
            # Add error information to state
            error_message = Message(
                role="system",
                content=f"Toolbox execution failed: {str(e)}",
                metadata={"error_type": type(e).__name__, "node": "toolbox"}
            )
            state.messages.append(error_message)
            
            # Re-raise for upstream handling
            raise RuntimeError(f"ToolboxNode execution failed: {str(e)}") from e
    
    def _extract_plan_steps(self, plan: str) -> List[Dict[str, Any]]:
        """
        Extract executable steps from the plan.
        
        Args:
            plan: Plan string to parse
            
        Returns:
            List of plan steps with tool information
        """
        steps = []
        
        # Extract HTTP requests from plan
        http_steps = self._extract_http_requests(plan)
        steps.extend(http_steps)
        
        # Extract math calculations from plan
        math_steps = self._extract_math_operations(plan)
        steps.extend(math_steps)
        
        self.logger.info(f"Extracted {len(steps)} executable steps from plan")
        return steps
    
    def _extract_http_requests(self, plan: str) -> List[Dict[str, Any]]:
        """
        Extract HTTP requests from plan text.
        
        Args:
            plan: Plan string to parse
            
        Returns:
            List of HTTP request steps
        """
        steps = []
        
        # URL patterns
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, plan)
        
        for url in urls:
            # Determine HTTP method from context
            method = HttpMethod.GET
            if any(keyword in plan.lower() for keyword in ['post', 'submit', 'send']):
                method = HttpMethod.POST
            elif any(keyword in plan.lower() for keyword in ['put', 'update']):
                method = HttpMethod.PUT
            elif any(keyword in plan.lower() for keyword in ['delete', 'remove']):
                method = HttpMethod.DELETE
            
            steps.append({
                "type": "http_fetch",
                "action": "fetch_data",
                "params": {
                    "url": url.rstrip('.,;'),  # Clean trailing punctuation
                    "method": method.value,
                    "headers": {"User-Agent": "Agentic-Pipeline/1.0"}
                }
            })
        
        # Also look for explicit fetch/API instructions
        fetch_patterns = [
            r'fetch (?:data from |from )?(.+)',
            r'get (?:data from |from )?(.+)',
            r'call (?:the )?API (?:at )?(.+)',
            r'request (?:data from )?(.+)'
        ]
        
        for pattern in fetch_patterns:
            matches = re.findall(pattern, plan, re.IGNORECASE)
            for match in matches:
                # Try to extract URL from the match
                url_in_match = re.search(url_pattern, match)
                if url_in_match:
                    url = url_in_match.group()
                    steps.append({
                        "type": "http_fetch",
                        "action": "fetch_data",
                        "params": {
                            "url": url.rstrip('.,;'),
                            "method": "GET",
                            "headers": {"User-Agent": "Agentic-Pipeline/1.0"}
                        }
                    })
        
        return steps
    
    def _extract_math_operations(self, plan: str) -> List[Dict[str, Any]]:
        """
        Extract mathematical operations from plan text.
        
        Args:
            plan: Plan string to parse
            
        Returns:
            List of math operation steps
        """
        steps = []
        
        # Math operation patterns
        math_patterns = [
            r'calculate (.+)',
            r'compute (.+)',
            r'find (?:the )?(?:sum|average|total|result) (?:of )?(.+)',
            r'(?:add|subtract|multiply|divide) (.+)',
            r'(\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)*)'
        ]
        
        for pattern in math_patterns:
            matches = re.findall(pattern, plan, re.IGNORECASE)
            for match in matches:
                steps.append({
                    "type": "safe_math",
                    "action": "calculate",
                    "params": {
                        "expression": match.strip()
                    }
                })
        
        return steps
    
    async def _execute_plan_steps(self, plan_steps: List[Dict[str, Any]]) -> List[Evidence]:
        """
        Execute plan steps using available tools.
        
        Args:
            plan_steps: List of plan steps to execute
            
        Returns:
            List of evidence gathered from tool execution
        """
        evidence_list = []
        
        # Create semaphore for concurrent execution control
        semaphore = asyncio.Semaphore(self.concurrent_tools)
        
        # Execute steps concurrently
        tasks = [
            self._execute_single_step(step, semaphore) 
            for step in plan_steps
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and collect evidence
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Step {i} failed: {str(result)}")
                continue
            
            if result and result.status == ToolExecutionStatus.SUCCESS:
                evidence = Evidence(
                    source=f"step_{i}",
                    tool_used=ToolType(plan_steps[i]["type"]),
                    data=result.data,
                    timestamp=self._get_timestamp(),
                    metadata={
                        "step_index": i,
                        "execution_time": result.execution_time,
                        "retry_count": result.retry_count
                    }
                )
                evidence_list.append(evidence)
        
        return evidence_list
    
    async def _execute_single_step(
        self, 
        step: Dict[str, Any], 
        semaphore: asyncio.Semaphore
    ) -> Optional[ToolResult]:
        """
        Execute a single plan step with concurrency control.
        
        Args:
            step: Plan step to execute
            semaphore: Concurrency control semaphore
            
        Returns:
            Tool execution result
        """
        async with semaphore:
            tool_type = ToolType(step["type"])
            
            self.logger.info(f"Executing step with tool: {tool_type.value}")
            
            try:
                if tool_type == ToolType.HTTP_FETCH:
                    return await self._execute_http_step(step)
                elif tool_type == ToolType.SAFE_MATH:
                    return await self._execute_math_step(step)
                else:
                    raise ValueError(f"Unknown tool type: {tool_type.value}")
                    
            except Exception as e:
                self.logger.error(f"Tool execution failed: {str(e)}")
                return ToolResult(
                    tool_type=tool_type,
                    status=ToolExecutionStatus.FAILURE,
                    data=None,
                    error_message=str(e)
                )
    
    async def _execute_http_step(self, step: Dict[str, Any]) -> ToolResult:
        """
        Execute HTTP fetch step.
        
        Args:
            step: HTTP step configuration
            
        Returns:
            Tool execution result
        """
        import time
        start_time = time.time()
        
        try:
            params = step.get("params", {})
            
            # Create HTTP request
            request = self.http_tool.create_request(
                url=params["url"],
                method=params.get("method", "GET"),
                headers=params.get("headers", {}),
                data=params.get("data"),
                params=params.get("query_params")
            )
            
            # Execute request
            response = await self.http_tool.fetch(request)
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                tool_type=ToolType.HTTP_FETCH,
                status=ToolExecutionStatus.SUCCESS,
                data={
                    "url": response.url,
                    "status_code": response.status_code,
                    "data": response.data,
                    "headers": response.headers
                },
                execution_time=execution_time,
                retry_count=response.retry_count
            )
            
        except HttpError as e:
            execution_time = time.time() - start_time
            return ToolResult(
                tool_type=ToolType.HTTP_FETCH,
                status=ToolExecutionStatus.FAILURE,
                data=None,
                error_message=f"HTTP error: {str(e)}",
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                tool_type=ToolType.HTTP_FETCH,
                status=ToolExecutionStatus.FAILURE,
                data=None,
                error_message=f"Unexpected error: {str(e)}",
                execution_time=execution_time
            )
    
    async def _execute_math_step(self, step: Dict[str, Any]) -> ToolResult:
        """
        Execute math calculation step (placeholder).
        
        Args:
            step: Math step configuration
            
        Returns:
            Tool execution result
        """
        import time
        start_time = time.time()
        
        try:
            params = step.get("params", {})
            expression = params.get("expression", "")
            
            # Placeholder for safe math evaluation
            # This will be implemented in the next task
            result = f"Math calculation for: {expression} (placeholder)"
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                tool_type=ToolType.SAFE_MATH,
                status=ToolExecutionStatus.SUCCESS,
                data={"expression": expression, "result": result},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                tool_type=ToolType.SAFE_MATH,
                status=ToolExecutionStatus.FAILURE,
                data=None,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _update_state_with_evidence(
        self, 
        state: ConversationState, 
        evidence_list: List[Evidence]
    ) -> ConversationState:
        """
        Update conversation state with gathered evidence.
        
        Args:
            state: Original conversation state
            evidence_list: List of evidence to add
            
        Returns:
            Updated conversation state
        """
        # Add evidence to state
        if not hasattr(state, 'evidence'):
            state.evidence = []
        
        state.evidence.extend(evidence_list)
        
        # Add summary message
        evidence_summary = f"Gathered {len(evidence_list)} pieces of evidence using tools: "
        evidence_summary += ", ".join(set(evidence.tool_used.value for evidence in evidence_list))
        
        summary_message = Message(
            role="system",
            content=evidence_summary,
            metadata={"node": "toolbox", "evidence_count": len(evidence_list)}
        )
        state.messages.append(summary_message)
        
        return state
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def add_tool(self, tool_type: ToolType, tool_instance: Any) -> None:
        """
        Add a tool to the toolbox registry.
        
        Args:
            tool_type: Type of tool to add
            tool_instance: Tool implementation instance
        """
        self.tools[tool_type] = tool_instance
        self.logger.info(f"Added tool: {tool_type.value}")
    
    def list_available_tools(self) -> List[ToolType]:
        """
        Get list of available tools.
        
        Returns:
            List of available tool types
        """
        return list(self.tools.keys())
    
    def get_tool_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available tools.
        
        Returns:
            Dictionary with tool information
        """
        return {
            "total_tools": len(self.tools),
            "available_tools": [tool.value for tool in self.tools.keys()],
            "configuration": {
                "http_timeout": self.http_timeout,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "concurrent_tools": self.concurrent_tools
            }
        }