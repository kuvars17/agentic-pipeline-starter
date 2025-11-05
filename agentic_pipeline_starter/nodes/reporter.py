"""
Reporter Node for Agentic Pipeline

This module implements the ReporterNode that synthesizes all gathered information
into a final, human-readable answer using LLM-based summarization. The node serves
as the communication component of the agentic system.

Author: Agentic Pipeline Team
Date: November 2025
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from ..state.conversation_state import ConversationState, Message
from ..llm.base import BaseLLM


class ReportType(Enum):
    """Types of reports that can be generated."""
    COMPREHENSIVE = "comprehensive"
    SUMMARY = "summary"
    DETAILED = "detailed"
    MINIMAL = "minimal"


class ConfidenceLevel(Enum):
    """Confidence levels for reporting."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class ReportSection:
    """Individual section of a report."""
    title: str
    content: str
    confidence: ConfidenceLevel
    source_evidence: List[str]
    metadata: Dict[str, Any]


@dataclass
class ReportResult:
    """Complete report generation result."""
    final_answer: str
    sections: List[ReportSection]
    overall_confidence: ConfidenceLevel
    reasoning_process: List[str]
    sources_used: List[str]
    generation_time: float
    word_count: int
    error_message: Optional[str] = None


class ReporterNode:
    """
    Reporter node that synthesizes all gathered information into final answers.
    
    This node takes the complete conversation state (plan, evidence, verdict) and
    generates a comprehensive, well-structured final answer using LLM-based
    summarization with transparency about the reasoning process.
    
    Features:
    - LLM-based answer synthesis from complete state
    - Integration of plan, evidence, and verdict
    - Human-readable and well-structured output
    - Support for both Ollama and Mock LLM backends
    - Graceful handling of incomplete or low-confidence evidence
    - Transparency about reasoning process and confidence levels
    - Deterministic output in test mode
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        default_report_type: ReportType = ReportType.COMPREHENSIVE,
        min_confidence_threshold: float = 0.3,
        max_answer_length: int = 2000,
        include_reasoning: bool = True,
        deterministic_mode: bool = False
    ):
        """
        Initialize the ReporterNode.
        
        Args:
            llm: LLM instance for answer synthesis
            default_report_type: Default type of report to generate
            min_confidence_threshold: Minimum confidence for including evidence
            max_answer_length: Maximum length of final answer
            include_reasoning: Whether to include reasoning process
            deterministic_mode: Whether to use deterministic output for testing
        """
        self.llm = llm
        self.default_report_type = default_report_type
        self.min_confidence_threshold = min_confidence_threshold
        self.max_answer_length = max_answer_length
        self.include_reasoning = include_reasoning
        self.deterministic_mode = deterministic_mode
        self.logger = logging.getLogger(__name__)
        
        # Report templates - will be populated in subsequent implementations
        self.templates: Dict[ReportType, str] = {}
        
        # Mock responses for deterministic testing
        self.mock_responses: Dict[str, str] = {}
        
        self.logger.info(
            f"ReporterNode initialized: report_type={default_report_type.value}, "
            f"confidence_threshold={min_confidence_threshold}, "
            f"deterministic_mode={deterministic_mode}"
        )
    
    async def execute(self, state: ConversationState) -> ConversationState:
        """
        Execute the reporter node on the given conversation state.
        
        This method synthesizes all information from the conversation state
        (plan, evidence, verdict) and generates a comprehensive final answer.
        
        Args:
            state: Complete conversation state with plan, evidence, and verdict
            
        Returns:
            Updated conversation state with final answer
            
        Raises:
            ValueError: If state is invalid or missing required data
            RuntimeError: If report generation fails
        """
        self.logger.info("Starting ReporterNode execution")
        
        try:
            # Validate input state
            self._validate_input_state(state)
            
            # Extract information components
            plan_info = self._extract_plan_information(state)
            evidence_info = self._extract_evidence_information(state)
            verdict_info = self._extract_verdict_information(state)
            
            # Generate report sections
            report_sections = await self._generate_report_sections(
                plan_info, evidence_info, verdict_info
            )
            
            # Synthesize final answer
            report_result = await self._synthesize_final_answer(
                state, plan_info, evidence_info, verdict_info, report_sections
            )
            
            # Update state with final answer
            updated_state = self._update_state_with_report(state, report_result)
            
            # Add execution metadata
            updated_state.add_metadata("reporter_execution", {
                "report_type": self.default_report_type.value,
                "sections_generated": len(report_sections),
                "overall_confidence": report_result.overall_confidence.value,
                "word_count": report_result.word_count,
                "generation_time": report_result.generation_time,
                "sources_used": len(report_result.sources_used),
                "execution_timestamp": self._get_timestamp()
            })
            
            self.logger.info(
                f"ReporterNode execution completed successfully. "
                f"Generated {report_result.word_count} word answer with "
                f"{report_result.overall_confidence.value} confidence"
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"ReporterNode execution failed: {str(e)}")
            
            # Generate fallback answer for critical failures
            fallback_answer = self._generate_fallback_answer(state, str(e))
            
            # Add error information to state
            error_message = Message(
                role="assistant",
                content=fallback_answer,
                metadata={"error_type": type(e).__name__, "node": "reporter", "is_fallback": True}
            )
            state.messages.append(error_message)
            
            # Re-raise for upstream handling
            raise RuntimeError(f"ReporterNode execution failed: {str(e)}") from e
    
    def _validate_input_state(self, state: ConversationState) -> None:
        """
        Validate that the input state contains necessary information.
        
        Args:
            state: Conversation state to validate
            
        Raises:
            ValueError: If state is missing required components
        """
        if not state.messages:
            raise ValueError("ConversationState must contain messages")
        
        # Check for minimum required information
        has_plan = bool(getattr(state, 'plan', None))
        has_evidence = bool(getattr(state, 'evidence', None))
        has_verdict = bool(getattr(state, 'verdict', None))
        
        if not (has_plan or has_evidence):
            self.logger.warning(
                "State contains minimal information. Report quality may be reduced."
            )
        
        self.logger.info(
            f"State validation: plan={has_plan}, evidence={has_evidence}, verdict={has_verdict}"
        )
    
    def _extract_plan_information(self, state: ConversationState) -> Dict[str, Any]:
        """
        Extract plan information from conversation state.
        
        Args:
            state: Conversation state
            
        Returns:
            Dictionary with plan information
        """
        plan_info = {
            "has_plan": False,
            "plan_content": "",
            "plan_steps": [],
            "plan_confidence": ConfidenceLevel.UNKNOWN
        }
        
        if hasattr(state, 'plan') and state.plan:
            plan_info["has_plan"] = True
            plan_info["plan_content"] = state.plan
            
            # Extract plan steps (basic parsing)
            steps = []
            for line in state.plan.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*') or 
                           line.startswith('1.') or line.startswith('Step')):
                    steps.append(line)
            
            plan_info["plan_steps"] = steps
            plan_info["plan_confidence"] = ConfidenceLevel.MEDIUM  # Default
        
        self.logger.info(f"Extracted plan information: {len(plan_info['plan_steps'])} steps")
        return plan_info
    
    def _extract_evidence_information(self, state: ConversationState) -> Dict[str, Any]:
        """
        Extract evidence information from conversation state.
        
        Args:
            state: Conversation state
            
        Returns:
            Dictionary with evidence information
        """
        evidence_info = {
            "has_evidence": False,
            "evidence_count": 0,
            "evidence_items": [],
            "tools_used": [],
            "evidence_confidence": ConfidenceLevel.UNKNOWN
        }
        
        if hasattr(state, 'evidence') and state.evidence:
            evidence_info["has_evidence"] = True
            evidence_info["evidence_count"] = len(state.evidence)
            evidence_info["evidence_items"] = state.evidence
            
            # Extract tools used
            tools_used = set()
            for evidence in state.evidence:
                if hasattr(evidence, 'tool_used'):
                    tools_used.add(evidence.tool_used.value)
            
            evidence_info["tools_used"] = list(tools_used)
            
            # Determine evidence confidence based on quantity and variety
            if len(state.evidence) >= 3 and len(tools_used) >= 2:
                evidence_info["evidence_confidence"] = ConfidenceLevel.HIGH
            elif len(state.evidence) >= 1:
                evidence_info["evidence_confidence"] = ConfidenceLevel.MEDIUM
            else:
                evidence_info["evidence_confidence"] = ConfidenceLevel.LOW
        
        self.logger.info(
            f"Extracted evidence information: {evidence_info['evidence_count']} items, "
            f"tools: {evidence_info['tools_used']}"
        )
        return evidence_info
    
    def _extract_verdict_information(self, state: ConversationState) -> Dict[str, Any]:
        """
        Extract verdict information from conversation state.
        
        Args:
            state: Conversation state
            
        Returns:
            Dictionary with verdict information
        """
        verdict_info = {
            "has_verdict": False,
            "verdict_content": "",
            "confidence_score": 0.0,
            "verdict_confidence": ConfidenceLevel.UNKNOWN,
            "reasoning": []
        }
        
        if hasattr(state, 'verdict') and state.verdict:
            verdict_info["has_verdict"] = True
            verdict_info["verdict_content"] = str(state.verdict)
            
            # Extract confidence score if available
            if hasattr(state, 'confidence_score'):
                verdict_info["confidence_score"] = state.confidence_score
                
                # Map confidence score to level
                if state.confidence_score >= 0.8:
                    verdict_info["verdict_confidence"] = ConfidenceLevel.HIGH
                elif state.confidence_score >= 0.5:
                    verdict_info["verdict_confidence"] = ConfidenceLevel.MEDIUM
                else:
                    verdict_info["verdict_confidence"] = ConfidenceLevel.LOW
        
        self.logger.info(
            f"Extracted verdict information: has_verdict={verdict_info['has_verdict']}, "
            f"confidence={verdict_info['confidence_score']}"
        )
        return verdict_info
    
    async def _generate_report_sections(
        self,
        plan_info: Dict[str, Any],
        evidence_info: Dict[str, Any],
        verdict_info: Dict[str, Any]
    ) -> List[ReportSection]:
        """
        Generate individual report sections from extracted information.
        
        Args:
            plan_info: Extracted plan information
            evidence_info: Extracted evidence information
            verdict_info: Extracted verdict information
            
        Returns:
            List of report sections
        """
        sections = []
        
        # Plan section
        if plan_info["has_plan"]:
            sections.append(ReportSection(
                title="Approach",
                content=f"The analysis followed this plan:\n{plan_info['plan_content']}",
                confidence=plan_info["plan_confidence"],
                source_evidence=["plan"],
                metadata={"section_type": "plan", "steps_count": len(plan_info["plan_steps"])}
            ))
        
        # Evidence section
        if evidence_info["has_evidence"]:
            evidence_summary = f"Gathered {evidence_info['evidence_count']} pieces of evidence using: {', '.join(evidence_info['tools_used'])}"
            sections.append(ReportSection(
                title="Evidence",
                content=evidence_summary,
                confidence=evidence_info["evidence_confidence"],
                source_evidence=["evidence"],
                metadata={"section_type": "evidence", "tools_used": evidence_info["tools_used"]}
            ))
        
        # Verdict section
        if verdict_info["has_verdict"]:
            sections.append(ReportSection(
                title="Assessment",
                content=f"Analysis verdict: {verdict_info['verdict_content']}",
                confidence=verdict_info["verdict_confidence"],
                source_evidence=["verdict"],
                metadata={"section_type": "verdict", "confidence_score": verdict_info["confidence_score"]}
            ))
        
        self.logger.info(f"Generated {len(sections)} report sections")
        return sections
    
    async def _synthesize_final_answer(
        self,
        state: ConversationState,
        plan_info: Dict[str, Any],
        evidence_info: Dict[str, Any],
        verdict_info: Dict[str, Any],
        sections: List[ReportSection]
    ) -> ReportResult:
        """
        Synthesize final answer using LLM or deterministic mock.
        
        Args:
            state: Original conversation state
            plan_info: Plan information
            evidence_info: Evidence information
            verdict_info: Verdict information
            sections: Report sections
            
        Returns:
            Complete report result
        """
        import time
        start_time = time.time()
        
        try:
            if self.deterministic_mode:
                # Use deterministic mock for testing
                final_answer = self._generate_mock_answer(state, sections)
            else:
                # Use LLM for real synthesis
                final_answer = await self._generate_llm_answer(state, sections)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                plan_info, evidence_info, verdict_info
            )
            
            # Extract reasoning process
            reasoning_process = self._extract_reasoning_process(sections)
            
            # Collect sources
            sources_used = []
            for section in sections:
                sources_used.extend(section.source_evidence)
            
            generation_time = time.time() - start_time
            word_count = len(final_answer.split())
            
            return ReportResult(
                final_answer=final_answer,
                sections=sections,
                overall_confidence=overall_confidence,
                reasoning_process=reasoning_process,
                sources_used=list(set(sources_used)),
                generation_time=generation_time,
                word_count=word_count
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = f"Answer synthesis failed: {str(e)}"
            
            return ReportResult(
                final_answer=self._generate_fallback_answer(state, error_msg),
                sections=sections,
                overall_confidence=ConfidenceLevel.LOW,
                reasoning_process=["Error in synthesis process"],
                sources_used=["fallback"],
                generation_time=generation_time,
                word_count=0,
                error_message=error_msg
            )
    
    def _generate_mock_answer(self, state: ConversationState, sections: List[ReportSection]) -> str:
        """
        Generate deterministic mock answer for testing.
        
        Args:
            state: Conversation state
            sections: Report sections
            
        Returns:
            Mock answer string
        """
        # Generate deterministic answer based on state content
        answer_parts = []
        
        if sections:
            answer_parts.append("Based on the analysis:")
            for i, section in enumerate(sections, 1):
                answer_parts.append(f"{i}. {section.title}: {section.content[:100]}...")
        else:
            answer_parts.append("Analysis completed with limited information.")
        
        if hasattr(state, 'messages') and state.messages:
            original_query = state.messages[0].content if state.messages else "the query"
            answer_parts.append(f"\nIn response to {original_query[:50]}...")
        
        answer_parts.append("\nThis is a deterministic mock response for testing purposes.")
        
        return "\n".join(answer_parts)
    
    async def _generate_llm_answer(self, state: ConversationState, sections: List[ReportSection]) -> str:
        """
        Generate LLM-based answer synthesis (placeholder).
        
        Args:
            state: Conversation state
            sections: Report sections
            
        Returns:
            LLM-generated answer
        """
        # Placeholder for LLM integration - will be implemented in next task
        self.logger.info("Generating LLM-based answer synthesis")
        
        # Basic prompt construction
        prompt_parts = ["Please synthesize a comprehensive answer based on:"]
        
        for section in sections:
            prompt_parts.append(f"- {section.title}: {section.content}")
        
        prompt = "\n".join(prompt_parts)
        
        # Use LLM to generate response
        try:
            response = await self.llm.generate(prompt)
            return response.strip()
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            return f"LLM synthesis failed. Fallback summary: {len(sections)} sections analyzed."
    
    def _calculate_overall_confidence(
        self,
        plan_info: Dict[str, Any],
        evidence_info: Dict[str, Any],
        verdict_info: Dict[str, Any]
    ) -> ConfidenceLevel:
        """
        Calculate overall confidence level from all components.
        
        Args:
            plan_info: Plan information
            evidence_info: Evidence information
            verdict_info: Verdict information
            
        Returns:
            Overall confidence level
        """
        confidence_scores = []
        
        # Map confidence levels to numeric scores
        level_scores = {
            ConfidenceLevel.HIGH: 0.9,
            ConfidenceLevel.MEDIUM: 0.6,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.UNKNOWN: 0.1
        }
        
        # Add component confidence scores
        if plan_info["has_plan"]:
            confidence_scores.append(level_scores[plan_info["plan_confidence"]])
        
        if evidence_info["has_evidence"]:
            confidence_scores.append(level_scores[evidence_info["evidence_confidence"]])
        
        if verdict_info["has_verdict"]:
            confidence_scores.append(level_scores[verdict_info["verdict_confidence"]])
        
        # Calculate average confidence
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            if avg_confidence >= 0.8:
                return ConfidenceLevel.HIGH
            elif avg_confidence >= 0.5:
                return ConfidenceLevel.MEDIUM
            else:
                return ConfidenceLevel.LOW
        
        return ConfidenceLevel.UNKNOWN
    
    def _extract_reasoning_process(self, sections: List[ReportSection]) -> List[str]:
        """
        Extract reasoning process from report sections.
        
        Args:
            sections: Report sections
            
        Returns:
            List of reasoning steps
        """
        reasoning = []
        
        for section in sections:
            reasoning.append(f"{section.title}: {section.confidence.value} confidence")
        
        if not reasoning:
            reasoning.append("Limited reasoning information available")
        
        return reasoning
    
    def _generate_fallback_answer(self, state: ConversationState, error_msg: str) -> str:
        """
        Generate fallback answer when synthesis fails.
        
        Args:
            state: Conversation state
            error_msg: Error message
            
        Returns:
            Fallback answer string
        """
        fallback_parts = [
            "I encountered an issue while generating the complete response.",
            f"Error: {error_msg}",
        ]
        
        # Try to provide basic information from state
        if hasattr(state, 'plan') and state.plan:
            fallback_parts.append(f"Plan was available: {state.plan[:100]}...")
        
        if hasattr(state, 'evidence') and state.evidence:
            fallback_parts.append(f"Evidence was gathered: {len(state.evidence)} items")
        
        if hasattr(state, 'verdict') and state.verdict:
            fallback_parts.append(f"Verdict was available: {str(state.verdict)[:100]}...")
        
        fallback_parts.append("Please try again or contact support for assistance.")
        
        return "\n".join(fallback_parts)
    
    def _update_state_with_report(
        self,
        state: ConversationState,
        report_result: ReportResult
    ) -> ConversationState:
        """
        Update conversation state with generated report.
        
        Args:
            state: Original conversation state
            report_result: Generated report result
            
        Returns:
            Updated conversation state
        """
        # Add final answer as assistant message
        final_message = Message(
            role="assistant",
            content=report_result.final_answer,
            metadata={
                "node": "reporter",
                "confidence": report_result.overall_confidence.value,
                "word_count": report_result.word_count,
                "sources": report_result.sources_used,
                "generation_time": report_result.generation_time
            }
        )
        state.messages.append(final_message)
        
        # Store report result in state
        state.final_report = report_result
        
        return state
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()
    
    def set_deterministic_mode(self, enabled: bool) -> None:
        """
        Enable or disable deterministic mode for testing.
        
        Args:
            enabled: Whether to enable deterministic mode
        """
        self.deterministic_mode = enabled
        self.logger.info(f"Deterministic mode {'enabled' if enabled else 'disabled'}")
    
    def add_mock_response(self, key: str, response: str) -> None:
        """
        Add mock response for deterministic testing.
        
        Args:
            key: Response key
            response: Mock response content
        """
        self.mock_responses[key] = response
        self.logger.info(f"Added mock response for key: {key}")
    
    def get_report_info(self) -> Dict[str, Any]:
        """
        Get reporter configuration information.
        
        Returns:
            Reporter configuration details
        """
        return {
            "node_type": "reporter",
            "version": "1.0.0",
            "features": [
                "llm_synthesis",
                "multi_backend_support",
                "confidence_integration",
                "reasoning_transparency",
                "deterministic_testing",
                "fallback_handling"
            ],
            "configuration": {
                "default_report_type": self.default_report_type.value,
                "min_confidence_threshold": self.min_confidence_threshold,
                "max_answer_length": self.max_answer_length,
                "include_reasoning": self.include_reasoning,
                "deterministic_mode": self.deterministic_mode
            },
            "supported_report_types": [rt.value for rt in ReportType],
            "confidence_levels": [cl.value for cl in ConfidenceLevel]
        }