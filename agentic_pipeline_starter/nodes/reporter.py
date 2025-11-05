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
from .report_templates import ReportPromptTemplates, ReportFormatter


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
        
        # Initialize report templates
        self.templates = {
            ReportType.COMPREHENSIVE: ReportPromptTemplates.get_comprehensive_template(),
            ReportType.SUMMARY: ReportPromptTemplates.get_summary_template(),
            ReportType.DETAILED: ReportPromptTemplates.get_detailed_template(),
            ReportType.MINIMAL: ReportPromptTemplates.get_minimal_template()
        }
        
        # Error recovery template
        self.error_template = ReportPromptTemplates.get_error_recovery_template()
        
        # Report formatter
        self.formatter = ReportFormatter()
        
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
            
            # Extract confidence score from multiple sources
            confidence_score = 0.0
            
            # Check state-level confidence
            if hasattr(state, 'confidence_score'):
                confidence_score = state.confidence_score
            
            # Check verdict object confidence (if verdict is an object)
            elif hasattr(state.verdict, 'confidence'):
                confidence_score = state.verdict.confidence
            elif hasattr(state.verdict, 'confidence_score'):
                confidence_score = state.verdict.confidence_score
            
            # Check metadata for confidence information
            elif hasattr(state, 'metadata') and state.metadata:
                confidence_score = state.metadata.get('confidence_score', 0.0)
                if not confidence_score:
                    # Look for judge-related confidence
                    judge_meta = state.metadata.get('judge_execution', {})
                    confidence_score = judge_meta.get('confidence_score', 0.0)
            
            verdict_info["confidence_score"] = confidence_score
            
            # Map confidence score to level with enhanced thresholds
            if confidence_score >= 0.85:
                verdict_info["verdict_confidence"] = ConfidenceLevel.HIGH
            elif confidence_score >= 0.6:
                verdict_info["verdict_confidence"] = ConfidenceLevel.MEDIUM
            elif confidence_score >= 0.3:
                verdict_info["verdict_confidence"] = ConfidenceLevel.LOW
            else:
                verdict_info["verdict_confidence"] = ConfidenceLevel.UNKNOWN
            
            # Extract reasoning from verdict if available
            reasoning = []
            if hasattr(state.verdict, 'reasoning'):
                reasoning = state.verdict.reasoning
            elif hasattr(state.verdict, 'explanation'):
                reasoning = [state.verdict.explanation]
            elif hasattr(state, 'metadata') and state.metadata:
                judge_meta = state.metadata.get('judge_execution', {})
                if 'reasoning' in judge_meta:
                    reasoning = judge_meta['reasoning']
            
            verdict_info["reasoning"] = reasoning
        
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
        self.logger.info("Generating deterministic mock answer for testing")
        
        # Extract information for consistent mock generation
        plan_info = self._extract_plan_information(state)
        evidence_info = self._extract_evidence_information(state)
        verdict_info = self._extract_verdict_information(state)
        
        # Format using the same formatter
        original_query = self.formatter.extract_original_query(state)
        plan_summary = self.formatter.format_plan_summary(plan_info)
        evidence_summary = self.formatter.format_evidence_summary(evidence_info)
        verdict_summary = self.formatter.format_verdict_summary(verdict_info)
        
        # Generate deterministic answer based on report type
        if self.default_report_type == ReportType.COMPREHENSIVE:
            mock_answer = f"""## Summary
Mock comprehensive analysis for: {original_query[:100]}

## Analysis
{plan_summary}

## Evidence
{evidence_summary}

## Assessment
{verdict_summary}

## Conclusion
This is a deterministic mock response for testing purposes. The analysis included {len(sections)} report sections with structured information synthesis."""

        elif self.default_report_type == ReportType.SUMMARY:
            mock_answer = f"""Mock summary: {original_query[:100]}

Analysis completed with {evidence_info.get('evidence_count', 0)} evidence items and {len(plan_info.get('plan_steps', []))} plan steps.

{verdict_summary}

[Deterministic test response]"""

        elif self.default_report_type == ReportType.DETAILED:
            mock_answer = f"""# Technical Report: Mock Analysis

## Executive Summary
{original_query[:150]}

## Methodology
{plan_summary}

## Findings
{evidence_summary}
Tools: {', '.join(evidence_info.get('tools_used', []))}

## Assessment
{verdict_summary}

## Conclusions
Mock detailed technical analysis completed with {len(sections)} sections.

[Deterministic test response]"""

        else:  # MINIMAL
            mock_answer = f"Mock answer: {original_query[:50]}. {evidence_info.get('evidence_count', 0)} evidence items analyzed. [Test response]"
        
        # Apply length validation
        mock_answer = self.formatter.validate_response_length(mock_answer, self.max_answer_length)
        
        # Add confidence indicator
        if self.include_reasoning:
            overall_confidence = self._calculate_overall_confidence(
                plan_info, evidence_info, verdict_info
            )
            mock_answer = self.formatter.add_confidence_indicator(mock_answer, overall_confidence)
        
        return mock_answer
    
    async def _generate_llm_answer(self, state: ConversationState, sections: List[ReportSection]) -> str:
        """
        Generate LLM-based answer synthesis using proper templates.
        
        Args:
            state: Conversation state
            sections: Report sections
            
        Returns:
            LLM-generated answer
        """
        self.logger.info(f"Generating LLM-based answer using {self.default_report_type.value} template")
        
        try:
            # Extract information for template
            plan_info = self._extract_plan_information(state)
            evidence_info = self._extract_evidence_information(state)
            verdict_info = self._extract_verdict_information(state)
            
            # Format information using templates
            original_query = self.formatter.extract_original_query(state)
            plan_summary = self.formatter.format_plan_summary(plan_info)
            evidence_summary = self.formatter.format_evidence_summary(evidence_info)
            verdict_summary = self.formatter.format_verdict_summary(verdict_info)
            
            # Get appropriate template
            template = self.templates.get(self.default_report_type)
            if not template:
                raise ValueError(f"No template found for report type: {self.default_report_type.value}")
            
            # Prepare template variables
            template_vars = {
                "original_query": original_query,
                "plan_summary": plan_summary,
                "evidence_summary": evidence_summary,
                "verdict_summary": verdict_summary,
                "max_length": self.max_answer_length
            }
            
            # Add additional variables for detailed template
            if self.default_report_type == ReportType.DETAILED:
                template_vars.update({
                    "tools_used": self.formatter.format_tools_used(evidence_info),
                    "confidence_score": verdict_info.get("confidence_score", 0.0)
                })
            
            # Format prompt using template
            prompt = template.format(**template_vars)
            
            self.logger.debug(f"Generated prompt length: {len(prompt)} characters")
            
            # Use LLM to generate response
            response = await self.llm.generate(prompt)
            
            if not response:
                raise ValueError("LLM returned empty response")
            
            # Validate and format response
            formatted_response = self.formatter.validate_response_length(
                response.strip(), self.max_answer_length
            )
            
            # Add confidence indicator if requested
            if self.include_reasoning:
                overall_confidence = self._calculate_overall_confidence(
                    plan_info, evidence_info, verdict_info
                )
                formatted_response = self.formatter.add_confidence_indicator(
                    formatted_response, overall_confidence
                )
            
            self.logger.info(f"Successfully generated LLM response: {len(formatted_response.split())} words")
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            
            # Try error recovery template
            try:
                return await self._generate_error_recovery_answer(state, str(e))
            except Exception as recovery_error:
                self.logger.error(f"Error recovery also failed: {str(recovery_error)}")
                return f"LLM synthesis failed: {str(e)}. Fallback summary: {len(sections)} sections analyzed."
    
    async def _generate_error_recovery_answer(self, state: ConversationState, error_msg: str) -> str:
        """
        Generate answer using error recovery template.
        
        Args:
            state: Conversation state
            error_msg: Error message
            
        Returns:
            Error recovery answer
        """
        self.logger.info("Generating error recovery answer")
        
        # Extract partial information
        plan_info = self._extract_plan_information(state)
        evidence_info = self._extract_evidence_information(state)
        verdict_info = self._extract_verdict_information(state)
        
        # Format for error recovery
        original_query = self.formatter.extract_original_query(state)
        partial_info = self.formatter.format_partial_info(plan_info, evidence_info, verdict_info)
        
        # Use error recovery template
        template_vars = {
            "original_query": original_query,
            "error_message": error_msg,
            "partial_info": partial_info,
            "max_length": min(self.max_answer_length, 500)  # Shorter for error recovery
        }
        
        prompt = self.error_template.format(**template_vars)
        
        # Generate recovery response
        response = await self.llm.generate(prompt)
        return response.strip() if response else f"Error recovery failed: {error_msg}"
    
    def _calculate_overall_confidence(
        self,
        plan_info: Dict[str, Any],
        evidence_info: Dict[str, Any],
        verdict_info: Dict[str, Any]
    ) -> ConfidenceLevel:
        """
        Calculate overall confidence level from all components with enhanced Judge integration.
        
        Args:
            plan_info: Plan information
            evidence_info: Evidence information
            verdict_info: Verdict information
            
        Returns:
            Overall confidence level
        """
        confidence_scores = []
        weights = []
        
        # Map confidence levels to numeric scores
        level_scores = {
            ConfidenceLevel.HIGH: 0.9,
            ConfidenceLevel.MEDIUM: 0.6,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.UNKNOWN: 0.1
        }
        
        # Add component confidence scores with weights
        
        # Plan confidence (weight: 0.2)
        if plan_info["has_plan"]:
            confidence_scores.append(level_scores[plan_info["plan_confidence"]])
            weights.append(0.2)
        
        # Evidence confidence (weight: 0.3)
        if evidence_info["has_evidence"]:
            evidence_score = level_scores[evidence_info["evidence_confidence"]]
            
            # Boost confidence based on evidence quantity and diversity
            evidence_count = evidence_info.get("evidence_count", 0)
            tools_count = len(evidence_info.get("tools_used", []))
            
            if evidence_count >= 5 and tools_count >= 2:
                evidence_score = min(1.0, evidence_score + 0.1)
            elif evidence_count >= 3:
                evidence_score = min(1.0, evidence_score + 0.05)
            
            confidence_scores.append(evidence_score)
            weights.append(0.3)
        
        # Verdict confidence (weight: 0.5 - highest weight)
        if verdict_info["has_verdict"]:
            # Use actual confidence score from Judge if available
            judge_confidence = verdict_info.get("confidence_score", 0.0)
            
            if judge_confidence > 0:
                # Use judge's actual confidence score
                confidence_scores.append(judge_confidence)
            else:
                # Fallback to level-based score
                confidence_scores.append(level_scores[verdict_info["verdict_confidence"]])
            
            weights.append(0.5)
            
            # Additional boost if reasoning is available
            if verdict_info.get("reasoning"):
                reasoning_count = len(verdict_info["reasoning"])
                if reasoning_count >= 3:
                    confidence_scores[-1] = min(1.0, confidence_scores[-1] + 0.05)
        
        # Calculate weighted average confidence
        if confidence_scores and weights:
            weighted_sum = sum(score * weight for score, weight in zip(confidence_scores, weights))
            total_weight = sum(weights)
            avg_confidence = weighted_sum / total_weight
            
            # Apply threshold filtering based on minimum confidence
            if avg_confidence < self.min_confidence_threshold:
                self.logger.warning(
                    f"Overall confidence {avg_confidence:.2f} below threshold {self.min_confidence_threshold}"
                )
                return ConfidenceLevel.LOW
            
            # Map to confidence levels with enhanced thresholds
            if avg_confidence >= 0.85:
                return ConfidenceLevel.HIGH
            elif avg_confidence >= 0.6:
                return ConfidenceLevel.MEDIUM
            elif avg_confidence >= 0.3:
                return ConfidenceLevel.LOW
            else:
                return ConfidenceLevel.UNKNOWN
        
        self.logger.warning("No confidence information available for calculation")
        return ConfidenceLevel.UNKNOWN
    
    def get_confidence_breakdown(
        self,
        plan_info: Dict[str, Any],
        evidence_info: Dict[str, Any],
        verdict_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get detailed breakdown of confidence calculation for transparency.
        
        Args:
            plan_info: Plan information
            evidence_info: Evidence information
            verdict_info: Verdict information
            
        Returns:
            Detailed confidence breakdown
        """
        breakdown = {
            "components": {},
            "overall": self._calculate_overall_confidence(plan_info, evidence_info, verdict_info).value,
            "factors": []
        }
        
        # Plan component
        if plan_info["has_plan"]:
            breakdown["components"]["plan"] = {
                "confidence": plan_info["plan_confidence"].value,
                "steps_count": len(plan_info.get("plan_steps", [])),
                "weight": 0.2
            }
            breakdown["factors"].append("Structured plan available")
        
        # Evidence component
        if evidence_info["has_evidence"]:
            breakdown["components"]["evidence"] = {
                "confidence": evidence_info["evidence_confidence"].value,
                "evidence_count": evidence_info.get("evidence_count", 0),
                "tools_used": evidence_info.get("tools_used", []),
                "weight": 0.3
            }
            breakdown["factors"].append(f"{evidence_info.get('evidence_count', 0)} evidence items")
        
        # Verdict component  
        if verdict_info["has_verdict"]:
            breakdown["components"]["verdict"] = {
                "confidence": verdict_info["verdict_confidence"].value,
                "confidence_score": verdict_info.get("confidence_score", 0.0),
                "has_reasoning": bool(verdict_info.get("reasoning")),
                "weight": 0.5
            }
            breakdown["factors"].append("Judge verdict available")
        
        # Threshold check
        breakdown["threshold"] = {
            "minimum": self.min_confidence_threshold,
            "meets_threshold": breakdown["overall"] != "low"
        }
        
        return breakdown
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