"""
Report Templates and Prompt Engineering for Reporter Node

This module contains prompt templates and formatting logic for generating
structured, human-readable reports with LLM-based synthesis.

Author: Agentic Pipeline Team
Date: November 2025
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

from ..nodes.reporter import ReportType, ConfidenceLevel, ReportSection


class PromptTemplate:
    """Base class for prompt templates."""
    
    def __init__(self, template: str, required_variables: List[str]):
        self.template = template
        self.required_variables = required_variables
    
    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        # Check required variables
        missing = [var for var in self.required_variables if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        return self.template.format(**kwargs)


class ReportPromptTemplates:
    """Collection of prompt templates for different report types."""
    
    @staticmethod
    def get_comprehensive_template() -> PromptTemplate:
        """Get comprehensive report template."""
        template = """You are an expert analyst tasked with synthesizing a comprehensive final answer.

CONTEXT:
- Original Query: {original_query}
- Analysis Plan: {plan_summary}
- Evidence Gathered: {evidence_summary}
- Assessment Result: {verdict_summary}

TASK:
Generate a well-structured, human-readable final answer that:
1. Directly addresses the original query
2. Integrates all available information coherently
3. Provides clear reasoning and evidence
4. Indicates confidence levels appropriately
5. Maintains professional tone and clarity

STRUCTURE YOUR RESPONSE AS:
## Summary
[Brief, direct answer to the query]

## Analysis
[Detailed explanation with reasoning]

## Evidence
[Key findings and supporting data]

## Confidence Assessment
[Overall confidence level with justification]

## Conclusion
[Final synthesis and any recommendations]

GUIDELINES:
- Be specific and factual
- Acknowledge limitations or uncertainty
- Use clear, accessible language
- Provide actionable insights when possible
- Maximum length: {max_length} words

Please generate the final answer:"""
        
        return PromptTemplate(
            template=template,
            required_variables=[
                "original_query", "plan_summary", "evidence_summary", 
                "verdict_summary", "max_length"
            ]
        )
    
    @staticmethod
    def get_summary_template() -> PromptTemplate:
        """Get summary report template."""
        template = """You are tasked with creating a concise summary answer.

QUERY: {original_query}

AVAILABLE INFORMATION:
- Plan: {plan_summary}
- Evidence: {evidence_summary}
- Verdict: {verdict_summary}

Create a clear, concise answer (max {max_length} words) that directly addresses the query using the available information. Include confidence level in your response.

Answer:"""
        
        return PromptTemplate(
            template=template,
            required_variables=[
                "original_query", "plan_summary", "evidence_summary",
                "verdict_summary", "max_length"
            ]
        )
    
    @staticmethod
    def get_detailed_template() -> PromptTemplate:
        """Get detailed report template."""
        template = """You are an expert analyst providing a detailed technical report.

ANALYSIS REQUEST: {original_query}

METHODOLOGY:
Plan Executed: {plan_summary}

FINDINGS:
Evidence Collected: {evidence_summary}
Tools Used: {tools_used}

ASSESSMENT:
Verdict: {verdict_summary}
Confidence Score: {confidence_score}

TASK:
Provide a detailed technical report that includes:
1. Executive summary
2. Methodology explanation
3. Detailed findings analysis
4. Risk assessment and limitations
5. Technical conclusions
6. Recommendations for next steps

Use professional technical language appropriate for expert audience.
Maximum length: {max_length} words.

Report:"""
        
        return PromptTemplate(
            template=template,
            required_variables=[
                "original_query", "plan_summary", "evidence_summary",
                "verdict_summary", "tools_used", "confidence_score", "max_length"
            ]
        )
    
    @staticmethod
    def get_minimal_template() -> PromptTemplate:
        """Get minimal report template."""
        template = """Provide a brief, direct answer to: {original_query}

Based on: {evidence_summary}

Answer (max {max_length} words):"""
        
        return PromptTemplate(
            template=template,
            required_variables=["original_query", "evidence_summary", "max_length"]
        )
    
    @staticmethod
    def get_error_recovery_template() -> PromptTemplate:
        """Get template for error recovery scenarios."""
        template = """The analysis encountered issues, but partial information is available.

ORIGINAL REQUEST: {original_query}
ERROR ENCOUNTERED: {error_message}

PARTIAL INFORMATION AVAILABLE:
{partial_info}

Please provide the best possible answer given the limitations, clearly indicating:
1. What analysis was attempted
2. What information is available
3. What limitations exist
4. Suggested next steps

Keep response under {max_length} words and be transparent about limitations.

Response:"""
        
        return PromptTemplate(
            template=template,
            required_variables=[
                "original_query", "error_message", "partial_info", "max_length"
            ]
        )


class ReportFormatter:
    """Utility class for formatting report components."""
    
    @staticmethod
    def format_plan_summary(plan_info: Dict[str, Any]) -> str:
        """Format plan information for templates."""
        if not plan_info.get("has_plan"):
            return "No structured plan was available."
        
        plan_content = plan_info.get("plan_content", "")
        steps = plan_info.get("plan_steps", [])
        
        if steps:
            formatted_steps = "\n".join(f"- {step}" for step in steps[:5])  # Limit to 5 steps
            return f"Analysis Plan:\n{formatted_steps}"
        else:
            return f"Plan: {plan_content[:200]}{'...' if len(plan_content) > 200 else ''}"
    
    @staticmethod
    def format_evidence_summary(evidence_info: Dict[str, Any]) -> str:
        """Format evidence information for templates."""
        if not evidence_info.get("has_evidence"):
            return "No evidence was gathered during analysis."
        
        count = evidence_info.get("evidence_count", 0)
        tools = evidence_info.get("tools_used", [])
        
        summary_parts = [f"Gathered {count} pieces of evidence"]
        
        if tools:
            summary_parts.append(f"using tools: {', '.join(tools)}")
        
        # Add brief description of evidence types
        evidence_items = evidence_info.get("evidence_items", [])
        if evidence_items:
            sources = set()
            for item in evidence_items[:3]:  # Show first 3 sources
                if hasattr(item, 'source'):
                    sources.add(item.source)
            
            if sources:
                summary_parts.append(f"from sources: {', '.join(list(sources)[:3])}")
        
        return ". ".join(summary_parts) + "."
    
    @staticmethod
    def format_verdict_summary(verdict_info: Dict[str, Any]) -> str:
        """Format verdict information for templates."""
        if not verdict_info.get("has_verdict"):
            return "No formal verdict was generated."
        
        verdict_content = verdict_info.get("verdict_content", "")
        confidence_score = verdict_info.get("confidence_score", 0.0)
        confidence_level = verdict_info.get("verdict_confidence", ConfidenceLevel.UNKNOWN)
        
        summary_parts = []
        
        # Add verdict content (truncated)
        if verdict_content:
            truncated = verdict_content[:150] + ("..." if len(verdict_content) > 150 else "")
            summary_parts.append(f"Verdict: {truncated}")
        
        # Add confidence information
        if confidence_score > 0:
            summary_parts.append(f"Confidence: {confidence_level.value} ({confidence_score:.2f})")
        else:
            summary_parts.append(f"Confidence: {confidence_level.value}")
        
        return ". ".join(summary_parts) + "."
    
    @staticmethod
    def extract_original_query(state) -> str:
        """Extract original query from conversation state."""
        if hasattr(state, 'messages') and state.messages:
            # Find the first user message
            for message in state.messages:
                if message.role == "user":
                    return message.content
            
            # Fallback to first message
            return state.messages[0].content
        
        return "No original query available"
    
    @staticmethod
    def format_tools_used(evidence_info: Dict[str, Any]) -> str:
        """Format tools used for detailed templates."""
        tools = evidence_info.get("tools_used", [])
        if not tools:
            return "No tools were used"
        
        return ", ".join(tools)
    
    @staticmethod
    def format_partial_info(
        plan_info: Dict[str, Any],
        evidence_info: Dict[str, Any],
        verdict_info: Dict[str, Any]
    ) -> str:
        """Format partial information for error recovery."""
        info_parts = []
        
        if plan_info.get("has_plan"):
            info_parts.append(f"Plan: Available ({len(plan_info.get('plan_steps', []))} steps)")
        
        if evidence_info.get("has_evidence"):
            count = evidence_info.get("evidence_count", 0)
            tools = evidence_info.get("tools_used", [])
            info_parts.append(f"Evidence: {count} items from {len(tools)} tools")
        
        if verdict_info.get("has_verdict"):
            confidence = verdict_info.get("verdict_confidence", ConfidenceLevel.UNKNOWN)
            info_parts.append(f"Verdict: Available with {confidence.value} confidence")
        
        if not info_parts:
            return "No structured information is available"
        
        return "\n".join(f"- {part}" for part in info_parts)
    
    @staticmethod
    def validate_response_length(response: str, max_length: int) -> str:
        """Validate and truncate response if necessary."""
        words = response.split()
        
        if len(words) <= max_length:
            return response
        
        # Truncate and add indication
        truncated_words = words[:max_length-10]  # Reserve space for truncation notice
        truncated_response = " ".join(truncated_words)
        truncated_response += f"... [Response truncated to {max_length} words]"
        
        return truncated_response
    
    @staticmethod
    def add_confidence_indicator(response: str, confidence: ConfidenceLevel) -> str:
        """Add confidence indicator to response."""
        confidence_indicators = {
            ConfidenceLevel.HIGH: "🟢 High Confidence",
            ConfidenceLevel.MEDIUM: "🟡 Medium Confidence", 
            ConfidenceLevel.LOW: "🔴 Low Confidence",
            ConfidenceLevel.UNKNOWN: "⚪ Confidence Unknown"
        }
        
        indicator = confidence_indicators.get(confidence, "⚪ Confidence Unknown")
        
        # Add confidence indicator at the end
        return f"{response}\n\n---\n*{indicator}*"