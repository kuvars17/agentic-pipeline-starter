"""Plan validation and structuring utilities for the Planner node.

This module provides comprehensive validation and structuring logic to ensure
generated plans are well-formed, actionable, and meet quality standards.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class PlanQuality(Enum):
    """Quality levels for plan validation."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class ValidationResult:
    """Result of plan validation with quality metrics.
    
    Attributes:
        is_valid: Whether the plan passes basic validation
        quality: Overall quality assessment
        score: Numerical quality score (0-100)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        structured_steps: Cleaned and structured plan steps
    """
    is_valid: bool
    quality: PlanQuality
    score: int
    issues: List[str]
    suggestions: List[str]
    structured_steps: List[str]


class PlanValidator:
    """Validator for plan quality and structure."""
    
    # Minimum requirements for a valid plan
    MIN_STEPS = 2
    MAX_STEPS = 10
    MIN_STEP_LENGTH = 10
    MAX_STEP_LENGTH = 200
    
    # Quality scoring weights
    SCORING_WEIGHTS = {
        'step_count': 20,
        'step_quality': 30,
        'logical_flow': 25,
        'actionability': 25
    }
    
    def __init__(self):
        """Initialize the plan validator."""
        self.action_verbs = {
            'search', 'find', 'identify', 'analyze', 'research', 'gather',
            'collect', 'examine', 'investigate', 'verify', 'confirm',
            'calculate', 'compute', 'determine', 'assess', 'evaluate',
            'compare', 'contrast', 'synthesize', 'compile', 'organize',
            'present', 'provide', 'deliver', 'explain', 'describe'
        }
        
        self.quality_indicators = {
            'positive': [
                'specific', 'clear', 'detailed', 'comprehensive', 'systematic',
                'reliable', 'accurate', 'verified', 'authoritative', 'credible'
            ],
            'negative': [
                'vague', 'unclear', 'general', 'non-specific', 'ambiguous'
            ]
        }
    
    def validate_plan(self, plan_steps: List[str], query: str = "") -> ValidationResult:
        """Validate a plan and provide quality assessment.
        
        Args:
            plan_steps: List of plan steps to validate
            query: Original query for context (optional)
            
        Returns:
            ValidationResult with comprehensive assessment
        """
        issues = []
        suggestions = []
        
        # Clean and structure the steps
        structured_steps = self._clean_steps(plan_steps)
        
        # Basic validation checks
        basic_valid = self._basic_validation(structured_steps, issues, suggestions)
        
        if not basic_valid:
            return ValidationResult(
                is_valid=False,
                quality=PlanQuality.INVALID,
                score=0,
                issues=issues,
                suggestions=suggestions,
                structured_steps=structured_steps
            )
        
        # Quality assessment
        quality_score = self._assess_quality(structured_steps, query, issues, suggestions)
        quality_level = self._score_to_quality(quality_score)
        
        return ValidationResult(
            is_valid=True,
            quality=quality_level,
            score=quality_score,
            issues=issues,
            suggestions=suggestions,
            structured_steps=structured_steps
        )
    
    def _clean_steps(self, plan_steps: List[str]) -> List[str]:
        """Clean and normalize plan steps.
        
        Args:
            plan_steps: Raw plan steps
            
        Returns:
            Cleaned and structured steps
        """
        cleaned_steps = []
        
        for step in plan_steps:
            if not step or not step.strip():
                continue
            
            # Clean the step text
            cleaned = step.strip()
            
            # Remove common prefixes and numbering
            cleaned = re.sub(r'^[\d\.\-\*\•\s]+', '', cleaned)
            cleaned = re.sub(r'^(step|stage|phase)[\s\d\.\-:]*', '', cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            
            # Skip if too short after cleaning
            if len(cleaned) < self.MIN_STEP_LENGTH:
                continue
            
            # Ensure proper sentence structure
            if cleaned and not cleaned[0].isupper():
                cleaned = cleaned[0].upper() + cleaned[1:]
            
            if cleaned and not cleaned.endswith('.'):
                cleaned += '.'
            
            cleaned_steps.append(cleaned)
        
        return cleaned_steps
    
    def _basic_validation(self, steps: List[str], issues: List[str], suggestions: List[str]) -> bool:
        """Perform basic validation checks.
        
        Args:
            steps: Plan steps to validate
            issues: List to append issues to
            suggestions: List to append suggestions to
            
        Returns:
            True if plan passes basic validation
        """
        is_valid = True
        
        # Check step count
        if len(steps) < self.MIN_STEPS:
            issues.append(f"Plan has only {len(steps)} steps, minimum {self.MIN_STEPS} required")
            suggestions.append("Add more detailed steps to make the plan more comprehensive")
            is_valid = False
        
        if len(steps) > self.MAX_STEPS:
            issues.append(f"Plan has {len(steps)} steps, maximum {self.MAX_STEPS} recommended")
            suggestions.append("Consider consolidating some steps for better focus")
        
        # Check individual step quality
        for i, step in enumerate(steps, 1):
            if len(step) < self.MIN_STEP_LENGTH:
                issues.append(f"Step {i} is too short ({len(step)} chars)")
                suggestions.append(f"Expand step {i} with more specific details")
                is_valid = False
            
            if len(step) > self.MAX_STEP_LENGTH:
                issues.append(f"Step {i} is too long ({len(step)} chars)")
                suggestions.append(f"Break down step {i} into smaller, more focused actions")
        
        return is_valid
    
    def _assess_quality(self, steps: List[str], query: str, issues: List[str], suggestions: List[str]) -> int:
        """Assess the overall quality of the plan.
        
        Args:
            steps: Plan steps to assess
            query: Original query for context
            issues: List to append issues to
            suggestions: List to append suggestions to
            
        Returns:
            Quality score (0-100)
        """
        scores = {}
        
        # Step count score (20%)
        step_count = len(steps)
        if 3 <= step_count <= 6:
            scores['step_count'] = 100
        elif step_count == 2 or step_count == 7:
            scores['step_count'] = 80
        elif step_count == 8:
            scores['step_count'] = 60
        else:
            scores['step_count'] = 40
        
        # Step quality score (30%)
        scores['step_quality'] = self._assess_step_quality(steps, issues, suggestions)
        
        # Logical flow score (25%)
        scores['logical_flow'] = self._assess_logical_flow(steps, issues, suggestions)
        
        # Actionability score (25%)
        scores['actionability'] = self._assess_actionability(steps, issues, suggestions)
        
        # Calculate weighted total
        total_score = sum(
            scores[component] * (self.SCORING_WEIGHTS[component] / 100)
            for component in scores
        )
        
        return int(total_score)
    
    def _assess_step_quality(self, steps: List[str], issues: List[str], suggestions: List[str]) -> int:
        """Assess the quality of individual steps.
        
        Args:
            steps: Plan steps to assess
            issues: List to append issues to
            suggestions: List to append suggestions to
            
        Returns:
            Step quality score (0-100)
        """
        if not steps:
            return 0
        
        total_score = 0
        
        for i, step in enumerate(steps, 1):
            step_score = 50  # Base score
            step_lower = step.lower()
            
            # Check for action verbs
            has_action_verb = any(verb in step_lower for verb in self.action_verbs)
            if has_action_verb:
                step_score += 20
            else:
                issues.append(f"Step {i} lacks clear action verb")
                suggestions.append(f"Start step {i} with an action verb like 'analyze', 'research', or 'identify'")
            
            # Check for specificity
            has_specificity = any(word in step_lower for word in ['specific', 'detailed', 'particular'])
            if has_specificity or len(step) > 50:
                step_score += 15
            
            # Check for quality indicators
            positive_indicators = sum(1 for word in self.quality_indicators['positive'] if word in step_lower)
            negative_indicators = sum(1 for word in self.quality_indicators['negative'] if word in step_lower)
            
            step_score += positive_indicators * 5
            step_score -= negative_indicators * 10
            
            # Ensure score bounds
            step_score = max(0, min(100, step_score))
            total_score += step_score
        
        return int(total_score / len(steps))
    
    def _assess_logical_flow(self, steps: List[str], issues: List[str], suggestions: List[str]) -> int:
        """Assess the logical flow and ordering of steps.
        
        Args:
            steps: Plan steps to assess
            issues: List to append issues to
            suggestions: List to append suggestions to
            
        Returns:
            Logical flow score (0-100)
        """
        if len(steps) < 2:
            return 50
        
        score = 70  # Base score
        
        # Check for logical progression keywords
        progression_patterns = [
            (r'\b(first|start|begin|initial)', 0, 2),  # Should be early
            (r'\b(then|next|after|following)', 1, -1),  # Should not be first
            (r'\b(finally|last|conclude|complete)', -2, -1),  # Should be late
        ]
        
        for pattern, min_pos, max_pos in progression_patterns:
            for i, step in enumerate(steps):
                if re.search(pattern, step.lower()):
                    actual_pos = i
                    total_steps = len(steps)
                    
                    if max_pos == -1:
                        max_pos = total_steps - 1
                    if min_pos < 0:
                        min_pos = total_steps + min_pos
                    
                    if min_pos <= actual_pos <= max_pos:
                        score += 5
                    else:
                        score -= 10
                        issues.append(f"Step {i+1} seems out of logical order")
        
        # Check for dependencies and prerequisites
        dependency_keywords = ['verify', 'confirm', 'validate', 'check']
        gather_keywords = ['collect', 'gather', 'research', 'find']
        
        last_gather_index = -1
        first_verify_index = len(steps)
        
        for i, step in enumerate(steps):
            step_lower = step.lower()
            if any(keyword in step_lower for keyword in gather_keywords):
                last_gather_index = i
            if any(keyword in step_lower for keyword in dependency_keywords):
                first_verify_index = min(first_verify_index, i)
        
        if last_gather_index > first_verify_index:
            issues.append("Information gathering appears after verification steps")
            suggestions.append("Reorganize steps to gather information before verification")
            score -= 15
        
        return max(0, min(100, score))
    
    def _assess_actionability(self, steps: List[str], issues: List[str], suggestions: List[str]) -> int:
        """Assess how actionable and executable the steps are.
        
        Args:
            steps: Plan steps to assess
            issues: List to append issues to
            suggestions: List to append suggestions to
            
        Returns:
            Actionability score (0-100)
        """
        if not steps:
            return 0
        
        total_score = 0
        
        for i, step in enumerate(steps, 1):
            step_score = 60  # Base score
            step_lower = step.lower()
            
            # Check for concrete actions
            if any(verb in step_lower for verb in self.action_verbs):
                step_score += 20
            
            # Check for vague language
            vague_words = ['somehow', 'try to', 'attempt', 'maybe', 'perhaps', 'possibly']
            if any(word in step_lower for word in vague_words):
                step_score -= 20
                issues.append(f"Step {i} contains vague language")
                suggestions.append(f"Make step {i} more specific and definitive")
            
            # Check for measurable outcomes
            measurable_words = ['identify', 'list', 'calculate', 'determine', 'find', 'locate']
            if any(word in step_lower for word in measurable_words):
                step_score += 10
            
            # Check for implementation details
            if any(detail in step_lower for detail in ['source', 'method', 'tool', 'website']):
                step_score += 10
            
            total_score += max(0, min(100, step_score))
        
        return int(total_score / len(steps))
    
    def _score_to_quality(self, score: int) -> PlanQuality:
        """Convert numerical score to quality level.
        
        Args:
            score: Numerical score (0-100)
            
        Returns:
            Corresponding quality level
        """
        if score >= 85:
            return PlanQuality.EXCELLENT
        elif score >= 70:
            return PlanQuality.GOOD
        elif score >= 55:
            return PlanQuality.ACCEPTABLE
        elif score >= 30:
            return PlanQuality.POOR
        else:
            return PlanQuality.INVALID


class PlanStructurer:
    """Utilities for structuring and formatting plans."""
    
    @staticmethod
    def format_steps(steps: List[str], format_style: str = "numbered") -> List[str]:
        """Format plan steps with consistent styling.
        
        Args:
            steps: Plan steps to format
            format_style: Formatting style ("numbered", "bulleted", "plain")
            
        Returns:
            Formatted plan steps
        """
        if format_style == "numbered":
            return [f"{i}. {step}" for i, step in enumerate(steps, 1)]
        elif format_style == "bulleted":
            return [f"• {step}" for step in steps]
        else:
            return steps
    
    @staticmethod
    def add_step_metadata(steps: List[str]) -> List[Dict[str, Any]]:
        """Add metadata to plan steps for enhanced processing.
        
        Args:
            steps: Plan steps to enhance
            
        Returns:
            List of step dictionaries with metadata
        """
        enhanced_steps = []
        
        for i, step in enumerate(steps):
            step_data = {
                'index': i + 1,
                'text': step,
                'length': len(step),
                'word_count': len(step.split()),
                'has_action_verb': PlanValidator()._has_action_verb(step),
                'estimated_time': PlanStructurer._estimate_step_time(step)
            }
            enhanced_steps.append(step_data)
        
        return enhanced_steps
    
    @staticmethod
    def _estimate_step_time(step: str) -> str:
        """Estimate time required for a step.
        
        Args:
            step: Plan step text
            
        Returns:
            Estimated time as string
        """
        step_lower = step.lower()
        
        if any(word in step_lower for word in ['research', 'investigate', 'analyze']):
            return "5-15 minutes"
        elif any(word in step_lower for word in ['calculate', 'compute', 'solve']):
            return "2-5 minutes"
        elif any(word in step_lower for word in ['search', 'find', 'identify']):
            return "3-8 minutes"
        else:
            return "2-10 minutes"