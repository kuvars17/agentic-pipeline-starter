# AIP-006: Implement Reporter node

**Epic:** AIP-EP1 - Core Infrastructure  
**Story Points:** 4  
**Priority:** High  
**Assignee:** Developer  
**Sprint:** Sprint 1  

## Summary
Implement the Reporter node that synthesizes all gathered information into a final, human-readable answer using LLM-based summarization.

## Description
Create the synthesis component that takes the complete conversation state (plan, evidence, verdict) and generates a comprehensive, well-structured final answer. This node serves as the communication component of the agentic system.

## Acceptance Criteria
- [ ] Reporter generates final answer from complete state
- [ ] Answer integrates plan, evidence, and verdict coherently
- [ ] Output is human-readable and well-structured
- [ ] Integration with both Ollama and Mock LLM backends
- [ ] Handles incomplete or low-confidence evidence gracefully
- [ ] Provides transparency about reasoning process
- [ ] Deterministic output in test mode

## Technical Tasks
1. Create nodes/reporter.py with ReporterNode class
2. Implement LLM-based answer synthesis
3. Create report templates and formatting logic
4. Design answer structure with reasoning transparency
5. Implement confidence integration from Judge verdict
6. Add error handling for incomplete state
7. Create comprehensive logging and metrics
8. Implement deterministic mock reporting for testing

## Technical Specifications
- Input: ConversationState with plan, evidence, and verdict
- Output: ConversationState with final answer
- Report Format: Structured, human-readable response
- Transparency: Include reasoning process and confidence levels
- Error Handling: Graceful handling of incomplete information
- LLM Integration: Support both Ollama and Mock backends

## Definition of Done
- [ ] Reporter generates coherent final answers
- [ ] Answers properly integrate all state information
- [ ] Output quality is consistent and professional
- [ ] Error scenarios produce meaningful responses
- [ ] Unit tests achieve ≥90% coverage
- [ ] Integration tests with complete pipeline pass
- [ ] Mock reporting is deterministic for testing
- [ ] Documentation includes output format examples

## Dependencies
- AIP-001 (repository structure)
- AIP-002 (ConversationState)
- AIP-003 (Planner for plan input)
- AIP-004 (Toolbox for evidence input)
- AIP-005 (Judge for verdict input)
- LLM abstraction layer

## Branch Name
`feature/AIP-006-reporter-node`