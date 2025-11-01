# AIP-004: Implement Toolbox node

**Epic:** AIP-EP1 - Core Infrastructure  
**Story Points:** 5  
**Priority:** High  
**Assignee:** Developer  
**Sprint:** Sprint 1  

## Summary
Implement the Toolbox node that executes plan steps using available tools (HTTP fetch, safe math) and gathers evidence for the agentic pipeline.

## Description
Create the execution component that takes planned steps and uses available tools to gather data, perform calculations, and collect evidence. This node serves as the action-taking component of the agentic system.

## Acceptance Criteria
- [ ] Toolbox node executes plan steps and updates evidence
- [ ] Integration with http_fetch and safe_math tools
- [ ] Tool execution is safe and sandboxed
- [ ] Results are properly structured and stored
- [ ] Error handling for tool failures and timeouts
- [ ] Retry logic for transient failures
- [ ] Async execution for concurrent tool usage

## Technical Tasks
1. Create nodes/toolbox.py with ToolboxNode class
2. Implement tools/http_fetch.py with retry logic
3. Implement tools/safe_math.py with secure evaluation
4. Create tool registry and execution framework
5. Add result validation and formatting
6. Implement error handling and retry strategies
7. Add comprehensive logging and metrics
8. Create async interface for parallel tool execution

## Technical Specifications
- HTTP Tool: Configurable timeout, retry logic, error handling
- Math Tool: Safe expression evaluation without eval()
- Input: ConversationState with plan
- Output: ConversationState with populated evidence
- Error Handling: Graceful failure with detailed error messages
- Concurrency: Async execution for improved performance

## Definition of Done
- [ ] Tools execute successfully for valid inputs
- [ ] Error scenarios are handled gracefully
- [ ] Retry logic works for transient failures
- [ ] Evidence is properly structured and stored
- [ ] Unit tests achieve ≥90% coverage
- [ ] Integration tests with real and mock tools pass
- [ ] Security review confirms safe tool execution
- [ ] Performance meets requirements under load

## Dependencies
- AIP-001 (repository structure)
- AIP-002 (ConversationState)
- AIP-003 (Planner for plan input)

## Branch Name
`feature/AIP-004-toolbox-node`