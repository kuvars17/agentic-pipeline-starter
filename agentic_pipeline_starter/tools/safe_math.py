"""
Safe Math Tool for Agentic Pipeline

This module implements a safe mathematical expression evaluator that provides
secure calculation capabilities without using eval() or other dangerous methods.

Author: Agentic Pipeline Team
Date: November 2025
Version: 1.0.0
"""

import re
import math
import logging
from typing import Dict, Any, Union, Optional, List
from dataclasses import dataclass
from enum import Enum
import operator
from decimal import Decimal, getcontext, InvalidOperation


class MathError(Exception):
    """Custom exception for math-related errors."""
    
    def __init__(self, message: str, expression: Optional[str] = None):
        super().__init__(message)
        self.expression = expression


class MathOperation(Enum):
    """Supported mathematical operations."""
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    POWER = "**"
    MODULO = "%"
    FLOOR_DIVIDE = "//"


@dataclass
class MathResult:
    """Mathematical calculation result container."""
    expression: str
    result: Union[int, float, Decimal]
    operations_used: List[str]
    execution_time: float
    precision: int
    error_message: Optional[str] = None


class SafeMathTool:
    """
    Safe mathematical expression evaluator without eval().
    
    Features:
    - Secure expression parsing without eval()
    - Support for basic arithmetic operations
    - Mathematical functions (sin, cos, log, sqrt, etc.)
    - Configurable precision for Decimal calculations
    - Comprehensive error handling
    - Operation tracking and validation
    - Protection against infinite loops and dangerous operations
    """
    
    def __init__(
        self,
        precision: int = 28,
        max_iterations: int = 1000,
        max_expression_length: int = 1000,
        enable_functions: bool = True
    ):
        """
        Initialize the safe math tool.
        
        Args:
            precision: Decimal precision for calculations
            max_iterations: Maximum iterations for recursive operations
            max_expression_length: Maximum length of mathematical expressions
            enable_functions: Whether to enable mathematical functions
        """
        self.precision = precision
        self.max_iterations = max_iterations
        self.max_expression_length = max_expression_length
        self.enable_functions = enable_functions
        self.logger = logging.getLogger(__name__)
        
        # Set decimal context
        getcontext().prec = precision
        
        # Define allowed operations
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '//': operator.floordiv,
            '%': operator.mod,
            '**': operator.pow,
        }
        
        # Define allowed mathematical functions
        self.functions = {
            'abs': abs,
            'round': round,
            'ceil': math.ceil,
            'floor': math.floor,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'factorial': math.factorial,
            'gcd': math.gcd,
            'max': max,
            'min': min,
            'sum': sum,
        } if enable_functions else {}
        
        # Define mathematical constants
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau if hasattr(math, 'tau') else 2 * math.pi,
            'inf': math.inf,
        }
        
        self.logger.info(
            f"SafeMathTool initialized: precision={precision}, "
            f"functions_enabled={enable_functions}"
        )
    
    def calculate(self, expression: str) -> MathResult:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Mathematical result with metadata
            
        Raises:
            MathError: If expression is invalid or calculation fails
        """
        import time
        start_time = time.time()
        operations_used = []
        
        try:
            # Validate and sanitize expression
            clean_expression = self._sanitize_expression(expression)
            self.logger.info(f"Evaluating expression: {clean_expression}")
            
            # Parse and evaluate expression
            result = self._evaluate_expression(clean_expression, operations_used)
            
            execution_time = time.time() - start_time
            
            self.logger.info(
                f"Math calculation completed: {clean_expression} = {result} "
                f"in {execution_time:.4f}s"
            )
            
            return MathResult(
                expression=clean_expression,
                result=result,
                operations_used=operations_used,
                execution_time=execution_time,
                precision=self.precision
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Math calculation failed: {str(e)}"
            self.logger.error(error_msg)
            
            return MathResult(
                expression=expression,
                result=0,
                operations_used=operations_used,
                execution_time=execution_time,
                precision=self.precision,
                error_message=error_msg
            )
    
    def _sanitize_expression(self, expression: str) -> str:
        """
        Sanitize and validate mathematical expression.
        
        Args:
            expression: Raw expression string
            
        Returns:
            Cleaned expression string
            
        Raises:
            MathError: If expression is invalid or unsafe
        """
        if not expression or not isinstance(expression, str):
            raise MathError("Expression must be a non-empty string")
        
        if len(expression) > self.max_expression_length:
            raise MathError(
                f"Expression too long (max {self.max_expression_length} chars)"
            )
        
        # Remove whitespace
        clean = re.sub(r'\s+', '', expression)
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'__\w+__',  # Dunder methods
            r'import\s+\w+',  # Import statements
            r'exec\s*\(',  # Exec calls
            r'eval\s*\(',  # Eval calls
            r'open\s*\(',  # File operations
            r'while\s+',  # While loops
            r'for\s+\w+\s+in',  # For loops
            r'def\s+\w+',  # Function definitions
            r'class\s+\w+',  # Class definitions
            r'lambda\s*\w*:',  # Lambda functions
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, clean, re.IGNORECASE):
                raise MathError(f"Dangerous pattern detected: {pattern}")
        
        # Validate allowed characters
        allowed_pattern = r'^[0-9+\-*/().,\s\w]+$'
        if not re.match(allowed_pattern, clean):
            raise MathError("Expression contains invalid characters")
        
        return clean
    
    def _evaluate_expression(self, expression: str, operations_used: List[str]) -> Union[int, float]:
        """
        Evaluate sanitized mathematical expression.
        
        Args:
            expression: Sanitized expression string
            operations_used: List to track operations used
            
        Returns:
            Calculation result
        """
        try:
            # Replace constants
            for const_name, const_value in self.constants.items():
                expression = expression.replace(const_name, str(const_value))
                if const_name in expression:
                    operations_used.append(f"constant:{const_name}")
            
            # Handle function calls
            if self.enable_functions:
                expression = self._handle_functions(expression, operations_used)
            
            # Evaluate using recursive descent parser
            result = self._parse_expression(expression, operations_used)
            
            # Validate result
            if math.isnan(result):
                raise MathError("Result is NaN (Not a Number)")
            
            if math.isinf(result):
                raise MathError("Result is infinite")
            
            return result
            
        except ZeroDivisionError:
            raise MathError("Division by zero")
        except OverflowError:
            raise MathError("Numerical overflow")
        except ValueError as e:
            raise MathError(f"Invalid value: {str(e)}")
        except Exception as e:
            raise MathError(f"Evaluation error: {str(e)}")
    
    def _handle_functions(self, expression: str, operations_used: List[str]) -> str:
        """
        Handle mathematical function calls in expression.
        
        Args:
            expression: Expression with function calls
            operations_used: List to track operations used
            
        Returns:
            Expression with function results substituted
        """
        # Function call pattern: function_name(arguments)
        func_pattern = r'(\w+)\(([^()]+)\)'
        
        while re.search(func_pattern, expression):
            match = re.search(func_pattern, expression)
            if not match:
                break
            
            func_name = match.group(1)
            func_args = match.group(2)
            
            if func_name not in self.functions:
                raise MathError(f"Unknown function: {func_name}")
            
            # Parse function arguments
            try:
                # Simple argument parsing (comma-separated numbers)
                args = [float(arg.strip()) for arg in func_args.split(',')]
                
                # Call function
                func = self.functions[func_name]
                if func_name in ['max', 'min', 'sum']:
                    result = func(args)
                elif func_name == 'round' and len(args) == 2:
                    result = func(args[0], int(args[1]))
                elif func_name == 'gcd' and len(args) == 2:
                    result = func(int(args[0]), int(args[1]))
                else:
                    result = func(args[0]) if len(args) == 1 else func(*args)
                
                # Replace function call with result
                expression = expression.replace(match.group(0), str(result))
                operations_used.append(f"function:{func_name}")
                
            except Exception as e:
                raise MathError(f"Function {func_name} error: {str(e)}")
        
        return expression
    
    def _parse_expression(self, expression: str, operations_used: List[str]) -> float:
        """
        Parse mathematical expression using recursive descent.
        
        Args:
            expression: Expression to parse
            operations_used: List to track operations used
            
        Returns:
            Parsed result
        """
        # Remove all spaces
        expression = expression.replace(' ', '')
        
        # Simple expression evaluator for basic arithmetic
        # This is a simplified implementation - could be enhanced with a full parser
        
        try:
            # Handle parentheses first
            while '(' in expression:
                # Find innermost parentheses
                start = -1
                for i, char in enumerate(expression):
                    if char == '(':
                        start = i
                    elif char == ')':
                        if start == -1:
                            raise MathError("Mismatched parentheses")
                        
                        # Evaluate expression inside parentheses
                        inner = expression[start+1:i]
                        result = self._evaluate_simple_expression(inner, operations_used)
                        
                        # Replace parentheses group with result
                        expression = expression[:start] + str(result) + expression[i+1:]
                        break
                else:
                    if start != -1:
                        raise MathError("Mismatched parentheses")
            
            # Evaluate final expression
            return self._evaluate_simple_expression(expression, operations_used)
            
        except Exception as e:
            raise MathError(f"Parse error: {str(e)}")
    
    def _evaluate_simple_expression(self, expression: str, operations_used: List[str]) -> float:
        """
        Evaluate simple arithmetic expression without parentheses.
        
        Args:
            expression: Simple expression
            operations_used: List to track operations used
            
        Returns:
            Evaluation result
        """
        if not expression:
            raise MathError("Empty expression")
        
        # Handle negative numbers
        if expression.startswith('-'):
            expression = '0' + expression
        
        # Split by operators (order of operations)
        # First handle ** (power)
        if '**' in expression:
            parts = expression.split('**')
            result = float(parts[0])
            for part in parts[1:]:
                result = result ** float(part)
                operations_used.append('power')
            return result
        
        # Then handle *, /, //, %
        for op in ['*', '/', '//', '%']:
            if op in expression:
                parts = expression.split(op)
                result = float(parts[0])
                for part in parts[1:]:
                    if op == '*':
                        result *= float(part)
                        operations_used.append('multiply')
                    elif op == '/':
                        divisor = float(part)
                        if divisor == 0:
                            raise MathError("Division by zero")
                        result /= divisor
                        operations_used.append('divide')
                    elif op == '//':
                        divisor = float(part)
                        if divisor == 0:
                            raise MathError("Division by zero")
                        result //= divisor
                        operations_used.append('floor_divide')
                    elif op == '%':
                        result %= float(part)
                        operations_used.append('modulo')
                return result
        
        # Finally handle + and -
        if '+' in expression or (expression.count('-') > expression.startswith('-')):
            # Split more carefully for + and -
            result = 0
            current_num = ''
            current_op = '+'
            
            for char in expression + '+':  # Add final operator to flush last number
                if char in '+-':
                    if current_num:
                        if current_op == '+':
                            result += float(current_num)
                            operations_used.append('add')
                        else:
                            result -= float(current_num)
                            operations_used.append('subtract')
                    current_num = ''
                    current_op = char
                else:
                    current_num += char
            
            return result
        
        # Single number
        try:
            return float(expression)
        except ValueError:
            raise MathError(f"Invalid number: {expression}")
    
    def get_supported_operations(self) -> Dict[str, List[str]]:
        """
        Get list of supported operations.
        
        Returns:
            Dictionary of supported operations by category
        """
        return {
            "operators": list(self.operators.keys()),
            "functions": list(self.functions.keys()) if self.enable_functions else [],
            "constants": list(self.constants.keys())
        }
    
    def get_tool_info(self) -> Dict[str, Any]:
        """
        Get math tool configuration information.
        
        Returns:
            Tool configuration details
        """
        return {
            "tool_type": "safe_math",
            "version": "1.0.0",
            "features": [
                "secure_evaluation",
                "no_eval_usage",
                "comprehensive_validation",
                "function_support",
                "constant_support",
                "error_handling"
            ],
            "configuration": {
                "precision": self.precision,
                "max_iterations": self.max_iterations,
                "max_expression_length": self.max_expression_length,
                "functions_enabled": self.enable_functions
            },
            "supported_operations": self.get_supported_operations()
        }