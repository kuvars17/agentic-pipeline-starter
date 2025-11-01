"""Agentic Pipeline Starter - A production-quality agentic AI pipeline.

This package provides a complete agentic AI system using LangGraph, Pydantic, 
FastAPI, and local LLMs (Ollama) for building autonomous reasoning workflows.
"""

__version__ = "0.1.0"
__author__ = "Buğra Tokat"
__email__ = "bugra.tokat@icloud.com"

from .config import Settings

__all__ = ["Settings"]