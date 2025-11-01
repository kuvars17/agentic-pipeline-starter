"""Configuration management using Pydantic Settings.

This module provides centralized configuration for the agentic pipeline,
supporting environment variables and .env files.
"""

from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMMode(str, Enum):
    """LLM execution modes."""
    OLLAMA = "ollama"
    MOCK = "mock"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Settings
    app_name: str = Field(default="Agentic Pipeline Starter", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    
    # LLM Configuration
    llm_mode: LLMMode = Field(default=LLMMode.MOCK, description="LLM execution mode")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field(default="qwen2.5:3b", description="Ollama model name")
    ollama_timeout: int = Field(default=30, description="Ollama request timeout in seconds")
    
    # API Configuration
    api_host: str = Field(default="127.0.0.1", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_reload: bool = Field(default=False, description="Enable API auto-reload")
    
    # Pipeline Configuration
    max_retries: int = Field(default=3, description="Maximum number of retries for operations")
    request_timeout: int = Field(default=10, description="HTTP request timeout in seconds")
    
    # Security
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug or self.llm_mode == LLMMode.MOCK
    
    @property
    def ollama_url(self) -> str:
        """Get the complete Ollama API URL."""
        return f"{self.ollama_base_url.rstrip('/')}/api"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance.
    
    This function provides dependency injection for FastAPI and testing.
    """
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment (useful for testing)."""
    global settings
    settings = Settings()
    return settings