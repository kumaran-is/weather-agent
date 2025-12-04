"""Configuration module for Weather AI Agent Service.

This module provides configuration management using environment variables
and Pydantic settings.

Level 1 Implementation:
- Basic environment variable configuration
- Settings validation
- Configuration for MCP servers, LLM APIs, and databases

Example:
    >>> from backend.config.settings import settings
    >>> print(settings.ANTHROPIC_API_KEY)
"""

# Configuration will be loaded from settings.py when needed
__all__ = []
