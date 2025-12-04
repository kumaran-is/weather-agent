"""MCP (Model Context Protocol) client integration.

This module provides async HTTP clients for interacting with MCP servers.
"""

from backend.src.mcp.weather_client import WeatherMCPClient

__all__ = ["WeatherMCPClient"]
