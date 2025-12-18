"""Weather tools for LangChain agents.

This module provides LangChain tool wrappers for the Weather MCP client.
Tools are decorated with @tool to make them available to LangChain agents.

Level 1 Implementation:
- 3 tools: get_current_weather, get_forecast, retrieve_weather_context
- MCP protocol integration with weather-mcp server
- Clear docstrings for LLM understanding
- Basic try/except error handling
- NO tool registry, discovery, or metrics tracking (deferred to L2+)

MCP Tool Mapping:
- get_current_weather → MCP: get_current_weather
- get_forecast → MCP: get_weather_forecast
- retrieve_weather_context → MCP: retrieve_weather_context
"""

from langchain_core.tools import tool

from backend.config.settings import settings
from backend.src.mcp.weather_client import WeatherMCPClient


# ✅ Factory pattern (v1.x compliant - dependency injection)
def get_weather_mcp_client() -> WeatherMCPClient:
    """Factory function to create WeatherMCPClient instance.

    ✅ v1.x: Dependency injection pattern (recommended for tests and production)

    Returns:
        WeatherMCPClient: Initialized MCP client instance
    """
    return WeatherMCPClient(base_url=settings.MCP_WEATHER_SERVER_URL)


# ⚠️ Module-level client (backward compatibility)
# For Level 1 simplicity, but prefer get_weather_mcp_client() in production
weather_mcp_client = get_weather_mcp_client()


@tool
async def get_current_weather(location: str) -> dict:
    """Get current weather conditions for a location.

    This tool retrieves real-time weather data including temperature, conditions,
    humidity, and wind speed for any city or geographic coordinates.

    Args:
        location: City name or coordinates (e.g., 'Seattle', 'London', or '47.6062,-122.3321')

    Returns:
        dict: Current weather data including:
            - temperature: Current temperature
            - condition: Weather condition (e.g., 'sunny', 'rainy', 'cloudy')
            - humidity: Humidity percentage
            - wind_speed: Wind speed

    Example:
        >>> result = await get_current_weather("London")
        >>> print(f"Temperature in London: {result['temperature']}°C")

    Note:
        The agent should use this tool when users ask about current, now, or
        present weather conditions.
    """
    try:
        # Ensure MCP client is initialized
        if not weather_mcp_client._initialized:
            await weather_mcp_client.initialize()

        # Get current weather from MCP server
        result = await weather_mcp_client.get_current_weather(location)
        return result
    except Exception as e:
        # Return error in structured format for agent to handle
        return {
            "error": f"Failed to get current weather for {location}: {str(e)}",
            "location": location,
            "status": "failed"
        }


@tool
async def get_forecast(location: str, days: int = 5) -> dict:
    """Get weather forecast for a location.

    This tool retrieves multi-day weather forecasts with daily predictions
    including high/low temperatures and conditions.

    Args:
        location: City name or coordinates (e.g., 'Seattle', 'London', or '47.6062,-122.3321')
        days: Number of days to forecast (1-7, default 5). Most useful range is 3-7 days.

    Returns:
        dict: Forecast data including:
            - forecast_7day: List of daily forecasts
            - Each forecast contains: day, high, low, condition

    Example:
        >>> result = await get_forecast("London", days=5)
        >>> for day in result['forecast_7day']:
        ...     print(f"{day['day']}: High {day['high']}°C, Low {day['low']}°C")

    Note:
        The agent should use this tool when users ask about:
        - Future weather (tomorrow, this week, next week)
        - Weather forecasts
        - Planning questions (e.g., "What should I wear this weekend?")
        - Extended outlook
    """
    try:
        # Ensure MCP client is initialized
        if not weather_mcp_client._initialized:
            await weather_mcp_client.initialize()

        # Validate days parameter
        if not 1 <= days <= 7:
            days = 5  # Default to 5 days if out of range

        # Get forecast from MCP server
        result = await weather_mcp_client.get_forecast(location, days=days)
        return result
    except Exception as e:
        # Return error in structured format for agent to handle
        return {
            "error": f"Failed to get forecast for {location}: {str(e)}",
            "location": location,
            "days": days,
            "status": "failed"
        }


@tool
async def retrieve_weather_context(query: str) -> dict:
    """Retrieve weather context from natural language queries.

    This tool extracts weather information from natural language queries without
    requiring explicit location parameters. Useful for handling complex user questions
    that contain implicit location references.

    Args:
        query: Natural language query containing weather-related questions
              (e.g., "weather in Paris for my trip", "Should I bring an umbrella to London?",
               "What's it like in Tokyo tomorrow?")

    Returns:
        dict: Weather context extracted from the query, including:
            - Detected location
            - Relevant weather data
            - Contextual information for the query

    Example:
        >>> result = await retrieve_weather_context("Is it rainy in Seattle today?")
        >>> print(f"Context: {result}")

    Note:
        The agent should use this tool when users ask complex questions where:
        - Location is embedded in natural language
        - Context extraction is needed (e.g., "for travel", "for outdoor activity")
        - Multiple intents are present in one query
        - Ambiguous or conversational queries
    """
    try:
        # Ensure MCP client is initialized
        if not weather_mcp_client._initialized:
            await weather_mcp_client.initialize()

        # Get weather context from MCP server
        result = await weather_mcp_client.retrieve_weather_context(query)
        return result
    except Exception as e:
        # Return error in structured format for agent to handle
        return {
            "error": f"Failed to retrieve weather context for query: {str(e)}",
            "query": query,
            "status": "failed"
        }
