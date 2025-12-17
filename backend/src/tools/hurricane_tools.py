"""Hurricane tools for LangChain agents.

This module provides LangChain tool wrappers for the Hurricane MCP client.
Tools are decorated with @tool to make them available to LangChain agents.

Level 7 Implementation:
- 4 tools: get_active_storms, get_storm_forecast, get_hurricane_alerts, get_storm_history
- MCP protocol integration with hurricane-tracker-mcp server
- Clear docstrings for LLM understanding
- Basic try/except error handling

MCP Tool Mapping:
- get_active_storms → MCP: get_active_storms
- get_storm_forecast → MCP: get_storm_cone + get_storm_track
- get_hurricane_alerts → MCP: get_local_hurricane_alerts
- get_storm_history → MCP: search_historical_tracks

CRITICAL: These tools are only registered when MCP_HURRICANE_SERVER_ENABLED=true
"""

import logging

from langchain_core.tools import tool

from backend.config.settings import settings
from backend.src.mcp.hurricane_client import HurricaneMCPClient

logger = logging.getLogger(__name__)


# ✅ Factory pattern (v1.x compliant - dependency injection)
def get_hurricane_mcp_client() -> HurricaneMCPClient:
    """Factory function to create HurricaneMCPClient instance.

    ✅ v1.x: Dependency injection pattern (recommended for tests and production)

    Returns:
        HurricaneMCPClient: Initialized MCP client instance
    """
    return HurricaneMCPClient(base_url=settings.MCP_HURRICANE_SERVER_URL)


# ⚠️ Module-level client (backward compatibility)
# Only initialize if hurricane server is enabled
hurricane_mcp_client: HurricaneMCPClient | None = None

if settings.MCP_HURRICANE_SERVER_ENABLED:
    hurricane_mcp_client = get_hurricane_mcp_client()


@tool
async def get_active_storms(basin: str = "") -> dict:
    """Get currently active tropical storms and hurricanes.

    This tool retrieves real-time data about active tropical storms and hurricanes
    from the National Hurricane Center. Use this for current storm information.

    Args:
        basin: Optional 2-letter basin code (e.g., 'AL' for Atlantic, 'EP' for East Pacific).
               Leave empty for storms from all basins.

    Returns:
        dict containing:
            - storms: List of active storms with name, category, wind_speed, location
            - basin: Basin code if filtered
            - timestamp: When the data was retrieved

    Example queries:
        - "What hurricanes are active right now?"
        - "Are there any storms in the Atlantic?"
        - "Show me current tropical storms"
    """
    if not settings.MCP_HURRICANE_SERVER_ENABLED:
        return {
            "error": "Hurricane MCP server is not enabled",
            "storms": [],
            "message": "Set MCP_HURRICANE_SERVER_ENABLED=true to enable hurricane tracking"
        }

    try:
        client = hurricane_mcp_client or get_hurricane_mcp_client()

        # Initialize client if needed
        if not client._initialized:
            await client.initialize()

        # Get active storms
        basin_param = basin.upper() if basin else None
        result = await client.get_active_storms(basin=basin_param)

        return {
            "storms": result.get("storms", []),
            "basin": basin_param or "ALL",
            "count": len(result.get("storms", [])),
            "source": "National Hurricane Center via MCP",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error getting active storms: {e}")
        return {
            "error": str(e),
            "storms": [],
            "status": "error",
            "message": "Failed to retrieve active storm data"
        }


@tool
async def get_storm_forecast(storm_id: str) -> dict:
    """Get forecast information for a specific storm including track and cone of uncertainty.

    This tool retrieves the forecast cone (cone of uncertainty) and predicted
    track for a specific tropical storm or hurricane.

    Args:
        storm_id: Storm identifier (e.g., 'AL092023' for a 2023 Atlantic storm).
                  Format: {basin}{number}{year}

    Returns:
        dict containing:
            - storm_id: The requested storm identifier
            - forecast_track: Predicted storm path with positions and times
            - cone: Cone of uncertainty coordinates
            - intensity_forecast: Predicted intensity changes

    Example queries:
        - "What is the forecast for hurricane AL092023?"
        - "Where is the storm expected to go?"
        - "Show me the cone of uncertainty for the current hurricane"
    """
    if not settings.MCP_HURRICANE_SERVER_ENABLED:
        return {
            "error": "Hurricane MCP server is not enabled",
            "forecast": None,
            "message": "Set MCP_HURRICANE_SERVER_ENABLED=true to enable hurricane tracking"
        }

    if not storm_id:
        return {
            "error": "Storm ID is required",
            "forecast": None,
            "message": "Please provide a valid storm ID (e.g., 'AL092023')"
        }

    try:
        client = hurricane_mcp_client or get_hurricane_mcp_client()

        # Initialize client if needed
        if not client._initialized:
            await client.initialize()

        # Get both cone and track data
        cone_data = await client.get_storm_cone(storm_id)
        track_data = await client.get_storm_track(storm_id)

        return {
            "storm_id": storm_id,
            "forecast_cone": cone_data.get("cone_coordinates"),
            "forecast_track": track_data.get("forecast_positions", []),
            "historical_track": track_data.get("historical_positions", []),
            "intensity_forecast": cone_data.get("intensity_forecast"),
            "uncertainty": cone_data.get("uncertainty"),
            "source": "National Hurricane Center via MCP",
            "status": "success"
        }

    except ValueError as e:
        return {
            "error": str(e),
            "storm_id": storm_id,
            "forecast": None,
            "status": "error"
        }
    except Exception as e:
        logger.error(f"Error getting storm forecast for {storm_id}: {e}")
        return {
            "error": str(e),
            "storm_id": storm_id,
            "forecast": None,
            "status": "error",
            "message": f"Failed to retrieve forecast for storm {storm_id}"
        }


@tool
async def get_hurricane_alerts(latitude: float, longitude: float) -> dict:
    """Get hurricane alerts and warnings for a specific location.

    This tool retrieves hurricane watches, warnings, and alerts for a given
    geographic location. Essential for evacuation and safety decisions.

    CRITICAL: This is a LIFE-SAFETY tool. Always provide accurate, timely alerts.

    Args:
        latitude: Latitude of the location (-90 to 90)
        longitude: Longitude of the location (-180 to 180)

    Returns:
        dict containing:
            - alerts: List of active alerts (watches, warnings)
            - evacuation_zones: Affected evacuation zones
            - storm_surge: Storm surge warnings if applicable
            - wind_threat: Wind threat level

    Example queries:
        - "Are there any hurricane warnings for Miami?"
        - "Should I evacuate from Tampa?"
        - "What hurricane alerts are there for my location?"
    """
    if not settings.MCP_HURRICANE_SERVER_ENABLED:
        return {
            "error": "Hurricane MCP server is not enabled",
            "alerts": [],
            "message": "Set MCP_HURRICANE_SERVER_ENABLED=true to enable hurricane tracking"
        }

    # Validate coordinates
    if not (-90 <= latitude <= 90):
        return {
            "error": "Invalid latitude. Must be between -90 and 90.",
            "alerts": [],
            "status": "error"
        }

    if not (-180 <= longitude <= 180):
        return {
            "error": "Invalid longitude. Must be between -180 and 180.",
            "alerts": [],
            "status": "error"
        }

    try:
        client = hurricane_mcp_client or get_hurricane_mcp_client()

        # Initialize client if needed
        if not client._initialized:
            await client.initialize()

        # Get local alerts
        result = await client.get_local_hurricane_alerts(
            latitude=latitude,
            longitude=longitude
        )

        return {
            "location": {"latitude": latitude, "longitude": longitude},
            "alerts": result.get("alerts", []),
            "watches": result.get("watches", []),
            "warnings": result.get("warnings", []),
            "evacuation_zones": result.get("evacuation_zones", []),
            "storm_surge": result.get("storm_surge_warning"),
            "wind_threat": result.get("wind_threat_level"),
            "source": "National Hurricane Center via MCP",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error getting hurricane alerts for ({latitude}, {longitude}): {e}")
        return {
            "error": str(e),
            "location": {"latitude": latitude, "longitude": longitude},
            "alerts": [],
            "status": "error",
            "message": "Failed to retrieve hurricane alerts for this location"
        }


@tool
async def get_storm_history(
    start_year: int = 2020,
    end_year: int = 2024,
    basin: str = "AL",
    min_category: int = 1
) -> dict:
    """Search historical hurricane and tropical storm records.

    This tool searches historical storm data to find past hurricanes matching
    specified criteria. Useful for historical comparisons and pattern analysis.

    Args:
        start_year: Start year for search (default: 2020)
        end_year: End year for search (default: 2024)
        basin: Basin code - 'AL' (Atlantic), 'EP' (East Pacific), etc. (default: 'AL')
        min_category: Minimum hurricane category to include (1-5, default: 1)

    Returns:
        dict containing:
            - storms: List of matching historical storms
            - count: Number of storms found
            - criteria: Search parameters used

    Example queries:
        - "What major hurricanes hit in 2023?"
        - "Show me Category 4+ hurricanes in the Atlantic since 2020"
        - "Historical hurricane data for comparison"
    """
    if not settings.MCP_HURRICANE_SERVER_ENABLED:
        return {
            "error": "Hurricane MCP server is not enabled",
            "storms": [],
            "message": "Set MCP_HURRICANE_SERVER_ENABLED=true to enable hurricane tracking"
        }

    # Validate inputs
    current_year = 2024  # Update as needed
    if start_year > end_year:
        return {
            "error": "Start year must be before or equal to end year",
            "storms": [],
            "status": "error"
        }

    if min_category < 1 or min_category > 5:
        return {
            "error": "Category must be between 1 and 5",
            "storms": [],
            "status": "error"
        }

    try:
        client = hurricane_mcp_client or get_hurricane_mcp_client()

        # Initialize client if needed
        if not client._initialized:
            await client.initialize()

        # Search historical tracks
        result = await client.search_historical_tracks(
            start_year=start_year,
            end_year=end_year,
            basin=basin.upper(),
            min_category=min_category
        )

        storms = result.get("tracks", [])

        return {
            "storms": storms,
            "count": len(storms),
            "criteria": {
                "start_year": start_year,
                "end_year": end_year,
                "basin": basin.upper(),
                "min_category": min_category
            },
            "source": "National Hurricane Center Historical Database via MCP",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error searching historical storms: {e}")
        return {
            "error": str(e),
            "storms": [],
            "criteria": {
                "start_year": start_year,
                "end_year": end_year,
                "basin": basin,
                "min_category": min_category
            },
            "status": "error",
            "message": "Failed to search historical storm data"
        }


# Export tools for registration
__all__ = [
    "get_active_storms",
    "get_storm_forecast",
    "get_hurricane_alerts",
    "get_storm_history",
    "get_hurricane_mcp_client",
]
