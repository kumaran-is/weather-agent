"""Comprehensive MCP Integration Tests with LangGraph-bigtool.

This module validates:
1. Both MCP clients have resilience patterns (timeout, retry, circuit breaker)
2. BigtoolRegistry tool registration works correctly
3. Hurricane tools are conditionally registered
4. Resilience patterns are preserved when using tools

NOTE: Some tests use direct module imports to avoid circular import issues
that exist in the codebase's __init__.py files.

Run with:
    OPENAI_API_KEY=sk-test... MCP_HURRICANE_SERVER_ENABLED=true pytest tests/integration/test_mcp_integration.py -v
"""

import pytest


class TestSettingsConfiguration:
    """Test MCP settings configuration (no circular imports)."""

    def test_weather_settings_exist(self):
        """Verify weather MCP settings exist."""
        from backend.config.settings import settings

        assert hasattr(settings, 'MCP_WEATHER_SERVER_URL')
        assert hasattr(settings, 'MCP_WEATHER_SERVER_ENABLED')
        assert hasattr(settings, 'MCP_WEATHER_REQUEST_TIMEOUT')
        assert hasattr(settings, 'MCP_WEATHER_MAX_RETRIES')
        assert hasattr(settings, 'MCP_WEATHER_RETRY_DELAY')

    def test_hurricane_settings_exist(self):
        """Verify hurricane MCP settings exist."""
        from backend.config.settings import settings

        assert hasattr(settings, 'MCP_HURRICANE_SERVER_URL')
        assert hasattr(settings, 'MCP_HURRICANE_SERVER_ENABLED')
        assert hasattr(settings, 'MCP_HURRICANE_REQUEST_TIMEOUT')
        assert hasattr(settings, 'MCP_HURRICANE_MAX_RETRIES')
        assert hasattr(settings, 'MCP_HURRICANE_RETRY_DELAY')

    def test_failover_setting_exists(self):
        """Verify failover setting exists."""
        from backend.config.settings import settings

        assert hasattr(settings, 'MCP_ENABLE_FAILOVER')

    def test_timeout_settings_are_positive(self):
        """Verify timeout settings are positive integers."""
        from backend.config.settings import settings

        assert settings.MCP_WEATHER_REQUEST_TIMEOUT > 0
        assert settings.MCP_HURRICANE_REQUEST_TIMEOUT > 0

    def test_retry_settings_are_positive(self):
        """Verify retry settings are positive."""
        from backend.config.settings import settings

        assert settings.MCP_WEATHER_MAX_RETRIES > 0
        assert settings.MCP_HURRICANE_MAX_RETRIES > 0
        assert settings.MCP_WEATHER_RETRY_DELAY > 0
        assert settings.MCP_HURRICANE_RETRY_DELAY > 0


class TestMCPLoggerStandalone:
    """Test MCPLogger in isolation."""

    def test_weather_logger_creation(self):
        """Verify MCPLogger can be created for weather."""
        # Direct import to avoid circular import chain
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "logger", "/app/backend/src/mcp/logger.py"
        )
        logger_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(logger_module)
        MCPLogger = logger_module.MCPLogger

        logger = MCPLogger("weather")
        assert logger.server_name == "weather"

    def test_hurricane_logger_creation(self):
        """Verify MCPLogger can be created for hurricane."""
        # Direct import to avoid circular import chain
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "logger", "/app/backend/src/mcp/logger.py"
        )
        logger_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(logger_module)
        MCPLogger = logger_module.MCPLogger

        logger = MCPLogger("hurricane")
        assert logger.server_name == "hurricane"

    def test_correlation_id_generation(self):
        """Verify correlation ID generation."""
        # Direct import to avoid circular import chain
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "logger", "/app/backend/src/mcp/logger.py"
        )
        logger_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(logger_module)
        MCPLogger = logger_module.MCPLogger

        logger = MCPLogger("weather")
        correlation_id = logger.generate_correlation_id()

        assert correlation_id is not None
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0


class TestCircuitBreakerStandalone:
    """Test circuit breaker in isolation."""

    def test_circuit_breaker_registry_exists(self):
        """Verify circuit breaker registry exists."""
        from backend.src.agents.circuit_breaker import circuit_registry

        assert circuit_registry is not None

    def test_circuit_breaker_creation(self):
        """Verify circuit breaker can be created."""
        from backend.src.agents.circuit_breaker import circuit_registry

        breaker = circuit_registry.get_or_create(
            agent_name="test_mcp_server",
            failure_threshold=3,
            success_threshold=2,
            recovery_timeout=30,
        )

        assert breaker is not None
        assert breaker.failure_threshold == 3
        assert breaker.success_threshold == 2
        assert breaker.recovery_timeout_seconds == 30


class TestMCPFailoverStandalone:
    """Test MCPFailoverHandler in isolation."""

    def test_failover_handler_weather(self):
        """Verify MCPFailoverHandler for weather."""
        from backend.src.mcp.logger import MCPLogger
        from backend.src.mcp.failover import MCPFailoverHandler

        logger = MCPLogger("weather")
        handler = MCPFailoverHandler("weather", logger, enable_failover=True)

        assert handler.server_name == "weather"
        assert handler._circuit_breaker_name == "mcp_weather_server"
        assert handler._circuit_breaker is not None

    def test_failover_handler_hurricane(self):
        """Verify MCPFailoverHandler for hurricane."""
        from backend.src.mcp.logger import MCPLogger
        from backend.src.mcp.failover import MCPFailoverHandler

        logger = MCPLogger("hurricane")
        handler = MCPFailoverHandler("hurricane", logger, enable_failover=True)

        assert handler.server_name == "hurricane"
        assert handler._circuit_breaker_name == "mcp_hurricane_server"
        assert handler._circuit_breaker is not None

    def test_circuit_breaker_configuration(self):
        """Verify circuit breaker has correct configuration."""
        from backend.src.mcp.logger import MCPLogger
        from backend.src.mcp.failover import MCPFailoverHandler

        logger = MCPLogger("weather")
        handler = MCPFailoverHandler("weather", logger, enable_failover=True)

        breaker = handler._circuit_breaker

        assert breaker.failure_threshold == 3, "Should open after 3 failures"
        assert breaker.success_threshold == 2, "Should close after 2 successes"
        assert breaker.recovery_timeout_seconds == 30, "Should recover after 30s"


class TestBigtoolRegistryStandalone:
    """Test BigtoolRegistry in isolation."""

    def test_registry_singleton(self):
        """Verify registry is singleton."""
        from backend.src.registry import BigtoolRegistry, reset_bigtool_registry

        # Reset
        reset_bigtool_registry()

        r1 = BigtoolRegistry()
        r2 = BigtoolRegistry()

        assert r1 is r2

    def test_registry_manual_registration(self):
        """Verify manual tool registration."""
        from backend.src.registry import BigtoolRegistry, reset_bigtool_registry
        from backend.src.registry.bigtool_registry import ToolCategory
        from langchain_core.tools import StructuredTool

        # Reset
        reset_bigtool_registry()
        registry = BigtoolRegistry()
        registry._auto_registered = True  # Skip auto-registration

        def dummy_func(x: str = "") -> str:
            return x

        tool = StructuredTool.from_function(
            func=dummy_func,
            name="test_weather_tool",
            description="Test tool",
        )

        registry.register_tool(
            tool=tool,
            description="Test tool",
            category=ToolCategory.WEATHER_DATA,
        )

        assert registry.get_tool("test_weather_tool") is not None
        assert registry.count_tools() == 1

    def test_registry_statistics(self):
        """Verify registry statistics."""
        from backend.src.registry import BigtoolRegistry, reset_bigtool_registry
        from backend.src.registry.bigtool_registry import ToolCategory
        from langchain_core.tools import StructuredTool

        # Reset
        reset_bigtool_registry()
        registry = BigtoolRegistry()
        registry._auto_registered = True

        def dummy_func(x: str = "") -> str:
            return x

        for i, category in enumerate([
            ToolCategory.WEATHER_DATA,
            ToolCategory.RAG,
        ]):
            tool = StructuredTool.from_function(
                func=dummy_func,
                name=f"test_tool_{i}",
                description=f"Test tool {i}",
            )
            registry.register_tool(
                tool=tool,
                description=f"Test tool {i}",
                category=category,
            )

        stats = registry.get_statistics()

        assert stats.total_tools == 2
        assert "weather_data" in stats.category_counts
        assert "rag" in stats.category_counts
        assert stats.store_backend == "InMemoryStore"


class TestAutoRegistration:
    """Test auto-registration (may trigger circular imports - skip if needed)."""

    def test_get_bigtool_registry_triggers_auto_registration(self):
        """Verify get_bigtool_registry() triggers auto-registration."""
        from backend.src.registry import BigtoolRegistry, get_bigtool_registry, reset_bigtool_registry

        # Reset registry
        reset_bigtool_registry()

        # Get registry (should trigger auto-registration)
        registry = get_bigtool_registry()

        # Should have at least weather tools registered
        assert registry.count_tools() >= 6, f"Expected at least 6 tools, got {registry.count_tools()}"
        assert registry._auto_registered is True

        # Verify weather tools exist
        tool_names = registry.get_tool_names()
        assert "get_current_weather" in tool_names
        assert "get_forecast" in tool_names


class TestHurricaneToolsConditional:
    """Test hurricane tools conditional registration."""

    def test_hurricane_tools_registration_flag(self):
        """Verify hurricane registration depends on MCP_HURRICANE_SERVER_ENABLED."""
        from backend.config.settings import settings
        from backend.src.registry import BigtoolRegistry, get_bigtool_registry, reset_bigtool_registry

        # Reset registry
        reset_bigtool_registry()
        registry = get_bigtool_registry()

        tool_names = registry.get_tool_names()

        if settings.MCP_HURRICANE_SERVER_ENABLED:
            # Should have hurricane tools
            assert "get_active_storms" in tool_names, "get_active_storms should be registered"
            assert "get_storm_forecast" in tool_names, "get_storm_forecast should be registered"
            assert "get_hurricane_alerts" in tool_names, "get_hurricane_alerts should be registered"
            assert "get_storm_history" in tool_names, "get_storm_history should be registered"
            print(f"✅ Hurricane tools registered: 4 tools")
        else:
            # Should NOT have hurricane tools
            assert "get_active_storms" not in tool_names, "get_active_storms should NOT be registered when disabled"
            print(f"✅ Hurricane tools NOT registered (MCP_HURRICANE_SERVER_ENABLED=false)")

    def test_tool_count_with_hurricane_enabled(self):
        """Verify tool count when hurricane is enabled."""
        from backend.config.settings import settings
        from backend.src.registry import BigtoolRegistry, get_bigtool_registry, reset_bigtool_registry

        # Reset registry
        reset_bigtool_registry()
        registry = get_bigtool_registry()

        count = registry.count_tools()

        if settings.MCP_HURRICANE_SERVER_ENABLED:
            # 6 weather/rag tools + 4 hurricane tools = 10
            assert count >= 10, f"Expected at least 10 tools with hurricane enabled, got {count}"
            print(f"✅ Total tools with hurricane: {count}")
        else:
            # 6 weather/rag tools
            assert count >= 6, f"Expected at least 6 tools without hurricane, got {count}"
            print(f"✅ Total tools without hurricane: {count}")


class TestEndToEndAPIFormat:
    """Test the output format matches API expectations."""

    def test_statistics_endpoint_compatible_output(self):
        """Verify registry statistics can be serialized for API response."""
        from backend.src.registry import BigtoolRegistry, get_bigtool_registry, reset_bigtool_registry

        # Reset registry
        reset_bigtool_registry()
        registry = get_bigtool_registry()

        stats = registry.get_statistics()

        # Convert to dict (as API would)
        stats_dict = {
            "total_tools": stats.total_tools,
            "category_counts": stats.category_counts,
            "store_backend": stats.store_backend,
            "last_search_query": stats.last_search_query,
            "last_search_results": stats.last_search_results,
            "tools": [
                {
                    "name": t.name,
                    "category": t.category,  # category is already a string
                    "description": t.description,
                    "tags": t.tags,
                    "is_available": t.is_available,
                }
                for t in registry.list_tools()
            ],
        }

        # Validate types
        assert isinstance(stats_dict["total_tools"], int)
        assert isinstance(stats_dict["category_counts"], dict)
        assert isinstance(stats_dict["store_backend"], str)
        assert isinstance(stats_dict["tools"], list)

        # Validate content
        assert stats_dict["total_tools"] >= 6
        assert "weather_data" in stats_dict["category_counts"]
        assert stats_dict["store_backend"] == "InMemoryStore"

        print(f"✅ API format validated: {stats_dict['total_tools']} tools")


class TestBackwardCompatibility:
    """Test backward compatibility aliases."""

    def test_tool_registry_alias(self):
        """Verify ToolRegistry alias points to BigtoolRegistry."""
        from backend.src.registry import ToolRegistry, BigtoolRegistry

        assert ToolRegistry is BigtoolRegistry

    def test_get_tool_registry_alias(self):
        """Verify get_tool_registry alias works."""
        from backend.src.registry import get_tool_registry, BigtoolRegistry, reset_bigtool_registry

        reset_bigtool_registry()
        registry = get_tool_registry()
        assert isinstance(registry, BigtoolRegistry)


class TestSemanticSearch:
    """Test semantic search capabilities."""

    def test_search_tools_method_exists(self):
        """Verify search_tools method exists on registry."""
        from backend.src.registry import get_bigtool_registry, reset_bigtool_registry

        reset_bigtool_registry()
        registry = get_bigtool_registry()

        assert hasattr(registry, "search_tools")
        assert callable(registry.search_tools)

    def test_retrieve_tools_for_query_function(self):
        """Verify retrieve_tools_for_query function exists."""
        from backend.src.registry import retrieve_tools_for_query

        assert callable(retrieve_tools_for_query)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
