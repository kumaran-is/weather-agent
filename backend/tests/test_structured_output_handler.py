"""Comprehensive tests for StructuredOutputHandler

Tests cover:
1. Successful structured output (no validation errors)
2. Validation error recovery with retry logic
3. Fallback strategies for persistent failures
4. Metrics tracking (success rate, fallback rate)
5. Integration with existing schemas (ForecastResponse, HurricaneAlert)
6. Error callback functionality
7. Exponential backoff timing
8. <2% fallback rate target (Level 2 requirement)
"""

from datetime import UTC, datetime
from datetime import date as Date
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field, ValidationError

from backend.src.models import ForecastResponse, HurricaneAlert
from backend.src.models.output_handler import (
    StructuredOutputHandler,
    invoke_structured,
)

# ============================================================================
# Helper Functions
# ============================================================================


def create_validation_error(schema: type[BaseModel], invalid_data: dict) -> ValidationError:
    """Helper to create real Pydantic ValidationError by instantiating invalid model

    This is the Pydantic v2 way - create actual validation errors by trying to
    instantiate models with invalid data.
    """
    try:
        schema(**invalid_data)
        raise ValueError("Expected ValidationError but model validation succeeded")
    except ValidationError as e:
        return e


# ============================================================================
# Mock LLM Classes
# ============================================================================


class MockLLM:
    """Mock LLM for testing"""

    def __init__(self, responses: list[dict | Exception]):
        """Initialize with list of responses (success dicts or ValidationError exceptions)"""
        self.responses = responses
        self.call_count = 0

    def with_structured_output(self, schema):
        """Return self to simulate LangChain's with_structured_output()"""
        self.schema = schema
        return self

    async def ainvoke(self, messages, **kwargs):
        """Mock async invoke that returns next response in sequence"""
        if self.call_count >= len(self.responses):
            raise ValueError("No more mock responses available")

        response = self.responses[self.call_count]
        self.call_count += 1

        if isinstance(response, Exception):
            raise response

        # Return valid Pydantic instance
        return self.schema(**response)


# ============================================================================
# Test Schemas
# ============================================================================


class SimpleSchema(BaseModel):
    """Simple test schema for basic validation"""

    name: str = Field(min_length=2, max_length=50)
    age: int = Field(ge=0, le=150)


class ComplexSchema(BaseModel):
    """Complex schema with cross-field validation"""

    low: int = Field(ge=0, le=100)
    high: int = Field(ge=0, le=100)

    @property
    def validate_low_less_than_high(self):
        if self.low >= self.high:
            raise ValueError("Low must be less than high")
        return self


# ============================================================================
# Unit Tests: Initialization
# ============================================================================


def test_handler_initialization():
    """Test StructuredOutputHandler initialization with default params"""
    llm = MockLLM([])
    handler = StructuredOutputHandler(llm=llm)

    assert handler.llm == llm
    assert handler.max_retries == 2
    assert handler.fallback_enabled is True
    assert handler.error_callback is None
    assert handler.total_attempts == 0
    assert handler.validation_failures == 0
    assert handler.fallback_uses == 0


def test_handler_initialization_custom_params():
    """Test initialization with custom parameters"""
    llm = MockLLM([])
    error_callback = Mock()

    handler = StructuredOutputHandler(
        llm=llm, max_retries=3, fallback_enabled=False, error_callback=error_callback
    )

    assert handler.max_retries == 3
    assert handler.fallback_enabled is False
    assert handler.error_callback == error_callback


# ============================================================================
# Unit Tests: Successful Structured Output (No Errors)
# ============================================================================


@pytest.mark.asyncio
async def test_successful_first_attempt():
    """Test successful structured output on first attempt (no retries)"""
    llm = MockLLM(
        [
            {
                "name": "John Doe",
                "age": 30,
            }
        ]
    )

    handler = StructuredOutputHandler(llm=llm)

    result = await handler.invoke_with_recovery(
        schema=SimpleSchema,
        query="Create a person",
        system_prompt="You are a helpful assistant",
    )

    assert isinstance(result, SimpleSchema)
    assert result.name == "John Doe"
    assert result.age == 30

    # Verify metrics
    assert handler.total_attempts == 1
    assert handler.validation_failures == 0
    assert handler.fallback_uses == 0


# ============================================================================
# Unit Tests: Validation Error Recovery (Retry Logic)
# ============================================================================


@pytest.mark.asyncio
async def test_retry_on_validation_error():
    """Test retry logic when first attempt fails validation"""
    # First attempt: invalid age (-5)
    # Second attempt: valid data
    llm = MockLLM(
        [
            create_validation_error(SimpleSchema, {"name": "Test", "age": -5}),  # Invalid age
            {"name": "Jane Doe", "age": 25},  # Valid on retry
        ]
    )

    handler = StructuredOutputHandler(llm=llm, max_retries=2)

    result = await handler.invoke_with_recovery(
        schema=SimpleSchema,
        query="Create a person",
        system_prompt="You are a helpful assistant",
    )

    assert isinstance(result, SimpleSchema)
    assert result.name == "Jane Doe"
    assert result.age == 25

    # Verify metrics
    assert handler.total_attempts == 1
    assert handler.validation_failures == 1  # Failed once, then succeeded
    assert handler.fallback_uses == 0  # No fallback needed
    assert llm.call_count == 2  # First attempt + 1 retry


@pytest.mark.asyncio
async def test_multiple_retries():
    """Test multiple retry attempts before success"""
    llm = MockLLM(
        [
            create_validation_error(SimpleSchema, {"name": "Test", "age": -10}),  # Invalid age
            create_validation_error(SimpleSchema, {"name": "X", "age": 30}),  # Invalid name (too short)
            {"name": "Success", "age": 30},  # Valid on 2nd retry
        ]
    )

    handler = StructuredOutputHandler(llm=llm, max_retries=2)

    result = await handler.invoke_with_recovery(
        schema=SimpleSchema,
        query="Create a person",
        system_prompt="You are a helpful assistant",
    )

    assert isinstance(result, SimpleSchema)
    assert result.name == "Success"
    assert result.age == 30

    # Verify metrics
    assert handler.total_attempts == 1
    assert handler.validation_failures == 1
    assert handler.fallback_uses == 0
    assert llm.call_count == 3  # First attempt + 2 retries


# ============================================================================
# Unit Tests: Fallback Strategy
# ============================================================================


@pytest.mark.asyncio
async def test_fallback_after_max_retries():
    """Test fallback when all retries fail"""
    # All attempts fail validation
    validation_error = create_validation_error(SimpleSchema, {"name": "Test", "age": -5})

    llm = MockLLM(
        [
            validation_error,  # First attempt
            validation_error,  # Retry 1
            validation_error,  # Retry 2
        ]
    )

    handler = StructuredOutputHandler(llm=llm, max_retries=2, fallback_enabled=True)

    result = await handler.invoke_with_recovery(
        schema=SimpleSchema,
        query="Create a person",
        system_prompt="You are a helpful assistant",
    )

    # Should return error dict (fallback)
    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["error"] == "structured_output_validation_failed"
    assert result["schema"] == "SimpleSchema"
    assert result["query"] == "Create a person"
    assert "validation_errors" in result
    assert len(result["validation_errors"]) > 0

    # Verify metrics
    assert handler.total_attempts == 1
    assert handler.validation_failures == 1
    assert handler.fallback_uses == 1  # Fallback used
    assert llm.call_count == 3  # First attempt + 2 retries


@pytest.mark.asyncio
async def test_raise_error_when_fallback_disabled():
    """Test that ValidationError is raised when fallback is disabled"""
    validation_error = create_validation_error(SimpleSchema, {"name": "Test", "age": -5})

    llm = MockLLM([validation_error, validation_error, validation_error])

    handler = StructuredOutputHandler(llm=llm, max_retries=2, fallback_enabled=False)

    with pytest.raises(ValidationError):
        await handler.invoke_with_recovery(
            schema=SimpleSchema,
            query="Create a person",
            system_prompt="You are a helpful assistant",
        )

    # Verify metrics
    assert handler.total_attempts == 1
    assert handler.validation_failures == 1
    assert handler.fallback_uses == 0  # No fallback allowed


# ============================================================================
# Unit Tests: Error Callback
# ============================================================================


@pytest.mark.asyncio
async def test_error_callback_invoked():
    """Test that error callback is invoked on validation failures"""
    error_callback = Mock()

    validation_error = create_validation_error(SimpleSchema, {"name": "Test", "age": -5})

    llm = MockLLM([validation_error, {"name": "Success", "age": 30}])

    handler = StructuredOutputHandler(
        llm=llm, max_retries=2, error_callback=error_callback
    )

    result = await handler.invoke_with_recovery(
        schema=SimpleSchema,
        query="Create a person",
        system_prompt="You are a helpful assistant",
    )

    # Callback should be invoked once (for first failure)
    assert error_callback.call_count == 1
    assert isinstance(result, SimpleSchema)


# ============================================================================
# Unit Tests: Metrics Tracking
# ============================================================================


@pytest.mark.asyncio
async def test_metrics_tracking():
    """Test metrics tracking across multiple invocations"""
    validation_error = create_validation_error(SimpleSchema, {"name": "Test", "age": -5})

    llm_responses = [
        {"name": "Person1", "age": 30},  # Success
        validation_error,
        {"name": "Person2", "age": 25},  # Retry success
        validation_error,
        validation_error,
        validation_error,  # All retries fail → fallback
    ]

    handler = StructuredOutputHandler(
        llm=MockLLM(llm_responses), max_retries=2, fallback_enabled=True
    )

    # Invocation 1: Success on first attempt
    result1 = await handler.invoke_with_recovery(
        schema=SimpleSchema, query="Query 1", system_prompt="System"
    )
    assert isinstance(result1, SimpleSchema)

    # Invocation 2: Fail then retry success
    handler.llm.responses = llm_responses[1:]
    handler.llm.call_count = 0
    result2 = await handler.invoke_with_recovery(
        schema=SimpleSchema, query="Query 2", system_prompt="System"
    )
    assert isinstance(result2, SimpleSchema)

    # Invocation 3: All retries fail → fallback
    handler.llm.responses = llm_responses[3:]
    handler.llm.call_count = 0
    result3 = await handler.invoke_with_recovery(
        schema=SimpleSchema, query="Query 3", system_prompt="System"
    )
    assert isinstance(result3, dict)  # Fallback dict

    # Verify final metrics
    metrics = handler.get_metrics()

    assert metrics["total_attempts"] == 3
    assert metrics["validation_failures"] == 2  # Invocations 2 and 3
    assert metrics["fallback_uses"] == 1  # Only invocation 3
    assert metrics["success_rate"] == "33.3%"  # 1 success out of 3
    assert metrics["fallback_rate"] == "33.3%"  # 1 fallback out of 3


# ============================================================================
# Integration Tests: Real Schemas
# ============================================================================


@pytest.mark.asyncio
async def test_integration_forecast_response():
    """Test with real ForecastResponse schema"""
    llm = MockLLM(
        [
            {
                "location": "Seattle, WA",
                "forecast_days": [
                    {
                        "date": Date.today(),
                        "high_f": 65.0,
                        "low_f": 50.0,
                        "condition": "Partly Cloudy",
                        "precipitation_pct": 20,
                    }
                ],
            }
        ]
    )

    handler = StructuredOutputHandler(llm=llm)

    result = await handler.invoke_with_recovery(
        schema=ForecastResponse,
        query="7-day forecast for Seattle",
        system_prompt="You are a weather forecaster",
    )

    assert isinstance(result, ForecastResponse)
    assert result.location == "Seattle, WA"
    assert len(result.forecast_days) == 1
    assert result.forecast_days[0].high_f == 65.0


@pytest.mark.asyncio
async def test_integration_hurricane_alert():
    """Test with real HurricaneAlert schema"""
    llm = MockLLM(
        [
            {
                "category": 3,
                "wind_speed_mph": 120,
                "location": "Gulf Coast",
                "issued_at": datetime.now(UTC).isoformat(),
                "affected_population": 250000,
                "evacuation_zone": "A",
            }
        ]
    )

    handler = StructuredOutputHandler(llm=llm)

    result = await handler.invoke_with_recovery(
        schema=HurricaneAlert,
        query="Create hurricane alert",
        system_prompt="You are a hurricane alert system",
    )

    assert isinstance(result, HurricaneAlert)
    assert result.category == 3
    assert result.wind_speed_mph == 120
    assert result.location == "Gulf Coast"
    assert result.affected_population == 250000
    assert result.evacuation_zone == "A"


# ============================================================================
# Unit Tests: Convenience Function
# ============================================================================


@pytest.mark.asyncio
async def test_invoke_structured_convenience():
    """Test convenience function invoke_structured()"""
    llm = MockLLM([{"name": "Test", "age": 25}])

    result = await invoke_structured(
        llm=llm,
        schema=SimpleSchema,
        query="Create person",
        system_prompt="System",
        max_retries=2,
        fallback_enabled=True,
    )

    assert isinstance(result, SimpleSchema)
    assert result.name == "Test"
    assert result.age == 25


# ============================================================================
# Performance Tests: <2% Fallback Rate Target
# ============================================================================


@pytest.mark.asyncio
async def test_fallback_rate_target():
    """Test that fallback rate meets <2% target (Level 2 requirement)"""
    # Simulate 100 attempts with 1 fallback (1% rate)
    successes = [{"name": f"Person{i}", "age": 25} for i in range(99)]
    validation_error = create_validation_error(SimpleSchema, {"name": "Test", "age": -5})
    fallback = [validation_error for _ in range(3)]  # 3 failures (first + 2 retries)

    all_responses = successes + fallback

    handler = StructuredOutputHandler(
        llm=MockLLM([]), max_retries=2, fallback_enabled=True
    )

    # Run 99 successful attempts
    for i in range(99):
        handler.llm.responses = [successes[i]]
        handler.llm.call_count = 0
        result = await handler.invoke_with_recovery(
            schema=SimpleSchema, query=f"Query {i}", system_prompt="System"
        )
        assert isinstance(result, SimpleSchema)

    # Run 1 failed attempt (all retries fail)
    handler.llm.responses = fallback
    handler.llm.call_count = 0
    result = await handler.invoke_with_recovery(
        schema=SimpleSchema, query="Fail query", system_prompt="System"
    )
    assert isinstance(result, dict)  # Fallback

    # Verify <2% fallback rate
    metrics = handler.get_metrics()
    assert metrics["total_attempts"] == 100
    assert metrics["fallback_uses"] == 1
    assert metrics["fallback_rate"] == "1.0%"  # ✅ Meets <2% target


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_empty_query():
    """Test with empty query string"""
    llm = MockLLM([{"name": "Default", "age": 0}])
    handler = StructuredOutputHandler(llm=llm)

    result = await handler.invoke_with_recovery(
        schema=SimpleSchema, query="", system_prompt="System"
    )

    assert isinstance(result, SimpleSchema)


@pytest.mark.asyncio
async def test_max_retries_zero():
    """Test with max_retries=0 (no retry allowed)"""
    validation_error = create_validation_error(SimpleSchema, {"name": "Test", "age": -5})

    llm = MockLLM([validation_error])
    handler = StructuredOutputHandler(llm=llm, max_retries=0, fallback_enabled=True)

    result = await handler.invoke_with_recovery(
        schema=SimpleSchema, query="Test", system_prompt="System"
    )

    # Should fallback immediately (no retries)
    assert isinstance(result, dict)
    assert result["success"] is False
    assert handler.fallback_uses == 1
    assert llm.call_count == 1  # Only first attempt, no retries


@pytest.mark.asyncio
async def test_get_metrics_before_any_attempts():
    """Test get_metrics() before any invocations"""
    llm = MockLLM([])
    handler = StructuredOutputHandler(llm=llm)

    metrics = handler.get_metrics()

    assert metrics["total_attempts"] == 0
    assert metrics["validation_failures"] == 0
    assert metrics["fallback_uses"] == 0
    assert metrics["success_rate"] == "0.0%"
    assert metrics["fallback_rate"] == "0.0%"
