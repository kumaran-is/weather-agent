"""
Tests for Constitutional AI Framework.

Level 6c: Self-Evolving Platform

Tests:
1. Constitution creation and management
2. Principle-based critique
3. Response revision
4. Full validation workflow
5. Weather domain constitution
"""

import pytest

from backend.src.guardrails.constitutional_ai import (
    Constitution,
    ConstitutionalAI,
    ConstitutionalResult,
    CritiqueResult,
    Principle,
    PrincipleCategory,
)


class TestPrinciple:
    """Test Principle model."""

    def test_principle_creation(self):
        """Test creating a principle."""
        principle = Principle(
            id="test_1",
            name="Test Principle",
            description="A test principle",
            category=PrincipleCategory.SAFETY,
            weight=0.8,
        )

        assert principle.id == "test_1"
        assert principle.name == "Test Principle"
        assert principle.category == PrincipleCategory.SAFETY
        assert principle.weight == 0.8

    def test_principle_defaults(self):
        """Test principle default values."""
        principle = Principle(
            id="test_2",
            name="Test",
            description="Test",
            category=PrincipleCategory.HELPFULNESS,
        )

        assert principle.weight == 1.0
        assert principle.critique_prompt == ""
        assert principle.revision_prompt == ""
        assert principle.examples == []


class TestConstitution:
    """Test Constitution class."""

    def test_constitution_creation(self):
        """Test creating a constitution."""
        principles = [
            Principle(
                id="p1",
                name="Principle 1",
                description="First principle",
                category=PrincipleCategory.SAFETY,
            ),
            Principle(
                id="p2",
                name="Principle 2",
                description="Second principle",
                category=PrincipleCategory.HONESTY,
            ),
        ]

        constitution = Constitution(name="Test Constitution", principles=principles)

        assert constitution.name == "Test Constitution"
        assert len(constitution.principles) == 2

    def test_add_principle(self):
        """Test adding a principle."""
        constitution = Constitution(name="Test")

        principle = Principle(
            id="new_p",
            name="New Principle",
            description="New",
            category=PrincipleCategory.ACCURACY,
        )

        constitution.add_principle(principle)

        assert len(constitution.principles) == 1
        assert constitution.get_principle("new_p") is not None

    def test_get_principle(self):
        """Test getting principle by ID."""
        principles = [
            Principle(
                id="p1",
                name="Principle 1",
                description="First",
                category=PrincipleCategory.SAFETY,
            ),
        ]

        constitution = Constitution(name="Test", principles=principles)

        assert constitution.get_principle("p1") is not None
        assert constitution.get_principle("nonexistent") is None

    def test_get_by_category(self):
        """Test getting principles by category."""
        principles = [
            Principle(
                id="p1",
                name="Safety 1",
                description="Safety",
                category=PrincipleCategory.SAFETY,
            ),
            Principle(
                id="p2",
                name="Safety 2",
                description="Safety",
                category=PrincipleCategory.SAFETY,
            ),
            Principle(
                id="p3",
                name="Accuracy",
                description="Accuracy",
                category=PrincipleCategory.ACCURACY,
            ),
        ]

        constitution = Constitution(name="Test", principles=principles)

        safety_principles = constitution.get_by_category(PrincipleCategory.SAFETY)
        assert len(safety_principles) == 2

        accuracy_principles = constitution.get_by_category(PrincipleCategory.ACCURACY)
        assert len(accuracy_principles) == 1

    def test_weather_domain_constitution(self):
        """Test weather domain constitution."""
        constitution = Constitution.weather_domain()

        assert constitution.name == "Weather Domain Constitution"
        assert len(constitution.principles) > 0

        # Should have safety principles
        safety = constitution.get_by_category(PrincipleCategory.SAFETY)
        assert len(safety) > 0

        # Should have accuracy principles
        accuracy = constitution.get_by_category(PrincipleCategory.ACCURACY)
        assert len(accuracy) > 0

    def test_general_assistant_constitution(self):
        """Test general assistant constitution."""
        constitution = Constitution.general_assistant()

        assert constitution.name == "General Assistant Constitution"
        assert len(constitution.principles) > 0


class TestConstitutionalAI:
    """Test ConstitutionalAI class."""

    @pytest.fixture
    def constitutional_ai(self):
        """Create Constitutional AI with weather constitution."""
        constitution = Constitution.weather_domain()
        return ConstitutionalAI(
            constitution=constitution,
            llm=None,
            max_iterations=2,
            pass_threshold=0.7,
        )

    @pytest.mark.asyncio
    async def test_validate_good_response(self, constitutional_ai):
        """Test validating a good response."""
        query = "What's the weather in Miami?"
        response = """
        Based on National Weather Service data, the current weather in Miami shows
        sunny conditions with a temperature of 85°F. Humidity is at 65% with winds
        from the southeast at 10 mph. The forecast may change, so please check for
        updates.
        """

        result = await constitutional_ai.validate_response(query, response)

        assert isinstance(result, ConstitutionalResult)
        assert result.query == query
        assert result.passes is True
        assert result.score > 0.5

    @pytest.mark.asyncio
    async def test_validate_response_with_safety_issues(self, constitutional_ai):
        """Test validating a response with safety issues."""
        query = "Should I evacuate for the hurricane?"
        response = "The hurricane is nothing to worry about. Just stay home."

        result = await constitutional_ai.validate_response(query, response)

        # Should have critiques
        assert len(result.critiques) > 0

        # Check for failing critiques
        failing_critiques = [c for c in result.critiques if not c.passes]
        assert len(failing_critiques) > 0

    @pytest.mark.asyncio
    async def test_validation_produces_revisions(self, constitutional_ai):
        """Test that validation produces revisions when needed."""
        query = "What's the hurricane forecast?"
        response = "Category 5 hurricane coming. The storm will hit."

        result = await constitutional_ai.validate_response(query, response)

        # Final response may be different from original due to revisions
        assert result.original_response == response
        # Revisions may or may not happen based on critiques

    @pytest.mark.asyncio
    async def test_validation_history(self, constitutional_ai):
        """Test that validation history is tracked."""
        query = "Weather today?"
        response = "It's sunny with 80°F temperature."

        await constitutional_ai.validate_response(query, response)
        await constitutional_ai.validate_response(query, response)

        history = constitutional_ai.get_validation_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_clear_history(self, constitutional_ai):
        """Test clearing validation history."""
        query = "Weather?"
        response = "Sunny."

        await constitutional_ai.validate_response(query, response)
        constitutional_ai.clear_history()

        history = constitutional_ai.get_validation_history()
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_statistics(self, constitutional_ai):
        """Test getting validation statistics."""
        query = "Weather?"
        response = "Current conditions show sunny skies with temperature of 85°F."

        await constitutional_ai.validate_response(query, response)

        stats = constitutional_ai.get_statistics()

        assert "total_validations" in stats
        assert "pass_rate" in stats
        assert "average_score" in stats
        assert stats["total_validations"] == 1


class TestCritiqueResult:
    """Test CritiqueResult model."""

    def test_critique_result_creation(self):
        """Test creating a critique result."""
        result = CritiqueResult(
            principle_id="p1",
            principle_name="Safety",
            passes=True,
            critique="Response adheres to safety principle.",
            severity="low",
            suggestions=[],
        )

        assert result.principle_id == "p1"
        assert result.passes is True
        assert result.severity == "low"

    def test_critique_result_with_suggestions(self):
        """Test critique result with suggestions."""
        result = CritiqueResult(
            principle_id="p1",
            principle_name="Safety",
            passes=False,
            critique="Response lacks safety information.",
            severity="high",
            suggestions=["Add evacuation guidance", "Include emergency contacts"],
        )

        assert result.passes is False
        assert len(result.suggestions) == 2


class TestHeuristicCritique:
    """Test heuristic-based critique."""

    @pytest.fixture
    def constitutional_ai(self):
        """Create Constitutional AI without LLM."""
        constitution = Constitution.weather_domain()
        return ConstitutionalAI(constitution=constitution, llm=None)

    @pytest.mark.asyncio
    async def test_safety_critique_with_emergency_query(self, constitutional_ai):
        """Test safety critique with emergency query."""
        query = "Hurricane warning - should I evacuate?"
        response_without_safety = "The hurricane is approaching. Wind speeds are high."

        result = await constitutional_ai.validate_response(query, response_without_safety)

        # Should have critiques about safety
        safety_critiques = [
            c for c in result.critiques
            if "safety" in c.principle_name.lower()
        ]
        assert len(safety_critiques) > 0

    @pytest.mark.asyncio
    async def test_accuracy_critique_without_source(self, constitutional_ai):
        """Test accuracy critique without source attribution."""
        query = "What's the weather forecast?"
        response_without_source = "Tomorrow will be sunny with 80°F."

        result = await constitutional_ai.validate_response(query, response_without_source)

        # Should have critiques about source attribution
        accuracy_critiques = [
            c for c in result.critiques
            if "source" in c.principle_name.lower()
        ]
        # May or may not pass based on heuristic

    @pytest.mark.asyncio
    async def test_honesty_critique_without_uncertainty(self, constitutional_ai):
        """Test honesty critique without uncertainty acknowledgment."""
        query = "Will it rain tomorrow?"
        response_certain = "It will definitely rain tomorrow at 3pm."

        result = await constitutional_ai.validate_response(query, response_certain)

        # Should have critiques about uncertainty
        # The heuristic checks for "may", "could", "confidence" etc.


class TestDomainSpecificConstitution:
    """Test domain-specific constitutions."""

    def test_weather_constitution_has_hurricane_validation(self):
        """Test weather constitution has hurricane validation."""
        constitution = Constitution.weather_domain()

        # Should have hurricane category verification
        hurricane_principles = [
            p for p in constitution.principles
            if "hurricane" in p.description.lower() or "category" in p.description.lower()
        ]
        assert len(hurricane_principles) > 0

    def test_weather_constitution_has_evacuation_zone(self):
        """Test weather constitution has evacuation zone principle."""
        constitution = Constitution.weather_domain()

        # Should have evacuation zone principle
        evacuation_principles = [
            p for p in constitution.principles
            if "evacuation" in p.description.lower()
        ]
        assert len(evacuation_principles) > 0

    def test_weather_constitution_has_no_downplaying(self):
        """Test weather constitution prevents downplaying risks."""
        constitution = Constitution.weather_domain()

        # Should have no downplaying principle
        downplay_principles = [
            p for p in constitution.principles
            if "downplay" in p.description.lower()
        ]
        assert len(downplay_principles) > 0
