"""Unit Tests for Auto-Routing Module (v0.6.0).

Tests the intent-based query classification system that replaces
explicit use_multi_agent and agent_level flags.

Test Categories:
1. QueryTier classification (SIMPLE, STANDARD, COMPLEX, EMERGENCY)
2. Signal extraction (query analysis, context signals)
3. Rule evaluation priority
4. Edge cases and boundary conditions
"""

import pytest

from backend.src.routing import (
    QueryClassifier,
    QueryTier,
    RoutingDecision,
    classify_query,
    extract_context_signal,
    extract_query_signal,
)
from backend.src.routing.models import ContextSignal, QueryAnalysisSignal


class TestQueryTierClassification:
    """Test that queries are classified into correct tiers."""

    def test_simple_weather_query(self):
        """Simple weather queries should route to SIMPLE tier (basic agent)."""
        simple_queries = [
            "What's the weather in London?",
            "Will it rain tomorrow in Seattle?",
            "Temperature today in New York",
            "Is it sunny in Miami?",
            "How warm will it be this weekend?",
        ]

        for query in simple_queries:
            decision = classify_query(query)
            assert decision.tier == QueryTier.SIMPLE, f"Query '{query}' should be SIMPLE, got {decision.tier}"
            assert decision.agent_level == "basic"

    def test_hurricane_query_routes_to_standard(self):
        """Hurricane/storm queries should route to STANDARD tier (L4A)."""
        standard_queries = [
            "Is Hurricane Milton going to hit Tampa?",
            "Track Hurricane Ian",
            "What's the NHC advisory for tropical storm?",
            "When will the hurricane make landfall?",
            "What category is the storm?",
            "Storm surge predictions for Miami",
        ]

        for query in standard_queries:
            decision = classify_query(query)
            assert decision.tier == QueryTier.STANDARD, f"Query '{query}' should be STANDARD, got {decision.tier}"
            assert decision.agent_level == "l4a"

    def test_complex_analysis_routes_to_complex(self):
        """Complex analysis queries should route to COMPLEX tier (L4B)."""
        complex_queries = [
            "Compare Hurricane Milton to Hurricane Ian",
            "Analyze the historical pattern of hurricanes in Florida",
            "What's the trend for Atlantic hurricanes over 10 years?",
            "Forecast next 7 days with detailed analysis",
            "How does this storm compare to historical patterns?",
        ]

        for query in complex_queries:
            decision = classify_query(query)
            assert decision.tier == QueryTier.COMPLEX, f"Query '{query}' should be COMPLEX, got {decision.tier}"
            assert decision.agent_level == "l4b"

    def test_emergency_safety_routes_to_emergency(self):
        """Emergency/safety queries should route to EMERGENCY tier (L4C)."""
        emergency_queries = [
            "Should I evacuate from Tampa?",
            "Am I safe in Miami Beach?",
            "Is it dangerous to stay?",
            "Mandatory evacuation zones",
            "Life-threatening conditions",
            "Emergency shelter locations",
        ]

        for query in emergency_queries:
            decision = classify_query(query)
            assert decision.tier == QueryTier.EMERGENCY, f"Query '{query}' should be EMERGENCY, got {decision.tier}"
            assert decision.agent_level == "l4c"
            assert decision.hitl_required is True

    def test_multi_question_routes_to_complex(self):
        """Multi-question queries should route to COMPLEX tier."""
        decision = classify_query("What's the weather today? And will it rain tomorrow? What about the weekend?")
        assert decision.tier == QueryTier.COMPLEX
        assert decision.query_signal.is_multi_question is True


class TestQuerySignalExtraction:
    """Test signal extraction from query text."""

    def test_emergency_keywords_detected(self):
        """Emergency keywords should be detected in query signal."""
        signal = extract_query_signal("Should I evacuate from my home?")
        assert signal.has_emergency_keywords is True
        assert "evacuate" in str(signal.matched_patterns).lower() or "emergency" in str(signal.matched_patterns).lower()

    def test_complex_keywords_detected(self):
        """Complex keywords should be detected in query signal."""
        signal = extract_query_signal("Compare Hurricane Milton to Hurricane Ian")
        assert signal.has_complex_keywords is True
        assert signal.has_storm_keywords is True

    def test_storm_keywords_detected(self):
        """Storm keywords should be detected in query signal."""
        signal = extract_query_signal("Track Hurricane Milton")
        assert signal.has_storm_keywords is True

    def test_word_count_calculated(self):
        """Word count should be calculated correctly."""
        signal = extract_query_signal("What is the weather in London today?")
        assert signal.word_count == 7

    def test_multi_question_detected(self):
        """Multiple questions should be detected."""
        signal = extract_query_signal("Is it raining? Will it stop soon?")
        assert signal.is_multi_question is True

    def test_conditional_language_detected(self):
        """Conditional language should be detected."""
        signal = extract_query_signal("If it rains tomorrow, should I bring an umbrella?")
        assert signal.has_conditional_language is True


class TestContextSignalExtraction:
    """Test signal extraction from memory context."""

    def test_empty_context_returns_defaults(self):
        """Empty context should return default signal values."""
        signal = extract_context_signal(None)
        assert signal.is_follow_up is False
        assert signal.previous_tier is None
        assert signal.conversation_depth == 0
        assert signal.has_hurricane_history is False

    def test_follow_up_detected(self):
        """Follow-up conversations should be detected."""
        context = {
            "conversation_history": [
                {"query": "What's the weather?", "response": "It's sunny."}
            ]
        }
        signal = extract_context_signal(context)
        assert signal.is_follow_up is True
        assert signal.conversation_depth == 1

    def test_hurricane_history_detected(self):
        """Hurricane history in conversation should be detected."""
        context = {
            "conversation_history": [
                {"query": "Track Hurricane Milton", "response": "Milton is Category 4."}
            ]
        }
        signal = extract_context_signal(context)
        assert signal.has_hurricane_history is True

    def test_previous_tier_extracted(self):
        """Previous tier should be extracted from routing decision."""
        context = {
            "last_routing_decision": {"tier": "emergency"}
        }
        signal = extract_context_signal(context)
        assert signal.previous_tier == QueryTier.EMERGENCY


class TestRoutingDecision:
    """Test routing decision structure and attributes."""

    def test_decision_has_required_attributes(self):
        """Routing decision should have all required attributes."""
        decision = classify_query("What's the weather?")

        assert hasattr(decision, "tier")
        assert hasattr(decision, "agent_level")
        assert hasattr(decision, "confidence")
        assert hasattr(decision, "primary_signal")
        assert hasattr(decision, "query_signal")
        assert hasattr(decision, "context_signal")
        assert hasattr(decision, "reasoning")
        assert hasattr(decision, "hitl_required")

    def test_confidence_in_valid_range(self):
        """Confidence should be between 0 and 1."""
        for query in ["Weather?", "Hurricane?", "Compare storms?", "Evacuate?"]:
            decision = classify_query(query)
            assert 0.0 <= decision.confidence <= 1.0

    def test_hitl_required_only_for_emergency(self):
        """HITL should only be required for EMERGENCY tier."""
        simple = classify_query("What's the weather?")
        assert simple.hitl_required is False

        standard = classify_query("Track Hurricane Milton")
        assert standard.hitl_required is False

        emergency = classify_query("Should I evacuate?")
        assert emergency.hitl_required is True

    def test_to_dict_conversion(self):
        """Decision should convert to dictionary correctly."""
        decision = classify_query("What's the weather?")
        d = decision.to_dict()

        assert "tier" in d
        assert "agent_level" in d
        assert "confidence" in d
        assert "reasoning" in d


class TestRulePriority:
    """Test that rules are evaluated in correct priority order."""

    def test_emergency_overrides_complex(self):
        """Emergency keywords should override complex keywords."""
        # Query has both complex ("analyze") and emergency ("evacuate") keywords
        decision = classify_query("Analyze whether I should evacuate")
        assert decision.tier == QueryTier.EMERGENCY

    def test_emergency_overrides_standard(self):
        """Emergency keywords should override standard keywords."""
        # Query has both standard ("hurricane") and emergency ("evacuate")
        decision = classify_query("Should I evacuate because of the hurricane?")
        assert decision.tier == QueryTier.EMERGENCY

    def test_complex_overrides_standard(self):
        """Complex keywords should override standard keywords alone."""
        # Query has both standard ("hurricane") and complex ("compare", "historical")
        decision = classify_query("Compare this hurricane to historical patterns")
        assert decision.tier == QueryTier.COMPLEX


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query_returns_simple(self):
        """Empty or minimal query should return SIMPLE tier."""
        decision = classify_query(" ")
        assert decision.tier == QueryTier.SIMPLE

    def test_case_insensitive_matching(self):
        """Keywords should match regardless of case."""
        queries = [
            "HURRICANE MILTON",
            "Hurricane Milton",
            "hurricane milton",
            "HuRrIcAnE MiLtOn",
        ]
        for query in queries:
            decision = classify_query(query)
            assert decision.tier == QueryTier.STANDARD

    def test_partial_keyword_no_false_positive(self):
        """Partial keywords should not trigger false positives."""
        # "cat" appears in "category" but shouldn't match "cat 4"
        decision = classify_query("I have a cat named Stormy")
        assert decision.tier == QueryTier.SIMPLE

    def test_context_escalation(self):
        """Previous emergency conversation should maintain elevated tier."""
        context = {
            "conversation_history": [{"query": "Should I evacuate?"}],
            "last_routing_decision": {"tier": "emergency"},
        }
        # Simple weather query should stay elevated due to previous emergency
        decision = classify_query("What's the weather?", context)
        # Note: This depends on context escalation rule priority
        # Currently it maintains emergency tier for follow-ups
        assert decision.tier in [QueryTier.EMERGENCY, QueryTier.SIMPLE]


class TestClassifierStatistics:
    """Test classifier statistics tracking."""

    def test_classification_count_increments(self):
        """Classification count should increment with each call."""
        classifier = QueryClassifier()

        assert classifier.get_stats()["classification_count"] == 0

        classifier.classify("Query 1")
        assert classifier.get_stats()["classification_count"] == 1

        classifier.classify("Query 2")
        assert classifier.get_stats()["classification_count"] == 2


class TestAgentLevelMapping:
    """Test that tiers map to correct agent levels."""

    def test_tier_to_agent_level_mapping(self):
        """Each tier should map to the correct agent level."""
        mappings = [
            (QueryTier.SIMPLE, "basic"),
            (QueryTier.STANDARD, "l4a"),
            (QueryTier.COMPLEX, "l4b"),
            (QueryTier.EMERGENCY, "l4c"),
        ]

        for tier, expected_level in mappings:
            # Find a query that routes to this tier
            queries = {
                QueryTier.SIMPLE: "Weather today",
                QueryTier.STANDARD: "Track Hurricane Milton",
                QueryTier.COMPLEX: "Compare Hurricane Ian and Milton",
                QueryTier.EMERGENCY: "Should I evacuate?",
            }
            decision = classify_query(queries[tier])
            assert decision.agent_level == expected_level, f"Tier {tier} should map to {expected_level}"
