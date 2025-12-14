"""
Tests for Prompt Variation Generation.

Level 6b: Self-Improvement Platform

Tests:
1. Single variation generation
2. All variations generation
3. Template-based transformations
4. LLM-assisted variations
5. Domain-specific recommendations
"""

import pytest
from backend.src.prompts.prompt_variations import (
    PromptVariationGenerator,
    VariationTechnique,
    GeneratedVariation,
)


class TestVariationTechnique:
    """Test VariationTechnique enum."""

    def test_all_techniques_exist(self):
        """Test that all expected techniques exist."""
        expected = [
            "FORMAL_TONE",
            "CASUAL_TONE",
            "TECHNICAL_TONE",
            "CONCISE",
            "VERBOSE",
            "STRUCTURED",
            "BULLET_POINTS",
            "NUMBERED_STEPS",
            "EXAMPLES_ADDED",
            "QUESTION_FOCUSED",
            "STEP_BY_STEP",
            "CONTEXT_AWARE",
            "ROLE_EMPHASIS",
            "CONSTRAINT_FOCUSED",
        ]

        for technique in expected:
            assert hasattr(VariationTechnique, technique)

    def test_technique_values(self):
        """Test that technique values are snake_case."""
        for technique in VariationTechnique:
            assert technique.value == technique.name.lower()


class TestPromptVariationGenerator:
    """Test suite for PromptVariationGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator without LLM."""
        return PromptVariationGenerator(llm=None)

    @pytest.fixture
    def base_prompt(self):
        """Sample base prompt for testing."""
        return "You are a helpful weather assistant. Answer questions about weather conditions."

    def test_generator_initialization(self, generator):
        """Test generator initializes correctly."""
        assert generator.llm is None
        assert generator.TEMPLATES is not None
        assert len(generator.TEMPLATES) > 0

    def test_generate_formal_variation(self, generator, base_prompt):
        """Test formal tone variation."""
        variation = generator.generate_variation(
            base_prompt,
            VariationTechnique.FORMAL_TONE,
        )

        assert isinstance(variation, GeneratedVariation)
        assert variation.technique == VariationTechnique.FORMAL_TONE
        assert "Please kindly" in variation.prompt_text or "Thank you" in variation.prompt_text
        assert variation.description != ""
        assert variation.expected_improvement != ""

    def test_generate_casual_variation(self, generator, base_prompt):
        """Test casual tone variation."""
        variation = generator.generate_variation(
            base_prompt,
            VariationTechnique.CASUAL_TONE,
        )

        assert variation.technique == VariationTechnique.CASUAL_TONE
        assert "Hey!" in variation.prompt_text or "Thanks!" in variation.prompt_text

    def test_generate_technical_variation(self, generator, base_prompt):
        """Test technical tone variation."""
        variation = generator.generate_variation(
            base_prompt,
            VariationTechnique.TECHNICAL_TONE,
        )

        assert variation.technique == VariationTechnique.TECHNICAL_TONE
        assert "Technical specification" in variation.prompt_text

    def test_generate_concise_variation(self, generator):
        """Test concise variation with truncation."""
        long_prompt = " ".join(["word"] * 100)  # 100 words

        variation = generator.generate_variation(
            long_prompt,
            VariationTechnique.CONCISE,
        )

        assert variation.technique == VariationTechnique.CONCISE
        # Should be truncated to max_words
        words = variation.prompt_text.split()
        assert len(words) <= 51  # max_words + "..."

    def test_generate_structured_variation(self, generator, base_prompt):
        """Test structured variation."""
        variation = generator.generate_variation(
            base_prompt,
            VariationTechnique.STRUCTURED,
        )

        assert "# Instructions" in variation.prompt_text
        assert "# Expected Output" in variation.prompt_text

    def test_generate_bullet_points_variation(self, generator, base_prompt):
        """Test bullet points variation."""
        variation = generator.generate_variation(
            base_prompt,
            VariationTechnique.BULLET_POINTS,
        )

        assert "•" in variation.prompt_text
        assert "Key instructions" in variation.prompt_text

    def test_generate_numbered_steps_variation(self, generator, base_prompt):
        """Test numbered steps variation."""
        variation = generator.generate_variation(
            base_prompt,
            VariationTechnique.NUMBERED_STEPS,
        )

        assert "1." in variation.prompt_text
        assert "2." in variation.prompt_text
        assert "Follow these steps" in variation.prompt_text

    def test_generate_examples_variation(self, generator, base_prompt):
        """Test examples added variation."""
        variation = generator.generate_variation(
            base_prompt,
            VariationTechnique.EXAMPLES_ADDED,
        )

        assert "Example:" in variation.prompt_text
        assert "Query:" in variation.prompt_text
        assert "Response:" in variation.prompt_text

    def test_generate_step_by_step_variation(self, generator, base_prompt):
        """Test step-by-step variation."""
        variation = generator.generate_variation(
            base_prompt,
            VariationTechnique.STEP_BY_STEP,
        )

        assert "Think step by step" in variation.prompt_text
        assert "reasoning" in variation.prompt_text.lower()

    def test_generate_context_aware_variation(self, generator, base_prompt):
        """Test context-aware variation."""
        variation = generator.generate_variation(
            base_prompt,
            VariationTechnique.CONTEXT_AWARE,
        )

        assert "context" in variation.prompt_text.lower()
        assert "intent" in variation.prompt_text.lower()

    def test_generate_role_emphasis_variation(self, generator, base_prompt):
        """Test role emphasis variation."""
        variation = generator.generate_variation(
            base_prompt,
            VariationTechnique.ROLE_EMPHASIS,
        )

        assert "expert" in variation.prompt_text.lower()

    def test_generate_constraint_focused_variation(self, generator, base_prompt):
        """Test constraint-focused variation."""
        variation = generator.generate_variation(
            base_prompt,
            VariationTechnique.CONSTRAINT_FOCUSED,
        )

        assert "Constraints:" in variation.prompt_text
        assert "factual" in variation.prompt_text.lower()

    def test_generate_all_variations(self, generator, base_prompt):
        """Test generating all variations."""
        variations = generator.generate_all_variations(base_prompt)

        # Should generate one for each technique
        assert len(variations) == len(VariationTechnique)

        # Each should be a GeneratedVariation
        for variation in variations:
            assert isinstance(variation, GeneratedVariation)

    def test_generate_selected_variations(self, generator, base_prompt):
        """Test generating selected variations."""
        techniques = [
            VariationTechnique.FORMAL_TONE,
            VariationTechnique.STEP_BY_STEP,
        ]

        variations = generator.generate_all_variations(
            base_prompt,
            techniques=techniques,
        )

        assert len(variations) == 2
        assert variations[0].technique == VariationTechnique.FORMAL_TONE
        assert variations[1].technique == VariationTechnique.STEP_BY_STEP

    def test_technique_descriptions(self, generator):
        """Test that all techniques have descriptions."""
        for technique in VariationTechnique:
            desc = generator._get_technique_description(technique)
            assert desc != ""
            assert desc != "Custom technique"

    def test_technique_improvements(self, generator):
        """Test that all techniques have expected improvements."""
        for technique in VariationTechnique:
            improvement = generator._get_expected_improvement(technique)
            assert improvement != ""
            assert improvement != "Potential improvement"


class TestDomainRecommendations:
    """Test domain-specific recommendations."""

    @pytest.fixture
    def generator(self):
        return PromptVariationGenerator()

    def test_weather_domain_recommendations(self, generator):
        """Test weather domain recommendations."""
        techniques = generator.get_recommended_techniques("weather")

        assert VariationTechnique.STRUCTURED in techniques
        assert VariationTechnique.EXAMPLES_ADDED in techniques
        assert VariationTechnique.CONSTRAINT_FOCUSED in techniques

    def test_technical_domain_recommendations(self, generator):
        """Test technical domain recommendations."""
        techniques = generator.get_recommended_techniques("technical")

        assert VariationTechnique.TECHNICAL_TONE in techniques
        assert VariationTechnique.STRUCTURED in techniques

    def test_customer_service_recommendations(self, generator):
        """Test customer service domain recommendations."""
        techniques = generator.get_recommended_techniques("customer_service")

        assert VariationTechnique.CASUAL_TONE in techniques
        assert VariationTechnique.CONTEXT_AWARE in techniques

    def test_general_domain_recommendations(self, generator):
        """Test general domain recommendations."""
        techniques = generator.get_recommended_techniques("general")

        assert len(techniques) >= 3
        assert VariationTechnique.STRUCTURED in techniques

    def test_unknown_domain_falls_back(self, generator):
        """Test unknown domain falls back to general."""
        techniques = generator.get_recommended_techniques("unknown_domain")
        general = generator.get_recommended_techniques("general")

        assert techniques == general


class TestLLMVariation:
    """Test LLM-assisted variation generation."""

    @pytest.fixture
    def generator_no_llm(self):
        return PromptVariationGenerator(llm=None)

    @pytest.mark.asyncio
    async def test_llm_variation_fallback(self, generator_no_llm):
        """Test that LLM variation falls back when no LLM."""
        base_prompt = "You are a helpful assistant."

        variation = await generator_no_llm.generate_llm_variation(
            base_prompt,
            VariationTechnique.FORMAL_TONE,
        )

        # Should fall back to heuristic
        assert isinstance(variation, GeneratedVariation)
        assert variation.technique == VariationTechnique.FORMAL_TONE

    @pytest.mark.asyncio
    async def test_llm_variation_with_mock_llm(self):
        """Test LLM variation with mock LLM."""

        class MockLLM:
            async def ainvoke(self, prompt: str) -> object:
                class Response:
                    content = "This is a transformed prompt using LLM."

                return Response()

        generator = PromptVariationGenerator(llm=MockLLM())

        variation = await generator.generate_llm_variation(
            "Original prompt",
            VariationTechnique.FORMAL_TONE,
        )

        assert variation.prompt_text == "This is a transformed prompt using LLM."

    @pytest.mark.asyncio
    async def test_llm_variation_error_fallback(self):
        """Test LLM variation falls back on error."""

        class FailingLLM:
            async def ainvoke(self, prompt: str):
                raise Exception("LLM error")

        generator = PromptVariationGenerator(llm=FailingLLM())

        variation = await generator.generate_llm_variation(
            "Original prompt",
            VariationTechnique.FORMAL_TONE,
        )

        # Should fall back to heuristic
        assert isinstance(variation, GeneratedVariation)
        assert "Please kindly" in variation.prompt_text or "Thank you" in variation.prompt_text
