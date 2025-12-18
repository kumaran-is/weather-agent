"""
Prompt Variation Generation Module.

Level 6b: Self-Improvement Platform

Techniques:
1. Tone adjustment (formal, casual, technical)
2. Structure modification (bullet points, numbered, prose)
3. Example addition/removal
4. Specificity adjustment
5. Role framing changes
6. Instruction ordering

Target: Generate diverse, high-quality prompt variations
"""

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class VariationTechnique(str, Enum):
    """Techniques for generating prompt variations."""

    FORMAL_TONE = "formal_tone"
    CASUAL_TONE = "casual_tone"
    TECHNICAL_TONE = "technical_tone"
    CONCISE = "concise"
    VERBOSE = "verbose"
    STRUCTURED = "structured"
    BULLET_POINTS = "bullet_points"
    NUMBERED_STEPS = "numbered_steps"
    EXAMPLES_ADDED = "examples_added"
    QUESTION_FOCUSED = "question_focused"
    STEP_BY_STEP = "step_by_step"
    CONTEXT_AWARE = "context_aware"
    ROLE_EMPHASIS = "role_emphasis"
    CONSTRAINT_FOCUSED = "constraint_focused"


class GeneratedVariation(BaseModel):
    """A generated prompt variation."""

    technique: VariationTechnique
    prompt_text: str
    description: str = ""
    expected_improvement: str = ""


class PromptVariationGenerator:
    """
    Generate diverse prompt variations using multiple techniques.

    Provides:
    - Heuristic-based transformations
    - LLM-assisted generation
    - Template-based variations
    - Domain-specific optimizations
    """

    # Template transformations for each technique
    TEMPLATES = {
        VariationTechnique.FORMAL_TONE: {
            "prefix": "Please kindly ",
            "suffix": "\n\nThank you for your assistance.",
            "replace": {
                "you are": "you are kindly requested to be",
                "must": "should",
                "!": ".",
            },
        },
        VariationTechnique.CASUAL_TONE: {
            "prefix": "Hey! ",
            "suffix": "\n\nThanks!",
            "replace": {
                "You are": "You're",
                "do not": "don't",
                "cannot": "can't",
            },
        },
        VariationTechnique.TECHNICAL_TONE: {
            "prefix": "Technical specification:\n",
            "suffix": "\n\nOutput format: Structured JSON or technical specification.",
            "replace": {},
        },
        VariationTechnique.CONCISE: {
            "max_words": 50,
            "remove_filler": True,
        },
        VariationTechnique.STRUCTURED: {
            "template": "# Instructions\n{prompt}\n\n# Expected Output\nProvide a clear, structured response.",
        },
        VariationTechnique.BULLET_POINTS: {
            "template": "Key instructions:\n• {prompt}\n• Be accurate\n• Be helpful",
        },
        VariationTechnique.NUMBERED_STEPS: {
            "template": "Follow these steps:\n1. Read the query carefully\n2. {prompt}\n3. Format the response clearly",
        },
        VariationTechnique.EXAMPLES_ADDED: {
            "suffix": "\n\nExample:\nQuery: What's the weather?\nResponse: Current conditions show sunny skies with 75°F.",
        },
        VariationTechnique.STEP_BY_STEP: {
            "prefix": "Think step by step:\n",
            "suffix": "\n\nBreak down your reasoning before providing the final answer.",
        },
        VariationTechnique.CONTEXT_AWARE: {
            "prefix": "Consider the context and user's underlying intent:\n",
            "suffix": "\n\nAdapt your response to the user's apparent needs.",
        },
        VariationTechnique.ROLE_EMPHASIS: {
            "prefix": "As an expert AI assistant specializing in this domain:\n",
            "suffix": "",
        },
        VariationTechnique.CONSTRAINT_FOCUSED: {
            "suffix": "\n\nConstraints:\n- Be factual\n- Acknowledge uncertainty\n- Stay on topic",
        },
    }

    def __init__(self, llm: Any | None = None):
        """
        Initialize variation generator.

        Args:
            llm: Optional LLM for advanced variations
        """
        self.llm = llm
        logger.info("PromptVariationGenerator initialized")

    def generate_variation(
        self,
        base_prompt: str,
        technique: VariationTechnique,
    ) -> GeneratedVariation:
        """
        Generate a single prompt variation using specified technique.

        Args:
            base_prompt: Original prompt
            technique: Technique to apply

        Returns:
            GeneratedVariation with transformed prompt
        """
        template = self.TEMPLATES.get(technique, {})

        if "template" in template:
            # Use template substitution
            prompt_text = template["template"].format(prompt=base_prompt)
        else:
            # Apply transformations
            prompt_text = base_prompt

            # Apply replacements
            for old, new in template.get("replace", {}).items():
                prompt_text = prompt_text.replace(old, new)

            # Apply prefix
            if "prefix" in template:
                prompt_text = template["prefix"] + prompt_text

            # Apply suffix
            if "suffix" in template:
                prompt_text = prompt_text + template["suffix"]

            # Apply max words
            if "max_words" in template:
                words = prompt_text.split()
                if len(words) > template["max_words"]:
                    prompt_text = " ".join(words[: template["max_words"]]) + "..."

        return GeneratedVariation(
            technique=technique,
            prompt_text=prompt_text,
            description=self._get_technique_description(technique),
            expected_improvement=self._get_expected_improvement(technique),
        )

    def generate_all_variations(
        self,
        base_prompt: str,
        techniques: list[VariationTechnique] | None = None,
    ) -> list[GeneratedVariation]:
        """
        Generate variations using all or specified techniques.

        Args:
            base_prompt: Original prompt
            techniques: Optional list of techniques (defaults to all)

        Returns:
            List of GeneratedVariation
        """
        if techniques is None:
            techniques = list(VariationTechnique)

        variations = []
        for technique in techniques:
            try:
                variation = self.generate_variation(base_prompt, technique)
                variations.append(variation)
            except Exception as e:
                logger.warning(f"Failed to generate {technique} variation: {e}")

        logger.info(f"Generated {len(variations)} variations")
        return variations

    async def generate_llm_variation(
        self,
        base_prompt: str,
        technique: VariationTechnique,
    ) -> GeneratedVariation:
        """
        Generate variation using LLM for more sophisticated transformations.

        Args:
            base_prompt: Original prompt
            technique: Technique to apply

        Returns:
            GeneratedVariation with LLM-generated prompt
        """
        if self.llm is None:
            return self.generate_variation(base_prompt, technique)

        try:
            generation_prompt = f"""You are a prompt engineering expert. Transform the following prompt using the "{technique.value}" technique.

Original prompt:
{base_prompt}

Technique description: {self._get_technique_description(technique)}

Generate only the transformed prompt, nothing else:"""

            if hasattr(self.llm, "ainvoke"):
                response = await self.llm.ainvoke(generation_prompt)
                prompt_text = response.content.strip() if hasattr(response, "content") else str(response).strip()
            elif callable(self.llm):
                response = self.llm(generation_prompt)
                prompt_text = str(response).strip()
            else:
                return self.generate_variation(base_prompt, technique)

            return GeneratedVariation(
                technique=technique,
                prompt_text=prompt_text,
                description=self._get_technique_description(technique),
                expected_improvement=self._get_expected_improvement(technique),
            )

        except Exception as e:
            logger.warning(f"LLM variation failed, using heuristic: {e}")
            return self.generate_variation(base_prompt, technique)

    def _get_technique_description(self, technique: VariationTechnique) -> str:
        """Get human-readable description of technique."""
        descriptions = {
            VariationTechnique.FORMAL_TONE: "Uses formal, professional language",
            VariationTechnique.CASUAL_TONE: "Uses friendly, conversational language",
            VariationTechnique.TECHNICAL_TONE: "Uses technical, precise terminology",
            VariationTechnique.CONCISE: "Shorter, more direct instructions",
            VariationTechnique.VERBOSE: "More detailed, comprehensive instructions",
            VariationTechnique.STRUCTURED: "Organized with clear sections",
            VariationTechnique.BULLET_POINTS: "Key points in bullet format",
            VariationTechnique.NUMBERED_STEPS: "Sequential numbered steps",
            VariationTechnique.EXAMPLES_ADDED: "Includes concrete examples",
            VariationTechnique.QUESTION_FOCUSED: "Emphasizes understanding the query",
            VariationTechnique.STEP_BY_STEP: "Encourages step-by-step reasoning",
            VariationTechnique.CONTEXT_AWARE: "Emphasizes contextual understanding",
            VariationTechnique.ROLE_EMPHASIS: "Emphasizes expert role",
            VariationTechnique.CONSTRAINT_FOCUSED: "Highlights constraints and boundaries",
        }
        return descriptions.get(technique, "Custom technique")

    def _get_expected_improvement(self, technique: VariationTechnique) -> str:
        """Get expected improvement from technique."""
        improvements = {
            VariationTechnique.FORMAL_TONE: "Better for professional contexts",
            VariationTechnique.CASUAL_TONE: "Better engagement, more natural",
            VariationTechnique.TECHNICAL_TONE: "More precise responses",
            VariationTechnique.CONCISE: "Faster processing, clearer focus",
            VariationTechnique.VERBOSE: "More comprehensive coverage",
            VariationTechnique.STRUCTURED: "Better organization of responses",
            VariationTechnique.BULLET_POINTS: "Clearer key points",
            VariationTechnique.NUMBERED_STEPS: "More systematic responses",
            VariationTechnique.EXAMPLES_ADDED: "Better understanding of expectations",
            VariationTechnique.QUESTION_FOCUSED: "More relevant responses",
            VariationTechnique.STEP_BY_STEP: "Better reasoning, fewer errors",
            VariationTechnique.CONTEXT_AWARE: "More contextually appropriate",
            VariationTechnique.ROLE_EMPHASIS: "More authoritative responses",
            VariationTechnique.CONSTRAINT_FOCUSED: "More controlled outputs",
        }
        return improvements.get(technique, "Potential improvement")

    def get_recommended_techniques(
        self,
        domain: str = "general",
    ) -> list[VariationTechnique]:
        """
        Get recommended techniques for a domain.

        Args:
            domain: Domain type (weather, technical, customer_service, etc.)

        Returns:
            List of recommended techniques
        """
        recommendations = {
            "weather": [
                VariationTechnique.STRUCTURED,
                VariationTechnique.EXAMPLES_ADDED,
                VariationTechnique.CONSTRAINT_FOCUSED,
                VariationTechnique.STEP_BY_STEP,
            ],
            "technical": [
                VariationTechnique.TECHNICAL_TONE,
                VariationTechnique.STRUCTURED,
                VariationTechnique.NUMBERED_STEPS,
            ],
            "customer_service": [
                VariationTechnique.CASUAL_TONE,
                VariationTechnique.CONTEXT_AWARE,
                VariationTechnique.EXAMPLES_ADDED,
            ],
            "general": [
                VariationTechnique.STRUCTURED,
                VariationTechnique.STEP_BY_STEP,
                VariationTechnique.EXAMPLES_ADDED,
                VariationTechnique.CONTEXT_AWARE,
            ],
        }
        return recommendations.get(domain, recommendations["general"])
