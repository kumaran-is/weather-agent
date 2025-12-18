"""
Constitutional AI Framework.

Level 6c: Self-Evolving Platform

Implements Anthropic's Constitutional AI approach:
1. Define principles (constitution)
2. Self-critique responses against principles
3. Revise responses to align with principles
4. Validate final output

Key Features:
- Domain-specific constitutions (weather, safety, general)
- Principle-based critique and revision
- Harm reduction through iterative refinement
- Audit trail for compliance

Target: 99%+ principle adherence, zero harmful outputs
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PrincipleCategory(str, Enum):
    """Categories of constitutional principles."""

    SAFETY = "safety"
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    HONESTY = "honesty"
    HARMLESSNESS = "harmlessness"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    DOMAIN_SPECIFIC = "domain_specific"


class Principle(BaseModel):
    """A single constitutional principle."""

    id: str
    name: str
    description: str
    category: PrincipleCategory
    weight: float = Field(ge=0.0, le=1.0, default=1.0)
    critique_prompt: str = ""
    revision_prompt: str = ""
    examples: list[dict[str, str]] = Field(default_factory=list)


class CritiqueResult(BaseModel):
    """Result of critiquing response against a principle."""

    principle_id: str
    principle_name: str
    passes: bool
    critique: str
    severity: str = "low"  # low, medium, high, critical
    suggestions: list[str] = Field(default_factory=list)


class RevisionResult(BaseModel):
    """Result of revising response based on critique."""

    original_response: str
    revised_response: str
    revisions_made: list[str] = Field(default_factory=list)
    principles_applied: list[str] = Field(default_factory=list)


class ConstitutionalResult(BaseModel):
    """Complete result of constitutional AI validation."""

    query: str
    original_response: str
    final_response: str
    passes: bool
    score: float = Field(ge=0.0, le=1.0)
    critiques: list[CritiqueResult] = Field(default_factory=list)
    revisions: list[RevisionResult] = Field(default_factory=list)
    iterations: int = 0
    timestamp: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class Constitution:
    """
    A collection of principles that define acceptable AI behavior.

    Provides pre-defined constitutions for different domains
    and allows custom principle definition.
    """

    def __init__(self, name: str, principles: list[Principle] | None = None):
        """
        Initialize a constitution.

        Args:
            name: Constitution name
            principles: Optional list of principles
        """
        self.name = name
        self.principles = principles or []
        self._principles_by_id: dict[str, Principle] = {}
        self._principles_by_category: dict[PrincipleCategory, list[Principle]] = {}

        for p in self.principles:
            self._principles_by_id[p.id] = p
            if p.category not in self._principles_by_category:
                self._principles_by_category[p.category] = []
            self._principles_by_category[p.category].append(p)

        logger.info(f"Constitution '{name}' initialized with {len(self.principles)} principles")

    def add_principle(self, principle: Principle) -> None:
        """Add a principle to the constitution."""
        self.principles.append(principle)
        self._principles_by_id[principle.id] = principle
        if principle.category not in self._principles_by_category:
            self._principles_by_category[principle.category] = []
        self._principles_by_category[principle.category].append(principle)

    def get_principle(self, principle_id: str) -> Principle | None:
        """Get principle by ID."""
        return self._principles_by_id.get(principle_id)

    def get_by_category(self, category: PrincipleCategory) -> list[Principle]:
        """Get principles by category."""
        return self._principles_by_category.get(category, [])

    @classmethod
    def weather_domain(cls) -> "Constitution":
        """Create a constitution for weather domain."""
        principles = [
            # Safety principles
            Principle(
                id="weather_safety_1",
                name="Life Safety Priority",
                description="Always prioritize human life and safety in weather advisories",
                category=PrincipleCategory.SAFETY,
                weight=1.0,
                critique_prompt="Does this response appropriately prioritize human safety? Are evacuation recommendations clear when needed?",
                revision_prompt="Revise to ensure safety is the top priority. Add clear evacuation guidance if severe weather is mentioned.",
            ),
            Principle(
                id="weather_safety_2",
                name="Emergency Contact Information",
                description="Include emergency contacts and resources for severe weather",
                category=PrincipleCategory.SAFETY,
                weight=0.9,
                critique_prompt="Does this response include relevant emergency resources for severe weather situations?",
                revision_prompt="Add relevant emergency contacts (NWS, local emergency management) if discussing severe weather.",
            ),
            Principle(
                id="weather_safety_3",
                name="Evacuation Zone Accuracy",
                description="Provide accurate evacuation zone information using official designations",
                category=PrincipleCategory.SAFETY,
                weight=1.0,
                critique_prompt="Are evacuation zones referenced correctly (e.g., Zone A, Zone B)? Are they based on official designations?",
                revision_prompt="Use official evacuation zone designations (A, B, C) and clarify that users should verify with local authorities.",
            ),
            # Accuracy principles
            Principle(
                id="weather_accuracy_1",
                name="Source Attribution",
                description="Attribute weather data to authoritative sources (NWS, NHC)",
                category=PrincipleCategory.ACCURACY,
                weight=0.9,
                critique_prompt="Does this response cite authoritative weather sources? Is data attributed correctly?",
                revision_prompt="Add source attribution for weather data (NWS, NHC, or local meteorological authority).",
            ),
            Principle(
                id="weather_accuracy_2",
                name="Hurricane Category Verification",
                description="Verify hurricane categories match Saffir-Simpson scale wind speeds",
                category=PrincipleCategory.ACCURACY,
                weight=1.0,
                critique_prompt="If hurricane categories are mentioned, do they match the correct Saffir-Simpson wind speeds? (Cat 1: 74-95 mph, Cat 2: 96-110 mph, Cat 3: 111-129 mph, Cat 4: 130-156 mph, Cat 5: 157+ mph)",
                revision_prompt="Correct any hurricane category mismatches. Ensure wind speeds align with Saffir-Simpson scale.",
            ),
            Principle(
                id="weather_accuracy_3",
                name="Time Zone Clarity",
                description="Specify time zones for all time-sensitive information",
                category=PrincipleCategory.ACCURACY,
                weight=0.8,
                critique_prompt="Are time references clear and include time zones? Are they specific rather than vague?",
                revision_prompt="Add explicit time zones (ET, CT, UTC) to all time references. Replace 'soon' or 'later' with specific times.",
            ),
            # Helpfulness principles
            Principle(
                id="weather_helpful_1",
                name="Actionable Recommendations",
                description="Provide clear, actionable recommendations",
                category=PrincipleCategory.HELPFULNESS,
                weight=0.8,
                critique_prompt="Does this response provide actionable recommendations that users can follow?",
                revision_prompt="Add specific actionable steps the user can take based on the weather conditions.",
            ),
            Principle(
                id="weather_helpful_2",
                name="Local Context",
                description="Consider local context and geography",
                category=PrincipleCategory.HELPFULNESS,
                weight=0.7,
                critique_prompt="Does this response consider local geographic and contextual factors?",
                revision_prompt="Add relevant local context (coastal vs inland, elevation, flood zones) if applicable.",
            ),
            # Honesty principles
            Principle(
                id="weather_honesty_1",
                name="Uncertainty Acknowledgment",
                description="Acknowledge forecast uncertainty appropriately",
                category=PrincipleCategory.HONESTY,
                weight=0.8,
                critique_prompt="Does this response appropriately communicate forecast uncertainty? Are confidence levels mentioned?",
                revision_prompt="Add uncertainty acknowledgment (e.g., 'forecast confidence is moderate' or 'conditions may change').",
            ),
            Principle(
                id="weather_honesty_2",
                name="Data Limitations",
                description="Disclose data limitations or gaps",
                category=PrincipleCategory.HONESTY,
                weight=0.7,
                critique_prompt="Does this response acknowledge any data limitations or gaps in coverage?",
                revision_prompt="Add disclosure of any data limitations if relevant to the query.",
            ),
            # Harmlessness principles
            Principle(
                id="weather_harmless_1",
                name="No Downplaying Risks",
                description="Never downplay severe weather risks",
                category=PrincipleCategory.HARMLESSNESS,
                weight=1.0,
                critique_prompt="Does this response downplay or minimize severe weather risks? Is the tone appropriately serious for dangerous conditions?",
                revision_prompt="Ensure severe weather risks are not minimized. Use appropriately urgent language for dangerous conditions.",
            ),
        ]

        return cls(name="Weather Domain Constitution", principles=principles)

    @classmethod
    def general_assistant(cls) -> "Constitution":
        """Create a general-purpose constitution."""
        principles = [
            Principle(
                id="general_helpful_1",
                name="Be Helpful",
                description="Provide helpful, relevant responses to user queries",
                category=PrincipleCategory.HELPFULNESS,
                weight=0.9,
                critique_prompt="Is this response helpful and relevant to the user's query?",
                revision_prompt="Revise to be more helpful and directly address the user's needs.",
            ),
            Principle(
                id="general_honest_1",
                name="Be Honest",
                description="Be truthful and avoid deception",
                category=PrincipleCategory.HONESTY,
                weight=1.0,
                critique_prompt="Is this response truthful? Does it avoid misleading the user?",
                revision_prompt="Correct any inaccuracies. Be transparent about limitations.",
            ),
            Principle(
                id="general_harmless_1",
                name="Be Harmless",
                description="Avoid causing harm through responses",
                category=PrincipleCategory.HARMLESSNESS,
                weight=1.0,
                critique_prompt="Could this response cause harm to the user or others?",
                revision_prompt="Remove or modify any potentially harmful content.",
            ),
            Principle(
                id="general_privacy_1",
                name="Protect Privacy",
                description="Do not expose or request sensitive personal information",
                category=PrincipleCategory.PRIVACY,
                weight=1.0,
                critique_prompt="Does this response protect user privacy? Does it avoid requesting unnecessary personal information?",
                revision_prompt="Remove any exposure of sensitive information. Avoid requesting unnecessary personal data.",
            ),
            Principle(
                id="general_fair_1",
                name="Be Fair",
                description="Treat all users fairly without bias",
                category=PrincipleCategory.FAIRNESS,
                weight=0.9,
                critique_prompt="Is this response fair and unbiased? Does it avoid discriminatory language?",
                revision_prompt="Remove any biased or discriminatory content. Ensure fair treatment.",
            ),
        ]

        return cls(name="General Assistant Constitution", principles=principles)


class ConstitutionalAI:
    """
    Constitutional AI implementation for response validation and refinement.

    Uses critique-revision cycles to align responses with constitutional principles.
    """

    def __init__(
        self,
        constitution: Constitution,
        llm: Any | None = None,
        max_iterations: int = 3,
        pass_threshold: float = 0.8,
    ):
        """
        Initialize Constitutional AI.

        Args:
            constitution: Constitution defining principles
            llm: Optional LLM for critique and revision
            max_iterations: Maximum revision iterations
            pass_threshold: Score threshold to pass (0-1)
        """
        self.constitution = constitution
        self.llm = llm
        self.max_iterations = max_iterations
        self.pass_threshold = pass_threshold

        # History
        self.validation_history: list[ConstitutionalResult] = []

        logger.info(
            f"ConstitutionalAI initialized | constitution={constitution.name} | "
            f"principles={len(constitution.principles)} | max_iterations={max_iterations}"
        )

    async def validate_response(
        self,
        query: str,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> ConstitutionalResult:
        """
        Validate and refine response against constitutional principles.

        Args:
            query: Original user query
            response: AI response to validate
            context: Optional context (user info, session data, etc.)

        Returns:
            ConstitutionalResult with validation status and any revisions
        """
        context = context or {}
        critiques: list[CritiqueResult] = []
        revisions: list[RevisionResult] = []
        current_response = response
        iteration = 0

        for iteration in range(self.max_iterations):
            # Step 1: Critique response against all principles
            iteration_critiques = await self._critique_response(
                query, current_response, context
            )
            critiques.extend(iteration_critiques)

            # Check if all principles pass
            failing_critiques = [c for c in iteration_critiques if not c.passes]
            if not failing_critiques:
                break

            # Step 2: Revise response based on critiques
            revision = await self._revise_response(
                query, current_response, failing_critiques
            )
            revisions.append(revision)
            current_response = revision.revised_response

        # Calculate final score
        total_weight = sum(p.weight for p in self.constitution.principles)
        passing_weight = sum(
            p.weight
            for p in self.constitution.principles
            if any(c.principle_id == p.id and c.passes for c in critiques)
        )
        score = passing_weight / total_weight if total_weight > 0 else 0.0

        result = ConstitutionalResult(
            query=query,
            original_response=response,
            final_response=current_response,
            passes=score >= self.pass_threshold,
            score=round(score, 4),
            critiques=critiques,
            revisions=revisions,
            iterations=iteration + 1,
            timestamp=datetime.now().isoformat(),
            details={
                "constitution": self.constitution.name,
                "total_principles": len(self.constitution.principles),
                "passing_principles": len([c for c in critiques if c.passes]),
                "context_used": bool(context),
            },
        )

        # Store in history
        self.validation_history.append(result)

        logger.info(
            f"Constitutional validation | passes={result.passes} | score={score:.2%} | "
            f"iterations={iteration + 1}"
        )

        return result

    async def _critique_response(
        self,
        query: str,
        response: str,
        context: dict[str, Any],
    ) -> list[CritiqueResult]:
        """Critique response against all principles."""
        critiques = []

        for principle in self.constitution.principles:
            if self.llm is not None:
                critique = await self._llm_critique(query, response, principle, context)
            else:
                critique = self._heuristic_critique(query, response, principle, context)

            critiques.append(critique)

        return critiques

    async def _llm_critique(
        self,
        query: str,
        response: str,
        principle: Principle,
        context: dict[str, Any],
    ) -> CritiqueResult:
        """Use LLM for principle critique."""
        try:
            critique_prompt = f"""You are evaluating an AI response against a constitutional principle.

Principle: {principle.name}
Description: {principle.description}
Critique Question: {principle.critique_prompt}

User Query: {query}
AI Response: {response}

Evaluate whether the response adheres to this principle. Respond with:
PASSES: [YES/NO]
SEVERITY: [LOW/MEDIUM/HIGH/CRITICAL] (if NO)
CRITIQUE: [Your detailed critique]
SUGGESTIONS: [Comma-separated suggestions if NO]
"""

            if hasattr(self.llm, "ainvoke"):
                llm_response = await self.llm.ainvoke(critique_prompt)
                content = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
            elif callable(self.llm):
                content = str(self.llm(critique_prompt))
            else:
                return self._heuristic_critique(query, response, principle, context)

            # Parse response
            return self._parse_critique_response(content, principle)

        except Exception as e:
            logger.warning(f"LLM critique failed for {principle.id}: {e}")
            return self._heuristic_critique(query, response, principle, context)

    def _parse_critique_response(
        self,
        content: str,
        principle: Principle,
    ) -> CritiqueResult:
        """Parse LLM critique response."""
        content_lower = content.lower()

        passes = "passes: yes" in content_lower or "passes:yes" in content_lower
        severity = "low"

        for sev in ["critical", "high", "medium", "low"]:
            if f"severity: {sev}" in content_lower or f"severity:{sev}" in content_lower:
                severity = sev
                break

        # Extract critique text
        critique = content
        if "critique:" in content_lower:
            idx = content_lower.index("critique:")
            critique = content[idx + 9:].strip()
            if "suggestions:" in critique.lower():
                critique = critique[:critique.lower().index("suggestions:")].strip()

        # Extract suggestions
        suggestions = []
        if "suggestions:" in content_lower:
            idx = content_lower.index("suggestions:")
            suggestions_text = content[idx + 12:].strip()
            suggestions = [s.strip() for s in suggestions_text.split(",") if s.strip()]

        return CritiqueResult(
            principle_id=principle.id,
            principle_name=principle.name,
            passes=passes,
            critique=critique[:500],  # Limit length
            severity=severity if not passes else "low",
            suggestions=suggestions[:5],  # Limit to 5 suggestions
        )

    def _heuristic_critique(
        self,
        query: str,
        response: str,
        principle: Principle,
        context: dict[str, Any],
    ) -> CritiqueResult:
        """Heuristic-based critique without LLM."""
        response_lower = response.lower()
        query_lower = query.lower()

        passes = True
        critique = "Response adheres to principle."
        severity = "low"
        suggestions: list[str] = []

        # Safety principles
        if principle.category == PrincipleCategory.SAFETY:
            if any(word in query_lower for word in ["hurricane", "tornado", "flood", "emergency"]):
                # Should mention safety
                if not any(word in response_lower for word in ["safety", "evacuate", "shelter", "warning"]):
                    passes = False
                    severity = "high"
                    critique = "Response lacks safety information for severe weather query."
                    suggestions = ["Add safety recommendations", "Include evacuation guidance"]

        # Accuracy principles
        elif principle.category == PrincipleCategory.ACCURACY:
            # Check for source attribution
            if principle.id == "weather_accuracy_1":
                if not any(src in response_lower for src in ["nws", "national weather service", "nhc", "noaa"]):
                    passes = False
                    severity = "medium"
                    critique = "Response lacks source attribution."
                    suggestions = ["Add source attribution (NWS, NHC, NOAA)"]

            # Check hurricane categories
            elif principle.id == "weather_accuracy_2":
                if "category" in response_lower and "hurricane" in query_lower:
                    # Basic check for category-wind alignment
                    if "category 5" in response_lower and "157" not in response:
                        passes = False
                        severity = "critical"
                        critique = "Category 5 mentioned without 157+ mph verification."
                        suggestions = ["Verify wind speed matches Saffir-Simpson scale"]

        # Honesty principles
        elif principle.category == PrincipleCategory.HONESTY:
            if principle.id == "weather_honesty_1":
                # Should acknowledge uncertainty for forecasts
                if "forecast" in query_lower or "tomorrow" in query_lower:
                    if not any(word in response_lower for word in ["may", "could", "confidence", "uncertainty", "likely"]):
                        passes = False
                        severity = "medium"
                        critique = "Forecast lacks uncertainty acknowledgment."
                        suggestions = ["Add uncertainty language", "Include confidence level"]

        # Harmlessness principles
        elif principle.category == PrincipleCategory.HARMLESSNESS:
            danger_words = ["severe", "dangerous", "warning", "emergency", "hurricane", "tornado"]
            downplay_words = ["nothing to worry", "don't worry", "no big deal", "minor", "just a little"]

            if any(d in query_lower for d in danger_words):
                if any(dp in response_lower for dp in downplay_words):
                    passes = False
                    severity = "critical"
                    critique = "Response may downplay severe weather risks."
                    suggestions = ["Remove downplaying language", "Emphasize safety precautions"]

        # Privacy principles
        elif principle.category == PrincipleCategory.PRIVACY:
            pii_patterns = ["ssn", "social security", "credit card", "password"]
            if any(pii in response_lower for pii in pii_patterns):
                passes = False
                severity = "critical"
                critique = "Response may expose sensitive information."
                suggestions = ["Remove PII references"]

        return CritiqueResult(
            principle_id=principle.id,
            principle_name=principle.name,
            passes=passes,
            critique=critique,
            severity=severity,
            suggestions=suggestions,
        )

    async def _revise_response(
        self,
        query: str,
        response: str,
        failing_critiques: list[CritiqueResult],
    ) -> RevisionResult:
        """Revise response based on failing critiques."""
        if self.llm is not None:
            return await self._llm_revise(query, response, failing_critiques)
        return self._heuristic_revise(query, response, failing_critiques)

    async def _llm_revise(
        self,
        query: str,
        response: str,
        failing_critiques: list[CritiqueResult],
    ) -> RevisionResult:
        """Use LLM for response revision."""
        try:
            critiques_text = "\n".join([
                f"- {c.principle_name}: {c.critique}"
                for c in failing_critiques
            ])

            suggestions_text = "\n".join([
                f"- {s}"
                for c in failing_critiques
                for s in c.suggestions
            ])

            revision_prompt = f"""Revise the following AI response to address the constitutional critiques.

User Query: {query}

Original Response: {response}

Critiques to Address:
{critiques_text}

Suggestions:
{suggestions_text}

Generate a revised response that addresses all critiques while maintaining helpfulness.
Output ONLY the revised response, nothing else:
"""

            if hasattr(self.llm, "ainvoke"):
                llm_response = await self.llm.ainvoke(revision_prompt)
                revised = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
            elif callable(self.llm):
                revised = str(self.llm(revision_prompt))
            else:
                return self._heuristic_revise(query, response, failing_critiques)

            return RevisionResult(
                original_response=response,
                revised_response=revised.strip(),
                revisions_made=[c.critique for c in failing_critiques],
                principles_applied=[c.principle_name for c in failing_critiques],
            )

        except Exception as e:
            logger.warning(f"LLM revision failed: {e}")
            return self._heuristic_revise(query, response, failing_critiques)

    def _heuristic_revise(
        self,
        query: str,
        response: str,
        failing_critiques: list[CritiqueResult],
    ) -> RevisionResult:
        """Heuristic-based response revision."""
        revised = response
        revisions_made = []

        for critique in failing_critiques:
            principle = self.constitution.get_principle(critique.principle_id)
            if principle is None:
                continue

            # Apply category-specific revisions
            if principle.category == PrincipleCategory.SAFETY:
                if "safety" not in revised.lower():
                    revised += "\n\nSafety Note: Please follow local emergency management guidance and stay informed through official sources."
                    revisions_made.append("Added safety note")

            elif principle.category == PrincipleCategory.ACCURACY:
                if principle.id == "weather_accuracy_1":
                    if "source:" not in revised.lower() and "nws" not in revised.lower():
                        revised += "\n\n(Source: National Weather Service)"
                        revisions_made.append("Added source attribution")

            elif principle.category == PrincipleCategory.HONESTY:
                if "forecast" in query.lower():
                    if "may" not in revised.lower() and "could" not in revised.lower():
                        revised = revised.replace("will be", "may be")
                        revised += "\n\nNote: Weather forecasts are subject to change. Please check for updates."
                        revisions_made.append("Added uncertainty acknowledgment")

        return RevisionResult(
            original_response=response,
            revised_response=revised,
            revisions_made=revisions_made,
            principles_applied=[c.principle_name for c in failing_critiques],
        )

    def get_validation_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent validation history."""
        return [r.model_dump() for r in self.validation_history[-limit:]]

    def clear_history(self) -> None:
        """Clear validation history."""
        self.validation_history = []
        logger.info("Constitutional AI validation history cleared")

    def get_statistics(self) -> dict[str, Any]:
        """Get validation statistics."""
        if not self.validation_history:
            return {"total_validations": 0}

        total = len(self.validation_history)
        passed = sum(1 for r in self.validation_history if r.passes)
        avg_score = sum(r.score for r in self.validation_history) / total
        avg_iterations = sum(r.iterations for r in self.validation_history) / total

        return {
            "total_validations": total,
            "passed_count": passed,
            "pass_rate": round(passed / total * 100, 2),
            "average_score": round(avg_score, 4),
            "average_iterations": round(avg_iterations, 2),
            "constitution": self.constitution.name,
        }
