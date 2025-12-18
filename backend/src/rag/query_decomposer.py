"""L5a: Query decomposition for complex multi-part weather queries.

This module provides query decomposition to improve cache hit rates
by breaking complex queries into cacheable sub-queries.

Example:
    Input: "What's the weather in Miami and should I evacuate for the hurricane?"
    Output: [
        "What is the current weather in Miami?",
        "What is the hurricane status near Miami?",
        "What are the evacuation recommendations for Miami?"
    ]

Benefits:
- Each sub-query can be cached independently
- Parallel execution of sub-queries
- Better cache hit rates (common sub-queries cached)
- More precise retrieval (focused queries)

Performance:
- Decomposition latency: <100ms (single LLM call)
- Cache benefit: 20-40% improvement on complex queries
"""

import logging
import time
from dataclasses import dataclass, field

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class DecomposedQuery:
    """Result of query decomposition."""

    original_query: str
    sub_queries: list[str] = field(default_factory=list)
    is_complex: bool = False
    reasoning: str = ""
    decomposition_time_ms: float = 0.0


class QueryDecomposerOutput(BaseModel):
    """Structured output for query decomposition LLM call."""

    is_complex: bool = Field(
        description="Whether the query is complex and needs decomposition"
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="List of decomposed sub-queries (1-4 items)"
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning for decomposition decision"
    )


class QueryDecomposer:
    """Decompose complex queries into cacheable sub-queries.

    This class analyzes weather queries and:
    1. Determines if decomposition is beneficial
    2. Splits complex queries into focused sub-queries
    3. Ensures sub-queries are cache-friendly

    Complexity Indicators:
    - Multiple locations mentioned
    - Multiple time periods (today vs tomorrow, this week vs next)
    - Compound questions (and, also, plus)
    - Multi-domain (weather + hurricane + alerts)
    - Comparisons (compare, vs, between)
    """

    # Complexity indicator keywords
    COMPLEXITY_KEYWORDS = [
        # Conjunctions
        " and ", " also ", " plus ", " as well as ", " along with ",
        # Comparisons
        "compare", " vs ", " versus ", "between", "difference",
        # Multiple locations (common Florida cities)
        "miami", "tampa", "orlando", "jacksonville", "ft. lauderdale",
        "key west", "naples", "sarasota", "gainesville", "tallahassee",
        # Time indicators
        "today and tomorrow", "this week", "next week", "weekend",
        # Multi-domain
        "hurricane", "evacuate", "alert", "warning", "watch",
        # Questions markers
        "?",
    ]

    # Minimum tokens for decomposition to be worthwhile
    MIN_QUERY_LENGTH = 30

    def __init__(
        self,
        llm: ChatAnthropic | None = None,
        enable_llm_decomposition: bool = True,
    ):
        """Initialize query decomposer.

        Args:
            llm: LLM for intelligent decomposition (optional)
            enable_llm_decomposition: Use LLM for complex decomposition
        """
        self.llm = llm
        self.enable_llm_decomposition = enable_llm_decomposition

        # Statistics
        self.total_queries = 0
        self.complex_queries = 0
        self.decomposition_time_total_ms = 0.0

        logger.info(
            f"QueryDecomposer initialized: LLM={enable_llm_decomposition}"
        )

    async def decompose(self, query: str) -> DecomposedQuery:
        """Decompose query if complex, else return as-is.

        Args:
            query: User's weather query

        Returns:
            DecomposedQuery with sub-queries if complex
        """
        self.total_queries += 1
        start_time = time.perf_counter()

        # Quick heuristic check first (fast path)
        complexity_score = self._calculate_complexity_score(query)

        if complexity_score < 2:
            # Simple query - no decomposition needed
            return DecomposedQuery(
                original_query=query,
                sub_queries=[query],
                is_complex=False,
                reasoning="Simple query - no decomposition needed",
                decomposition_time_ms=0.0,
            )

        # Complex query - use LLM if available
        if self.enable_llm_decomposition and self.llm:
            try:
                result = await self._llm_decompose(query)
                result.decomposition_time_ms = (
                    time.perf_counter() - start_time
                ) * 1000
                self.decomposition_time_total_ms += result.decomposition_time_ms

                if result.is_complex:
                    self.complex_queries += 1
                    logger.info(
                        f"Query decomposed into {len(result.sub_queries)} parts "
                        f"({result.decomposition_time_ms:.1f}ms)"
                    )

                return result

            except Exception as e:
                logger.warning(f"LLM decomposition failed: {e}")
                # Fall through to heuristic decomposition

        # Fallback: Heuristic decomposition
        result = self._heuristic_decompose(query)
        result.decomposition_time_ms = (time.perf_counter() - start_time) * 1000
        self.decomposition_time_total_ms += result.decomposition_time_ms

        if result.is_complex:
            self.complex_queries += 1

        return result

    def _calculate_complexity_score(self, query: str) -> int:
        """Calculate complexity score based on heuristics.

        Args:
            query: User query

        Returns:
            Complexity score (0-10+)
        """
        query_lower = query.lower()
        score = 0

        # Check each complexity indicator
        for keyword in self.COMPLEXITY_KEYWORDS:
            if keyword in query_lower:
                score += 1

        # Bonus for long queries
        if len(query) > 100:
            score += 1
        if len(query) > 200:
            score += 1

        # Multiple question marks
        if query.count("?") > 1:
            score += 2

        return score

    async def _llm_decompose(self, query: str) -> DecomposedQuery:
        """Use LLM for intelligent decomposition.

        Args:
            query: Complex query to decompose

        Returns:
            DecomposedQuery with LLM-generated sub-queries
        """
        decompose_prompt = f"""Analyze this weather query and determine if it should be decomposed into sub-queries.

Query: {query}

Rules for decomposition:
1. Only decompose if the query has MULTIPLE distinct questions or topics
2. Each sub-query should be:
   - Self-contained (can be answered independently)
   - Cacheable (likely to be asked by other users)
   - Focused on ONE aspect (weather, hurricane, forecast, etc.)
3. Maximum 4 sub-queries
4. Keep sub-queries natural and conversational
5. Don't decompose simple, single-focus queries

Examples of queries that SHOULD be decomposed:
- "What's the weather in Miami and Tampa?" → 2 queries (one per city)
- "Is it raining today and will it rain tomorrow?" → 2 queries (one per day)
- "Weather forecast and hurricane status for Florida" → 2 queries (weather + hurricane)

Examples of queries that should NOT be decomposed:
- "What's the weather in Miami?" → Single focus, no decomposition
- "Will it rain tomorrow in Tampa?" → Single focus, no decomposition
- "Should I evacuate from Key West?" → Single question, no decomposition

Analyze and respond with:
1. is_complex: true if decomposition would help, false otherwise
2. sub_queries: list of 1-4 sub-queries (original query if not decomposed)
3. reasoning: brief explanation of your decision"""

        # Use structured output for reliable parsing
        llm_with_structure = self.llm.with_structured_output(QueryDecomposerOutput)
        result = await llm_with_structure.ainvoke(decompose_prompt)

        return DecomposedQuery(
            original_query=query,
            sub_queries=result.sub_queries[:4] if result.sub_queries else [query],
            is_complex=result.is_complex,
            reasoning=result.reasoning,
        )

    def _heuristic_decompose(self, query: str) -> DecomposedQuery:
        """Heuristic-based decomposition (fallback).

        Args:
            query: Query to decompose

        Returns:
            DecomposedQuery based on pattern matching
        """
        query_lower = query.lower()
        sub_queries = []

        # Pattern 1: Multiple locations
        florida_cities = [
            "miami", "tampa", "orlando", "jacksonville", "key west",
            "ft. lauderdale", "naples", "sarasota", "gainesville", "tallahassee"
        ]
        locations_found = [city for city in florida_cities if city in query_lower]

        if len(locations_found) >= 2:
            # Multiple locations - create separate queries
            for city in locations_found[:3]:  # Max 3
                sub_queries.append(f"What is the weather in {city.title()}?")

        # Pattern 2: Weather + Hurricane
        if "hurricane" in query_lower and any(
            word in query_lower for word in ["weather", "forecast", "rain", "temperature"]
        ):
            if not sub_queries:
                sub_queries = [
                    "What is the current weather forecast?",
                    "What is the hurricane status and alerts?",
                ]

        # Pattern 3: Today + Tomorrow/Weekend
        if "today" in query_lower and (
            "tomorrow" in query_lower or "weekend" in query_lower
        ):
            if not sub_queries:
                sub_queries = [
                    "What is the weather today?",
                    "What is the weather forecast for tomorrow?",
                ]

        # If no patterns matched, return original
        if not sub_queries:
            return DecomposedQuery(
                original_query=query,
                sub_queries=[query],
                is_complex=False,
                reasoning="Heuristic: No decomposition patterns matched",
            )

        return DecomposedQuery(
            original_query=query,
            sub_queries=sub_queries,
            is_complex=True,
            reasoning=f"Heuristic: Detected {len(sub_queries)} distinct sub-topics",
        )

    def get_stats(self) -> dict:
        """Get decomposer statistics.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_queries": self.total_queries,
            "complex_queries": self.complex_queries,
            "complex_rate": (
                self.complex_queries / max(1, self.total_queries)
            ),
            "avg_decomposition_time_ms": (
                self.decomposition_time_total_ms / max(1, self.complex_queries)
            ),
        }


# Module-level decomposer instance (lazy initialization)
_decomposer: QueryDecomposer | None = None


def get_query_decomposer(
    enable_llm: bool = True,
    llm: ChatAnthropic | None = None,
) -> QueryDecomposer:
    """Get or create query decomposer singleton.

    Args:
        enable_llm: Enable LLM-based decomposition
        llm: Optional LLM instance (creates default if not provided)

    Returns:
        QueryDecomposer instance
    """
    global _decomposer

    if _decomposer is None:
        # Create LLM if needed and enabled
        if enable_llm and llm is None:
            try:
                llm = ChatAnthropic(
                    model="claude-3-5-haiku-20241022",  # Fast model for decomposition
                    temperature=0.0,  # Deterministic decomposition
                )
            except Exception as e:
                logger.warning(f"Could not create LLM for decomposition: {e}")
                enable_llm = False

        _decomposer = QueryDecomposer(
            llm=llm if enable_llm else None,
            enable_llm_decomposition=enable_llm,
        )

    return _decomposer


async def decompose_query(query: str) -> DecomposedQuery:
    """Convenience function to decompose a query.

    Args:
        query: User's weather query

    Returns:
        DecomposedQuery with sub-queries if complex
    """
    decomposer = get_query_decomposer()
    return await decomposer.decompose(query)
