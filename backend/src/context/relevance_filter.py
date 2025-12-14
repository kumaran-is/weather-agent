"""
Relevance Filtering for Context Optimization.

Phase 3 of the 5-phase context optimization pipeline.

Provides semantic relevance scoring and filtering of context chunks
to ensure only the most relevant information is passed to the LLM.

Key Features:
- Semantic similarity scoring using embeddings
- Keyword-based fallback when embeddings unavailable
- Weather domain-specific relevance boosting
- Configurable minimum relevance threshold
"""

from typing import Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class RelevanceFilter:
    """
    Phase 3: Relevance filtering with semantic scoring.

    Scores chunks by their relevance to the query using embedding-based
    semantic similarity or keyword-based fallback.
    """

    def __init__(
        self,
        embeddings: Any | None = None,
        min_score: float = 0.7,
        weather_boost: float = 0.1,
    ):
        """
        Initialize the relevance filter.

        Args:
            embeddings: Embeddings model for semantic similarity (optional)
            min_score: Minimum relevance score to keep chunks
            weather_boost: Score boost for weather-related content
        """
        self.embeddings = embeddings
        self.min_score = min_score
        self.weather_boost = weather_boost

        # Weather domain keywords for relevance boosting
        self.weather_keywords = {
            # High relevance weather terms
            "high": [
                "hurricane",
                "tropical storm",
                "evacuation",
                "warning",
                "alert",
                "emergency",
                "category",
                "storm surge",
                "wind speed",
                "landfall",
            ],
            # Medium relevance weather terms
            "medium": [
                "forecast",
                "temperature",
                "precipitation",
                "humidity",
                "pressure",
                "wind",
                "rain",
                "cloudy",
                "sunny",
                "conditions",
            ],
            # Location terms
            "location": [
                "florida",
                "gulf",
                "atlantic",
                "coast",
                "tampa",
                "miami",
                "orlando",
                "jacksonville",
                "naples",
                "key west",
            ],
        }

        logger.debug(
            f"RelevanceFilter initialized | min_score={min_score} | "
            f"embeddings_available={embeddings is not None}"
        )

    async def score_chunks(
        self, query: str, chunks: list[str]
    ) -> list[dict[str, Any]]:
        """
        Score chunks by relevance to query.

        Uses embedding-based semantic similarity if available,
        otherwise falls back to keyword-based scoring.

        Args:
            query: User's query string
            chunks: List of text chunks to score

        Returns:
            List of dicts with chunk, score, and scoring method
        """
        if not chunks:
            return []

        if self.embeddings is not None:
            try:
                return await self._semantic_scoring(query, chunks)
            except Exception as e:
                logger.warning(f"Semantic scoring failed, using keyword fallback: {e}")
                return self._keyword_scoring(query, chunks)
        else:
            return self._keyword_scoring(query, chunks)

    async def _semantic_scoring(
        self, query: str, chunks: list[str]
    ) -> list[dict[str, Any]]:
        """Score chunks using embedding-based semantic similarity."""
        # Get query embedding
        query_embedding = await self.embeddings.aembed_query(query)

        scored = []
        for chunk in chunks:
            # Get chunk embedding
            chunk_embedding = await self.embeddings.aembed_query(chunk)

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, chunk_embedding)

            # Apply weather domain boost
            boosted_score = self._apply_weather_boost(chunk, similarity)

            scored.append(
                {
                    "chunk": chunk,
                    "score": float(boosted_score),
                    "base_score": float(similarity),
                    "method": "semantic",
                }
            )

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)

        logger.debug(
            f"Semantic scoring complete | chunks={len(chunks)} | "
            f"above_threshold={sum(1 for s in scored if s['score'] >= self.min_score)}"
        )

        return scored

    def _keyword_scoring(
        self, query: str, chunks: list[str]
    ) -> list[dict[str, Any]]:
        """
        Fallback keyword-based scoring when embeddings unavailable.

        Scores based on:
        1. Query keyword overlap
        2. Weather domain keyword presence
        3. Location term matching
        """
        # Extract query keywords
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "just",
            "don",
            "now",
            "what",
            "me",
            "my",
            "i",
        }
        query_words = query_words - stop_words

        scored = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            chunk_words = set(chunk_lower.split()) - stop_words

            # Base score: query keyword overlap
            if query_words:
                overlap = len(query_words & chunk_words)
                base_score = overlap / len(query_words)
            else:
                base_score = 0.5  # Default score if no query keywords

            # Weather boost
            weather_score = self._calculate_weather_score(chunk_lower)
            boosted_score = min(1.0, base_score + weather_score)

            # Check for query entity matches (names, places)
            entity_boost = self._entity_match_boost(query_lower, chunk_lower)
            final_score = min(1.0, boosted_score + entity_boost)

            scored.append(
                {
                    "chunk": chunk,
                    "score": float(final_score),
                    "base_score": float(base_score),
                    "weather_boost": float(weather_score),
                    "entity_boost": float(entity_boost),
                    "method": "keyword",
                }
            )

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)

        logger.debug(
            f"Keyword scoring complete | chunks={len(chunks)} | "
            f"above_threshold={sum(1 for s in scored if s['score'] >= self.min_score)}"
        )

        return scored

    def _cosine_similarity(
        self, vec1: list[float], vec2: list[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)

        dot_product = np.dot(vec1_array, vec2_array)
        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _apply_weather_boost(self, chunk: str, base_score: float) -> float:
        """Apply weather domain relevance boost."""
        chunk_lower = chunk.lower()
        boost = 0.0

        # Check for high relevance keywords
        for keyword in self.weather_keywords["high"]:
            if keyword in chunk_lower:
                boost += 0.15
                break  # Only apply once per category

        # Check for medium relevance keywords
        for keyword in self.weather_keywords["medium"]:
            if keyword in chunk_lower:
                boost += 0.05
                break

        # Check for location keywords
        for keyword in self.weather_keywords["location"]:
            if keyword in chunk_lower:
                boost += 0.05
                break

        return min(1.0, base_score + boost)

    def _calculate_weather_score(self, chunk_lower: str) -> float:
        """Calculate weather-specific relevance score."""
        score = 0.0

        # High relevance keywords (0.2 each, max 0.4)
        high_matches = sum(
            1 for kw in self.weather_keywords["high"] if kw in chunk_lower
        )
        score += min(0.4, high_matches * 0.2)

        # Medium relevance keywords (0.1 each, max 0.2)
        medium_matches = sum(
            1 for kw in self.weather_keywords["medium"] if kw in chunk_lower
        )
        score += min(0.2, medium_matches * 0.1)

        # Location keywords (0.05 each, max 0.1)
        location_matches = sum(
            1 for kw in self.weather_keywords["location"] if kw in chunk_lower
        )
        score += min(0.1, location_matches * 0.05)

        return score

    def _entity_match_boost(self, query: str, chunk: str) -> float:
        """Boost score if specific entities from query appear in chunk."""
        boost = 0.0

        # Look for quoted terms in query
        import re

        quoted = re.findall(r'"([^"]*)"', query)
        for term in quoted:
            if term.lower() in chunk:
                boost += 0.2

        # Look for capitalized words (potential proper nouns)
        query_caps = re.findall(r"\b[A-Z][a-z]+\b", query)
        for cap in query_caps:
            if cap.lower() in chunk:
                boost += 0.1

        return min(0.3, boost)  # Cap at 0.3

    def filter_chunks(
        self, scored_chunks: list[dict[str, Any]], min_score: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Filter chunks by minimum score threshold.

        Args:
            scored_chunks: List of scored chunk dicts
            min_score: Override minimum score (uses instance default if None)

        Returns:
            List of chunks meeting the threshold
        """
        threshold = min_score if min_score is not None else self.min_score
        filtered = [c for c in scored_chunks if c["score"] >= threshold]

        logger.debug(
            f"Filtered chunks | total={len(scored_chunks)} | "
            f"kept={len(filtered)} | threshold={threshold}"
        )

        return filtered

    def get_top_chunks(
        self, scored_chunks: list[dict[str, Any]], k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Get top-k chunks by score.

        Args:
            scored_chunks: List of scored chunk dicts
            k: Number of top chunks to return

        Returns:
            Top k chunks by score
        """
        # Already sorted by score, just take top k
        return scored_chunks[:k]
