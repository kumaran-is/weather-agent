"""
Dynamic Context Assembly for Context Optimization.

Phase 4 of the 5-phase context optimization pipeline.

Provides intelligent context assembly based on query type,
adapting the amount and type of context included based on
the complexity and nature of the query.

Key Features:
- Query type-aware assembly (SIMPLE, STANDARD, COMPLEX, EMERGENCY)
- Priority-based chunk selection
- Token budget management
- Weather domain-specific assembly rules
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DynamicAssembler:
    """
    Phase 4: Dynamic context assembly based on query type.

    Assembles context intelligently based on query complexity,
    ensuring token budget compliance while maximizing relevance.
    """

    # Chunk limits by query type
    CHUNK_LIMITS = {
        "SIMPLE": 2,  # Quick queries - minimal context
        "STANDARD": 5,  # Normal queries - moderate context
        "COMPLEX": 8,  # Multi-part queries - extensive context
        "EMERGENCY": 15,  # Safety queries - maximum relevant context
    }

    # Token budgets by query type (percentage of target)
    TOKEN_BUDGETS = {
        "SIMPLE": 0.5,  # 50% of target tokens
        "STANDARD": 0.8,  # 80% of target tokens
        "COMPLEX": 1.0,  # 100% of target tokens
        "EMERGENCY": 1.2,  # 120% of target tokens (safety priority)
    }

    # Priority weights by semantic type
    SEMANTIC_PRIORITIES = {
        "evacuation": 10,  # Highest priority - life safety
        "alert": 9,
        "warning": 9,
        "hurricane": 8,
        "storm": 7,
        "emergency": 10,
        "forecast": 6,
        "temperature": 5,
        "wind": 5,
        "heading": 4,
        "json": 3,
        "table": 3,
        "list": 2,
        "general": 1,
    }

    def __init__(self, target_tokens: int = 4000):
        """
        Initialize the dynamic assembler.

        Args:
            target_tokens: Target token count for assembled context
        """
        self.target_tokens = target_tokens

        logger.debug(f"DynamicAssembler initialized | target_tokens={target_tokens}")

    def assemble(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        query_type: str = "STANDARD",
    ) -> str:
        """
        Assemble context dynamically based on query type.

        Args:
            query: User's query string
            chunks: List of scored chunk dicts with 'chunk' and 'score' keys
            query_type: Type of query (SIMPLE, STANDARD, COMPLEX, EMERGENCY)

        Returns:
            Assembled context string optimized for the query type
        """
        if not chunks:
            return ""

        # Normalize query type
        query_type = query_type.upper() if query_type else "STANDARD"
        if query_type not in self.CHUNK_LIMITS:
            query_type = "STANDARD"

        # Get limits for this query type
        chunk_limit = self.CHUNK_LIMITS[query_type]
        token_budget = int(self.target_tokens * self.TOKEN_BUDGETS[query_type])

        logger.debug(
            f"Assembling context | query_type={query_type} | "
            f"chunk_limit={chunk_limit} | token_budget={token_budget}"
        )

        # Sort chunks by priority (considering both score and semantic type)
        prioritized_chunks = self._prioritize_chunks(chunks, query_type)

        # Select chunks within limits
        selected_chunks = self._select_chunks(
            prioritized_chunks, chunk_limit, token_budget
        )

        # Assemble final context
        assembled = self._format_assembly(selected_chunks, query_type)

        # Final token budget enforcement
        assembled = self._enforce_token_budget(assembled, token_budget)

        logger.info(
            f"Context assembled | query_type={query_type} | "
            f"chunks_selected={len(selected_chunks)} | "
            f"approx_tokens={len(assembled)//4}"
        )

        return assembled

    def _prioritize_chunks(
        self, chunks: list[dict[str, Any]], query_type: str
    ) -> list[dict[str, Any]]:
        """
        Prioritize chunks based on score and semantic type.

        For EMERGENCY queries, prioritizes safety-related content.
        """
        prioritized = []

        for chunk_data in chunks:
            chunk_text = chunk_data.get("chunk", "")
            base_score = chunk_data.get("score", 0.5)

            # Determine semantic type
            semantic_type = self._infer_semantic_type(chunk_text)

            # Calculate priority score
            type_priority = self.SEMANTIC_PRIORITIES.get(semantic_type, 1)

            # For EMERGENCY queries, heavily weight safety content
            if query_type == "EMERGENCY":
                if semantic_type in ["evacuation", "alert", "warning", "emergency"]:
                    type_priority *= 2

            # Combined priority = base_score * type_priority
            priority_score = base_score * (1 + type_priority * 0.1)

            prioritized.append(
                {
                    **chunk_data,
                    "priority_score": priority_score,
                    "semantic_type": semantic_type,
                }
            )

        # Sort by priority score descending
        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)

        return prioritized

    def _select_chunks(
        self,
        chunks: list[dict[str, Any]],
        chunk_limit: int,
        token_budget: int,
    ) -> list[dict[str, Any]]:
        """Select chunks within limits while maximizing coverage."""
        selected = []
        current_tokens = 0

        for chunk_data in chunks:
            # Check chunk limit
            if len(selected) >= chunk_limit:
                break

            chunk_text = chunk_data.get("chunk", "")
            chunk_tokens = len(chunk_text) // 4  # Approximate token count

            # Check token budget
            if current_tokens + chunk_tokens > token_budget:
                # Try to fit a truncated version
                remaining_tokens = token_budget - current_tokens
                if remaining_tokens > 100:  # Only if meaningful space left
                    truncated = chunk_text[: remaining_tokens * 4]
                    chunk_data = {**chunk_data, "chunk": truncated, "truncated": True}
                    selected.append(chunk_data)
                break

            selected.append(chunk_data)
            current_tokens += chunk_tokens

        return selected

    def _format_assembly(
        self, chunks: list[dict[str, Any]], query_type: str
    ) -> str:
        """Format assembled chunks into final context string."""
        if not chunks:
            return ""

        # Group chunks by semantic type for better organization
        grouped: dict[str, list[str]] = {}

        for chunk_data in chunks:
            semantic_type = chunk_data.get("semantic_type", "general")
            chunk_text = chunk_data.get("chunk", "")

            if semantic_type not in grouped:
                grouped[semantic_type] = []
            grouped[semantic_type].append(chunk_text)

        # Build formatted output
        sections = []

        # Priority order for sections (safety first for EMERGENCY)
        if query_type == "EMERGENCY":
            section_order = [
                "emergency",
                "evacuation",
                "alert",
                "warning",
                "hurricane",
                "storm",
                "forecast",
                "general",
            ]
        else:
            section_order = [
                "forecast",
                "temperature",
                "wind",
                "hurricane",
                "alert",
                "general",
            ]

        # Add sections in priority order
        for section_type in section_order:
            if section_type in grouped:
                section_chunks = grouped.pop(section_type)
                if section_type in ["emergency", "evacuation", "alert", "warning"]:
                    sections.append(f"[IMPORTANT: {section_type.upper()}]")
                sections.extend(section_chunks)

        # Add any remaining sections
        for remaining_chunks in grouped.values():
            sections.extend(remaining_chunks)

        # Join with appropriate separators
        return "\n\n".join(sections)

    def _enforce_token_budget(self, text: str, token_budget: int) -> str:
        """Ensure assembled text stays within token budget."""
        max_chars = token_budget * 4  # Approximate

        if len(text) <= max_chars:
            return text

        # Truncate at word boundary
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")

        if last_space > max_chars * 0.8:  # Don't truncate too much
            truncated = truncated[:last_space]

        return truncated + "..."

    def _infer_semantic_type(self, chunk: str) -> str:
        """Infer semantic type from chunk content."""
        lower_chunk = chunk.lower()

        # Check for weather domain types
        type_keywords = {
            "evacuation": ["evacuation", "evacuate", "shelter", "leave", "flee"],
            "alert": ["alert", "advisory", "notice"],
            "warning": ["warning", "danger", "hazard", "threat"],
            "hurricane": ["hurricane", "cyclone", "typhoon", "category"],
            "storm": ["storm", "tropical", "surge"],
            "emergency": ["emergency", "critical", "urgent", "immediate"],
            "forecast": ["forecast", "prediction", "outlook", "expected"],
            "temperature": ["temperature", "temp", "°f", "°c", "degrees"],
            "wind": ["wind", "mph", "knots", "gust"],
        }

        for semantic_type, keywords in type_keywords.items():
            if any(kw in lower_chunk for kw in keywords):
                return semantic_type

        # Check for structural types
        if chunk.startswith("#") or chunk.startswith("##"):
            return "heading"
        if "|" in chunk and chunk.count("|") >= 2:
            return "table"
        if chunk.strip().startswith("{") or chunk.strip().startswith("["):
            return "json"
        if any(chunk.strip().startswith(f"{i}.") for i in range(1, 20)):
            return "list"

        return "general"

    def get_assembly_stats(
        self, chunks: list[dict[str, Any]], query_type: str
    ) -> dict[str, Any]:
        """Get statistics about the assembly process."""
        chunk_limit = self.CHUNK_LIMITS.get(query_type.upper(), 5)
        token_budget = int(self.target_tokens * self.TOKEN_BUDGETS.get(query_type.upper(), 0.8))

        return {
            "query_type": query_type,
            "chunk_limit": chunk_limit,
            "token_budget": token_budget,
            "chunks_available": len(chunks),
            "chunks_selected": min(len(chunks), chunk_limit),
            "semantic_types": list(set(
                self._infer_semantic_type(c.get("chunk", ""))
                for c in chunks[:chunk_limit]
            )),
        }
