"""
Context Window Optimization for Weather AI Agent.

CRITICAL GAP ADDRESSED: Reduce token usage from 8K-12K → <4K (50-60% reduction)

5-Phase Optimization:
1. Intelligent Truncation: Remove redundant, keep essential
2. Semantic Chunking: Preserve meaning boundaries
3. Relevance Filtering: Score and filter by query relevance
4. Dynamic Assembly: Adapt context to query type
5. Hierarchical Loading: Load critical first, lazy-load rest

Target: <4K tokens/query, maintain >98% context recall
"""

from typing import Any
import logging
import time
import tiktoken
from pydantic import BaseModel, Field

from backend.src.context.semantic_chunker import SemanticChunker
from backend.src.context.relevance_filter import RelevanceFilter
from backend.src.context.dynamic_assembler import DynamicAssembler
from backend.src.context.hierarchical_loader import HierarchicalLoader
from backend.src.context.query_type_detector import get_optimization_config

logger = logging.getLogger(__name__)


class OptimizationResult(BaseModel):
    """Result of context optimization."""

    optimized_context: str = Field(description="Optimized context string")
    token_count: int = Field(description="Token count after optimization")
    original_tokens: int = Field(description="Original token count")
    reduction_pct: float = Field(description="Percentage reduction achieved")
    load_order: list[dict[str, Any]] = Field(
        default_factory=list, description="Hierarchical load order"
    )
    chunks_used: int = Field(description="Number of chunks used")
    chunks_filtered: int = Field(description="Number of chunks filtered out")
    optimization_time_ms: float = Field(description="Optimization time in milliseconds")
    phases_completed: list[str] = Field(
        default_factory=list, description="List of completed phases"
    )


class OptimizationMetrics(BaseModel):
    """Metrics for optimization performance tracking."""

    total_optimizations: int = 0
    avg_reduction_pct: float = 0.0
    avg_optimization_time_ms: float = 0.0
    target_met_count: int = 0
    target_miss_count: int = 0


class ContextWindowOptimizer:
    """
    Optimize context window usage through intelligent truncation and assembly.

    This is the main orchestrator for the 5-phase context optimization pipeline.
    It coordinates the SemanticChunker, RelevanceFilter, DynamicAssembler, and
    HierarchicalLoader to achieve 50-60% token reduction.
    """

    def __init__(
        self,
        embeddings: Any | None = None,
        target_tokens: int = 4000,
        min_relevance_score: float = 0.7,
        model_name: str = "cl100k_base",
    ):
        """
        Initialize the context window optimizer.

        Args:
            embeddings: Embeddings model for relevance scoring (optional)
            target_tokens: Target token count (<4K default)
            min_relevance_score: Minimum relevance score to keep chunks
            model_name: Tokenizer model name for token counting
        """
        self.embeddings = embeddings
        self.target_tokens = target_tokens
        self.min_relevance_score = min_relevance_score
        self.model_name = model_name

        # Initialize tokenizer for accurate token counting
        self.tokenizer = None
        try:
            self.tokenizer = tiktoken.get_encoding(model_name)
        except Exception:
            try:
                # Fallback to cl100k_base if model not found
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                # Network/proxy errors - use character-based estimation
                logger.warning(f"Tiktoken unavailable, using character-based estimation: {e}")
                self.tokenizer = None

        # Initialize components
        self.chunker = SemanticChunker(chunk_size=500, overlap=50)
        self.relevance_filter = RelevanceFilter(
            embeddings=embeddings, min_score=min_relevance_score
        )
        self.assembler = DynamicAssembler(target_tokens=target_tokens)
        self.loader = HierarchicalLoader()

        # Metrics tracking
        self.metrics = OptimizationMetrics()

        logger.info(
            f"ContextWindowOptimizer initialized | target_tokens={target_tokens} | "
            f"min_relevance={min_relevance_score}"
        )

    async def optimize(
        self, query: str, context: str, query_type: str = "STANDARD"
    ) -> OptimizationResult:
        """
        Optimize context window for query.

        Args:
            query: User's query string
            context: Full context string (potentially 8K-12K tokens)
            query_type: Type of query (SIMPLE, STANDARD, COMPLEX, EMERGENCY)

        Returns:
            OptimizationResult with optimized_context, token_count, reduction_pct
        """
        start_time = time.time()
        phases_completed = []
        original_tokens = self.count_tokens(context)

        logger.info(
            f"Starting context optimization | query_type={query_type} | "
            f"original_tokens={original_tokens}"
        )

        # Phase 1: Intelligent Truncation
        truncated = self._intelligent_truncate(context)
        phases_completed.append("intelligent_truncation")
        truncated_tokens = self.count_tokens(truncated)
        if original_tokens > 0:
            phase1_reduction = (1 - truncated_tokens / original_tokens) * 100
        else:
            phase1_reduction = 0.0
        logger.debug(
            f"Phase 1 complete | truncated_tokens={truncated_tokens} | "
            f"reduction={phase1_reduction:.1f}%"
        )

        # Phase 2: Semantic Chunking
        chunks = self.chunker.chunk(truncated)
        phases_completed.append("semantic_chunking")
        logger.debug(f"Phase 2 complete | chunks_created={len(chunks)}")

        # Phase 3: Relevance Filtering (query-type aware)
        # Get query-type-specific optimization config
        query_config = get_optimization_config(query_type)
        type_min_relevance = query_config.get("min_relevance", self.min_relevance_score)
        type_target_tokens = query_config.get("target_tokens", self.target_tokens)

        logger.debug(
            f"Phase 3 using query-type config | query_type={query_type} | "
            f"min_relevance={type_min_relevance} | target_tokens={type_target_tokens}"
        )

        if self.embeddings is not None:
            scored_chunks = await self.relevance_filter.score_chunks(query, chunks)
            relevant_chunks = [
                c for c in scored_chunks if c["score"] >= type_min_relevance
            ]
        else:
            # Without embeddings, use keyword-based filtering with type-aware threshold
            relevant_chunks = self._keyword_filter(query, chunks, type_min_relevance)
        phases_completed.append("relevance_filtering")
        logger.debug(
            f"Phase 3 complete | relevant_chunks={len(relevant_chunks)} | "
            f"filtered_out={len(chunks) - len(relevant_chunks)}"
        )

        # Phase 4: Dynamic Assembly
        assembled_context = self.assembler.assemble(
            query=query, chunks=relevant_chunks, query_type=query_type
        )
        phases_completed.append("dynamic_assembly")
        assembled_tokens = self.count_tokens(assembled_context)
        logger.debug(f"Phase 4 complete | assembled_tokens={assembled_tokens}")

        # Phase 5: Hierarchical Loading (prepare load order)
        load_order = self.loader.prepare_load_order(assembled_context, query_type)
        phases_completed.append("hierarchical_loading")
        logger.debug(f"Phase 5 complete | load_levels={len(load_order)}")

        # Calculate final metrics
        optimized_tokens = self.count_tokens(assembled_context)
        reduction_pct = (
            (1 - optimized_tokens / original_tokens) * 100 if original_tokens > 0 else 0
        )
        optimization_time_ms = (time.time() - start_time) * 1000

        # Update metrics
        self._update_metrics(reduction_pct, optimization_time_ms)

        result = OptimizationResult(
            optimized_context=assembled_context,
            token_count=optimized_tokens,
            original_tokens=original_tokens,
            reduction_pct=reduction_pct,
            load_order=load_order,
            chunks_used=len(relevant_chunks),
            chunks_filtered=len(chunks) - len(relevant_chunks),
            optimization_time_ms=optimization_time_ms,
            phases_completed=phases_completed,
        )

        logger.info(
            f"Context optimization complete | reduction={reduction_pct:.1f}% | "
            f"tokens={original_tokens}→{optimized_tokens} | time={optimization_time_ms:.1f}ms"
        )

        return result

    def _intelligent_truncate(self, context: str) -> str:
        """
        Phase 1: Intelligent truncation (remove obvious redundancy).

        Rules:
        - Keep system prompts, instructions
        - Keep recent conversation turns (last 5)
        - Drop duplicate information
        - Drop verbose error messages
        - Keep structured data (JSON, tables)
        """
        lines = context.split("\n")
        kept_lines: list[str] = []
        seen_content: set[str] = set()

        # Patterns for critical content
        critical_patterns = [
            "You are",
            "SYSTEM:",
            "System:",
            "Instructions:",
            "IMPORTANT:",
            "WARNING:",
            "CRITICAL:",
            "### ",
            "## ",
            "# ",
        ]

        # Keep system prompts and critical instructions
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check for critical patterns
            for pattern in critical_patterns:
                if pattern in line:
                    normalized = stripped.lower()
                    if normalized not in seen_content:
                        kept_lines.append(line)
                        seen_content.add(normalized)
                    break

        # Keep recent conversation (last 5 turns)
        conversation_lines = [
            line
            for line in lines
            if any(
                marker in line
                for marker in ["User:", "Assistant:", "Human:", "AI:", "Q:", "A:"]
            )
        ]
        for line in conversation_lines[-10:]:  # Last 5 turns = 10 lines
            normalized = line.strip().lower()
            if normalized and normalized not in seen_content:
                kept_lines.append(line)
                seen_content.add(normalized)

        # Keep structured data (JSON blocks, tables)
        in_json = False
        json_buffer: list[str] = []

        for line in lines:
            stripped = line.strip()

            # Detect JSON start
            if stripped.startswith("{") or stripped.startswith("["):
                in_json = True
                json_buffer = [line]
            elif in_json:
                json_buffer.append(line)
                # Detect JSON end
                if stripped.endswith("}") or stripped.endswith("]"):
                    in_json = False
                    json_content = "\n".join(json_buffer)
                    if json_content not in seen_content:
                        kept_lines.extend(json_buffer)
                        seen_content.add(json_content)
                    json_buffer = []

            # Keep table rows
            if "|" in line and line.count("|") >= 2:
                normalized = stripped.lower()
                if normalized not in seen_content:
                    kept_lines.append(line)
                    seen_content.add(normalized)

        # Keep weather-specific content
        weather_keywords = [
            "temperature",
            "humidity",
            "wind",
            "forecast",
            "hurricane",
            "storm",
            "weather",
            "precipitation",
            "pressure",
            "mph",
            "°F",
            "°C",
            "evacuation",
            "alert",
            "warning",
        ]

        for line in lines:
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in weather_keywords):
                normalized = line.strip().lower()
                if normalized and normalized not in seen_content:
                    kept_lines.append(line)
                    seen_content.add(normalized)

        return "\n".join(kept_lines)

    def _keyword_filter(
        self, query: str, chunks: list[str], min_score: float = 0.1
    ) -> list[dict[str, Any]]:
        """
        Fallback keyword-based filtering when embeddings not available.

        Scores chunks based on keyword overlap with query.
        Uses query-type-aware threshold for EMERGENCY (lower = keep more).

        Args:
            query: User query string
            chunks: List of text chunks to filter
            min_score: Minimum score threshold (lower for EMERGENCY to preserve more)
        """
        query_words = set(query.lower().split())

        # Add weather/safety keywords for better matching
        safety_keywords = {
            "hurricane", "evacuation", "warning", "emergency", "shelter",
            "category", "storm", "surge", "mandatory", "zone", "danger"
        }

        scored_chunks = []

        for chunk in chunks:
            chunk_lower = chunk.lower()
            chunk_words = set(chunk_lower.split())
            overlap = len(query_words & chunk_words)
            base_score = overlap / max(len(query_words), 1)

            # Boost score for safety-critical content
            safety_matches = sum(1 for kw in safety_keywords if kw in chunk_lower)
            safety_boost = min(safety_matches * 0.15, 0.5)  # Max 0.5 boost

            final_score = min(base_score + safety_boost, 1.0)
            scored_chunks.append({"chunk": chunk, "score": final_score})

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)

        # Return chunks with score above threshold
        return [c for c in scored_chunks if c["score"] >= min_score]

    def count_tokens(self, text: str) -> int:
        """
        Accurately count tokens using tiktoken.

        Falls back to character-based estimation if tiktoken fails.
        """
        if not text:
            return 0

        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass

        # Fallback: 1 token ≈ 4 characters
        return len(text) // 4

    def _update_metrics(self, reduction_pct: float, time_ms: float) -> None:
        """Update optimization metrics for tracking."""
        self.metrics.total_optimizations += 1

        # Update rolling average
        n = self.metrics.total_optimizations
        self.metrics.avg_reduction_pct = (
            (self.metrics.avg_reduction_pct * (n - 1) + reduction_pct) / n
        )
        self.metrics.avg_optimization_time_ms = (
            (self.metrics.avg_optimization_time_ms * (n - 1) + time_ms) / n
        )

        # Track target achievement
        if reduction_pct >= 50:
            self.metrics.target_met_count += 1
        else:
            self.metrics.target_miss_count += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get current optimization metrics."""
        return {
            "total_optimizations": self.metrics.total_optimizations,
            "avg_reduction_pct": round(self.metrics.avg_reduction_pct, 2),
            "avg_optimization_time_ms": round(self.metrics.avg_optimization_time_ms, 2),
            "target_met_count": self.metrics.target_met_count,
            "target_miss_count": self.metrics.target_miss_count,
            "target_achievement_rate": (
                round(
                    self.metrics.target_met_count
                    / max(self.metrics.total_optimizations, 1)
                    * 100,
                    2,
                )
            ),
        }

    def reset_metrics(self) -> None:
        """Reset optimization metrics."""
        self.metrics = OptimizationMetrics()
        logger.info("Optimization metrics reset")

    def optimize_sync(
        self, query: str, context: str, query_type: str = "STANDARD"
    ) -> OptimizationResult:
        """
        Synchronous version of optimize() for use in sync contexts.

        Level 8a: Enables context optimization in sync functions like create_weather_agent.
        Only works when embeddings=None (keyword-based filtering).

        Args:
            query: User's query string
            context: Full context string (potentially 8K-12K tokens)
            query_type: Type of query (SIMPLE, STANDARD, COMPLEX, EMERGENCY)

        Returns:
            OptimizationResult with optimized_context, token_count, reduction_pct

        Raises:
            RuntimeError: If embeddings are configured (requires async)
        """
        import time as time_module

        start_time = time_module.time()
        phases_completed = []
        original_tokens = self.count_tokens(context)

        logger.info(
            f"Starting sync context optimization | query_type={query_type} | "
            f"original_tokens={original_tokens}"
        )

        # Phase 1: Intelligent Truncation
        truncated = self._intelligent_truncate(context)
        phases_completed.append("intelligent_truncation")

        # Phase 2: Semantic Chunking
        chunks = self.chunker.chunk(truncated)
        phases_completed.append("semantic_chunking")

        # Phase 3: Relevance Filtering (sync only works with keyword-based)
        if self.embeddings is not None:
            # Fall back to async wrapper if embeddings configured
            import asyncio

            logger.warning(
                "Embeddings configured - using asyncio.run() for sync optimization"
            )
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context - can't use asyncio.run
                    raise RuntimeError(
                        "optimize_sync called with embeddings in async context. "
                        "Use optimize() instead."
                    )
            except RuntimeError:
                pass
            return asyncio.run(self.optimize(query, context, query_type))

        # Phase 3: Relevance Filtering (query-type aware, sync-safe)
        # Get query-type-specific optimization config
        query_config = get_optimization_config(query_type)
        type_min_relevance = query_config.get("min_relevance", self.min_relevance_score)

        logger.debug(
            f"Phase 3 (sync) using query-type config | query_type={query_type} | "
            f"min_relevance={type_min_relevance}"
        )

        relevant_chunks = self._keyword_filter(query, chunks, type_min_relevance)
        phases_completed.append("relevance_filtering")
        logger.debug(
            f"Phase 3 complete (sync) | relevant_chunks={len(relevant_chunks)} | "
            f"filtered_out={len(chunks) - len(relevant_chunks)}"
        )

        # Phase 4: Dynamic Assembly
        assembled_context = self.assembler.assemble(
            query=query, chunks=relevant_chunks, query_type=query_type
        )
        phases_completed.append("dynamic_assembly")

        # Phase 5: Hierarchical Loading (prepare load order)
        load_order = self.loader.prepare_load_order(assembled_context, query_type)
        phases_completed.append("hierarchical_loading")

        # Calculate final metrics
        optimized_tokens = self.count_tokens(assembled_context)
        reduction_pct = (
            (1 - optimized_tokens / original_tokens) * 100 if original_tokens > 0 else 0
        )
        optimization_time_ms = (time_module.time() - start_time) * 1000

        # Update metrics
        self._update_metrics(reduction_pct, optimization_time_ms)

        result = OptimizationResult(
            optimized_context=assembled_context,
            token_count=optimized_tokens,
            original_tokens=original_tokens,
            reduction_pct=reduction_pct,
            load_order=load_order,
            chunks_used=len(relevant_chunks),
            chunks_filtered=len(chunks) - len(relevant_chunks),
            optimization_time_ms=optimization_time_ms,
            phases_completed=phases_completed,
        )

        logger.info(
            f"Sync context optimization complete | reduction={reduction_pct:.1f}% | "
            f"tokens={original_tokens}→{optimized_tokens} | time={optimization_time_ms:.1f}ms"
        )

        return result
