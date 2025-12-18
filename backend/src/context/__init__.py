"""
Context Window Optimization Module for Weather AI Agent.

Level 6a: Self-Evolving AI Platform - Context Optimization
Level 8a: Agent Integration with Query Type Detection 🆕

This module provides intelligent context window management to achieve
50-60% token cost reduction while maintaining >98% context recall.

5-Phase Optimization Pipeline:
1. Intelligent Truncation: Remove redundant, keep essential
2. Semantic Chunking: Preserve meaning boundaries
3. Relevance Filtering: Score and filter by query relevance
4. Dynamic Assembly: Adapt context to query type
5. Hierarchical Loading: Load critical first, lazy-load rest

Query Type Detection (Level 8a):
- EMERGENCY: Life-safety queries (30-40% reduction, safety prioritized)
- COMPLEX: Multi-part queries (50-60% reduction)
- STANDARD: Normal weather queries (50-60% reduction)
- SIMPLE: Single data point queries (60-70% reduction)

Target Metrics:
- Token reduction: 8K-12K → <4K tokens (50-60% reduction)
- Context recall: >98% (no loss of critical information)
- Optimization latency: <100ms
- Relevance threshold: >0.7 for kept chunks
"""

from backend.src.context.context_optimizer import ContextWindowOptimizer
from backend.src.context.dynamic_assembler import DynamicAssembler
from backend.src.context.hierarchical_loader import HierarchicalLoader
from backend.src.context.query_type_detector import (  # 🆕 Level 8a
    QueryType,
    detect_query_type,
    get_optimization_config,
)
from backend.src.context.relevance_filter import RelevanceFilter
from backend.src.context.semantic_chunker import SemanticChunker

__all__ = [
    # Core optimizer
    "ContextWindowOptimizer",
    # Pipeline components
    "SemanticChunker",
    "RelevanceFilter",
    "DynamicAssembler",
    "HierarchicalLoader",
    # 🆕 Level 8a: Query type detection
    "detect_query_type",
    "get_optimization_config",
    "QueryType",
]
