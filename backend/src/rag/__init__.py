"""RAG (Retrieval-Augmented Generation) system for Weather AI Agent.

This module provides vector search capabilities over weather knowledge:
- Historical weather data (Kaggle datasets)
- Hurricane information (Saffir-Simpson scale, historical storms)
- Weather safety guidelines
- Climate patterns and terminology

Level 2 Implementation:
- Basic semantic search with Qdrant vector store
- OpenAI embeddings (text-embedding-3-small)
- 500+ document knowledge base
- Cosine distance similarity

Level 5a Enhancements:
- Query decomposition for complex multi-part queries
- Better cache hit rates through focused sub-queries
- Parallel execution of decomposed queries
"""

from backend.src.rag.embeddings import create_embeddings

# L5a: Query decomposition
from backend.src.rag.query_decomposer import (
    DecomposedQuery,
    QueryDecomposer,
    decompose_query,
    get_query_decomposer,
)
from backend.src.rag.retriever import get_retriever, retrieve_weather_knowledge
from backend.src.rag.vector_store import get_vector_store

__all__ = [
    # Core RAG
    "create_embeddings",
    "get_vector_store",
    "get_retriever",
    "retrieve_weather_knowledge",
    # L5a: Query decomposition
    "QueryDecomposer",
    "DecomposedQuery",
    "decompose_query",
    "get_query_decomposer",
]
