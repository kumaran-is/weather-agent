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

Future Enhancements:
- L3a: Hybrid search (dense + sparse)
- L3a: Query rewriting and multi-perspective retrieval
- L5a: Reranking and multi-vector retrieval
- L5a: Agentic RAG with LangGraph
"""

from backend.src.rag.embeddings import create_embeddings
from backend.src.rag.vector_store import get_vector_store
from backend.src.rag.retriever import get_retriever, retrieve_weather_knowledge

__all__ = [
    "create_embeddings",
    "get_vector_store",
    "get_retriever",
    "retrieve_weather_knowledge",
]
