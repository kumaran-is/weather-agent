"""RAG retriever configuration for agent integration.

Provides semantic search capabilities over weather knowledge base:
- Hurricane information (Saffir-Simpson scale, historical storms)
- Weather safety guidelines
- Climate patterns and historical data
- Weather terminology

Level 2 Implementation:
- Semantic search with Qdrant vector store
- Configurable k (number of results)
- Metadata filtering support
- LangChain retriever interface

Usage:
    >>> from backend.src.rag.retriever import get_retriever
    >>> retriever = get_retriever(k=5)
    >>> results = retriever.invoke("What is a Category 5 hurricane?")
    >>> print(results[0].page_content)
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from backend.src.rag.vector_store import get_vector_store


def get_retriever(
    k: int = 5,
    search_kwargs: dict[str, any] | None = None,
) -> BaseRetriever:
    """Get configured RAG retriever for weather knowledge base.

    This retriever provides semantic search over the weather knowledge base
    using Qdrant vector store and OpenAI embeddings.

    Args:
        k: Number of documents to retrieve (default: 5)
        search_kwargs: Additional search parameters for Qdrant
            - filter: Metadata filters (dict)
            - score_threshold: Minimum similarity score (float)

    Returns:
        BaseRetriever: LangChain retriever instance configured for weather knowledge

    Example:
        >>> # Basic usage
        >>> retriever = get_retriever(k=5)
        >>> results = retriever.invoke("Category 5 hurricane")
        >>> print(f"Found {len(results)} documents")
        Found 5 documents

        >>> # With metadata filtering
        >>> retriever = get_retriever(
        ...     k=3,
        ...     search_kwargs={"filter": {"category": "hurricanes"}}
        ... )
        >>> results = retriever.invoke("evacuation zones")

        >>> # With score threshold
        >>> retriever = get_retriever(
        ...     k=10,
        ...     search_kwargs={"score_threshold": 0.7}
        ... )
        >>> results = retriever.invoke("heat index calculation")

    Note:
        - Uses QdrantVectorStore with text-embedding-3-small (1536 dims)
        - Cosine distance similarity metric
        - Returns documents sorted by relevance score
        - L2: Basic semantic search
        - L3a: Will add hybrid search (dense + sparse)
    """
    # Get vector store
    vectorstore = get_vector_store()

    # Configure search kwargs
    if search_kwargs is None:
        search_kwargs = {}

    # Set k parameter
    search_kwargs["k"] = k

    # Create retriever from vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # L2: Basic similarity, L3a: hybrid
        search_kwargs=search_kwargs,
    )

    return retriever


def retrieve_weather_knowledge(
    query: str,
    k: int = 5,
    filter_metadata: dict[str, any] | None = None,
) -> list[Document]:
    """Retrieve weather knowledge documents for a query.

    Convenience function for direct retrieval without creating retriever instance.
    Useful for one-off queries or testing.

    Args:
        query: Search query (natural language)
        k: Number of documents to retrieve (default: 5)
        filter_metadata: Optional metadata filters

    Returns:
        list[Document]: Retrieved documents with page_content and metadata

    Example:
        >>> from backend.src.rag.retriever import retrieve_weather_knowledge
        >>> docs = retrieve_weather_knowledge("What is a Category 5 hurricane?", k=3)
        >>> for doc in docs:
        ...     print(f"Source: {doc.metadata.get('source', 'unknown')}")
        ...     print(f"Content: {doc.page_content[:100]}...")
        Source: curated/saffir_simpson_scale.txt
        Content: Category 5 hurricanes have sustained winds of 157 mph or higher...

    Note:
        - This is a convenience wrapper around get_retriever()
        - For multiple queries, prefer creating a retriever instance once
        - Results are sorted by relevance (cosine similarity)
    """
    # Configure search kwargs
    search_kwargs: dict[str, any] = {}
    if filter_metadata:
        search_kwargs["filter"] = filter_metadata

    # Get retriever
    retriever = get_retriever(k=k, search_kwargs=search_kwargs)

    # Invoke retrieval
    results = retriever.invoke(query)

    return results


# Convenience function for testing
def test_retriever() -> bool:
    """Test RAG retriever connection and basic retrieval.

    Returns:
        bool: True if retrieval successful, False otherwise

    Example:
        >>> from backend.src.rag.retriever import test_retriever
        >>> test_retriever()
        ✅ Retriever created successfully
        ✅ Query: 'Category 5 hurricane'
        ✅ Retrieved 5 documents
        ✅ Sample result: Category 5 hurricanes have sustained winds of 157 mph...
        True
    """
    try:
        # Create retriever
        retriever = get_retriever(k=5)
        print("✅ Retriever created successfully")

        # Test query
        query = "Category 5 hurricane"
        print(f"✅ Query: '{query}'")

        # Retrieve documents
        results = retriever.invoke(query)
        print(f"✅ Retrieved {len(results)} documents")

        # Print sample result
        if results:
            sample = results[0].page_content[:100]
            print(f"✅ Sample result: {sample}...")

        return True

    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        print("   Make sure Qdrant is running: docker-compose up -d")
        print("   Make sure knowledge base is loaded: make rag-load")
        return False


if __name__ == "__main__":
    # Run retriever test
    test_retriever()
