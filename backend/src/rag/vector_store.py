"""Qdrant vector store configuration for RAG system.

Provides semantic search over weather knowledge base using Qdrant vector database.

Level 2 Configuration:
- Collection: weather_knowledge (single collection for all docs)
- Vector size: 1536 (text-embedding-3-small)
- Distance metric: COSINE (recommended for OpenAI embeddings)
- Storage: In-memory for L2 MVP (persist to disk in L5a)
- API: QdrantVectorStore (modern LangChain API, not deprecated Qdrant class)

Architecture:
- Development: http://localhost:6333
- Docker: http://qdrant:6333 (container networking)
- Production: Qdrant Cloud or self-hosted (L5a)

Future Enhancements (L5a):
- Multiple collections (hurricanes, safety, climate)
- Hybrid search (dense + sparse/BM25)
- Reranking with cross-encoder
- Query filters (metadata-based filtering)
"""

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from backend.config.settings import settings


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client with environment-aware configuration.

    Returns:
        QdrantClient: Configured Qdrant client instance

    Configuration:
        QDRANT_URL: From settings (configured in .env)
        QDRANT_API_KEY: Optional API key for Qdrant Cloud (L5a+)

    Example:
        >>> client = get_qdrant_client()
        >>> collections = client.get_collections()
        >>> print(collections)

    Note:
        - Development: Uses localhost:6333 (no API key)
        - Docker: Uses qdrant:6333 (container DNS, no API key)
        - Production: Uses Qdrant Cloud with API key (L5a+)
    """
    # Get Qdrant configuration from centralized settings
    qdrant_url = settings.QDRANT_URL
    api_key = settings.QDRANT_API_KEY

    # Create client
    if api_key:
        # Qdrant Cloud with API key (production, L5a)
        return QdrantClient(url=qdrant_url, api_key=api_key)
    else:
        # Local Qdrant (development, L2)
        return QdrantClient(url=qdrant_url)


def create_collection_if_not_exists(
    client: QdrantClient,
    collection_name: str = "weather_knowledge",
    vector_size: int = 1536,
    distance: Distance = Distance.COSINE,
) -> None:
    """Create Qdrant collection if it doesn't exist.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection (default: weather_knowledge)
        vector_size: Embedding dimension (default: 1536 for text-embedding-3-small)
        distance: Distance metric (default: COSINE for OpenAI embeddings)

    Distance Metrics:
        - COSINE: Range [0, 2], smaller = more similar (RECOMMENDED for OpenAI)
        - EUCLIDEAN: L2 distance, smaller = more similar
        - DOT: Dot product, larger = more similar (for normalized vectors)

    Example:
        >>> client = get_qdrant_client()
        >>> create_collection_if_not_exists(client)
        Collection 'weather_knowledge' created successfully

    Note:
        - Idempotent: Safe to call multiple times
        - Collection persists across restarts
        - L2: Single collection, L5a: Multiple collections by category
    """
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]

    if collection_name in collection_names:
        print(f"✅ Collection '{collection_name}' already exists")
        return

    # Create collection with vector configuration
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )

    print(f"✅ Collection '{collection_name}' created successfully")
    print(f"   Vector size: {vector_size}")
    print(f"   Distance metric: {distance}")


def get_vector_store(
    collection_name: str = "weather_knowledge",
    embeddings: OpenAIEmbeddings | None = None,
) -> QdrantVectorStore:
    """Get configured Qdrant vector store for RAG.

    This is the main entry point for vector store operations.
    Creates collection if needed and returns LangChain QdrantVectorStore wrapper.

    Args:
        collection_name: Qdrant collection name (default: weather_knowledge)
        embeddings: OpenAI embeddings instance (creates if None)

    Returns:
        QdrantVectorStore: LangChain Qdrant vector store wrapper (modern API)

    Example:
        >>> from backend.src.rag import get_vector_store
        >>> vectorstore = get_vector_store()
        >>>
        >>> # Add documents
        >>> from langchain_core.documents import Document
        >>> docs = [Document(page_content="Hurricane Katrina was a Category 5 hurricane")]
        >>> vectorstore.add_documents(docs)
        >>>
        >>> # Search
        >>> results = vectorstore.similarity_search("Category 5 hurricanes", k=5)
        >>> print(results[0].page_content)

    Usage in Agent:
        >>> retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        >>> from langchain.chains import RetrievalQA
        >>> qa_chain = RetrievalQA.from_chain_type(
        ...     llm=model,
        ...     retriever=retriever,
        ...     return_source_documents=True
        ... )

    Note:
        - Automatically creates collection if missing
        - Uses environment QDRANT_URL and OPENAI_API_KEY
        - L2: Single collection, L5a: Multiple collections with routing
    """
    # Import embeddings creator if not provided
    if embeddings is None:
        from backend.src.rag.embeddings import create_embeddings

        embeddings = create_embeddings()

    # Get Qdrant client
    client = get_qdrant_client()

    # Ensure collection exists
    create_collection_if_not_exists(client, collection_name)

    # Create LangChain QdrantVectorStore wrapper (modern API, LangChain 0.1.2+)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,  # Note: singular "embedding" in new API
    )

    return vectorstore


# Convenience function for testing
def test_vector_store_connection() -> bool:
    """Test Qdrant connection and vector store setup.

    Returns:
        bool: True if connection successful, False otherwise

    Example:
        >>> from backend.src.rag.vector_store import test_vector_store_connection
        >>> test_vector_store_connection()
        ✅ Qdrant connection successful
        ✅ Collection 'weather_knowledge' ready
        True
    """
    try:
        # Get client
        client = get_qdrant_client()

        # Test connection
        collections = client.get_collections()
        print(f"✅ Qdrant connection successful")
        print(f"   URL: {settings.QDRANT_URL}")
        print(f"   Collections: {len(collections.collections)}")

        # Test vector store creation
        vectorstore = get_vector_store()
        print(f"✅ Collection 'weather_knowledge' ready")

        return True

    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print(f"   Make sure Qdrant is running:")
        print(f"   docker run -p 6333:6333 qdrant/qdrant")
        return False


if __name__ == "__main__":
    # Run connection test
    test_vector_store_connection()
