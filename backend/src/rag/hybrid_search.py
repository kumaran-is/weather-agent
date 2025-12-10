"""Hybrid Search: Dense (Semantic) + Sparse (Keyword) Retrieval.

This module implements: Hybrid Search combining:
- Dense retrieval: Embeddings-based semantic search with MMR for diversity
- Sparse retrieval: BM25 keyword-based search for exact term matching

Uses Reciprocal Rank Fusion (RRF) to combine both approaches (70% semantic, 30% keyword)
to get the best of both worlds:
- Semantic captures meaning and related concepts
- Keyword ensures exact term matches aren't missed

Level 2 Implementation (LangChain v1.x):
- Custom RRF (Reciprocal Rank Fusion) implementation
- MMR (Maximum Marginal Relevance) for diverse results
- Configurable weights (70/30 default)
- NO EnsembleRetriever (deprecated in LangChain 1.x)

Deferred to Level 5a:
- Cross-encoder reranking
- Query expansion
- Advanced RRF variants
"""

# ✅ LangChain v1.x Compliance Note:
# BM25Retriever from langchain_community is acceptable for Level 2
# No direct equivalent in langchain_core yet (as of v1.0)
# This is the recommended approach until a core BM25 retriever exists
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from backend.src.rag.vector_store import get_vector_store
from backend.config.settings import settings
import logging

logger = logging.getLogger(__name__)


def get_all_documents_from_vectorstore() -> list[Document]:
    """Retrieve all documents from Qdrant vector store for BM25 indexing.

    BM25 requires access to all documents to build the term frequency index.
    We fetch all documents from Qdrant and convert them to LangChain Document format.

    Returns:
        list[Document]: All documents in the vector store

    Note:
        This loads all documents into memory. For large collections (>10K docs),
        consider pagination or incremental BM25 updates (deferred to L5a).
    """
    # Qdrant client for raw access (uses settings.QDRANT_URL for Docker compatibility)
    from qdrant_client import QdrantClient
    client = QdrantClient(url=settings.QDRANT_URL)

    collection_name = "weather_knowledge"

    # Fetch all documents (paginated for safety)
    all_docs = []
    offset = None
    batch_size = 100

    while True:
        # Scroll through collection
        result = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False  # Don't need vectors for BM25
        )

        points, next_offset = result

        if not points:
            break

        # Convert Qdrant points to LangChain Documents
        for point in points:
            payload = point.payload
            doc = Document(
                page_content=payload.get("page_content", ""),
                metadata=payload.get("metadata", {})
            )
            all_docs.append(doc)

        if next_offset is None:
            break

        offset = next_offset

    logger.info(f"✅ Loaded {len(all_docs)} documents from Qdrant for BM25 indexing")
    return all_docs


def reciprocal_rank_fusion(
    results_list: list[list[Document]],
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int = 10
) -> list[Document]:
    """Reciprocal Rank Fusion (RRF) algorithm to combine ranked lists.

    Modern replacement for EnsembleRetriever in LangChain 1.x.

    RRF formula: RRF_score(d) = sum(weight_i × 1 / (k + rank_i(d)))
    where k is a constant (typically 60) and rank is the position in each list.

    Args:
        results_list: List of document lists from different retrievers
        weights: Optional weights for each retriever (default: equal weights)
        k: RRF constant (default: 60, prevents division by small numbers)
        top_k: Number of final documents to return

    Returns:
        list[Document]: Merged and re-ranked documents

    Example:
        >>> vector_results = vector_retriever.invoke("Category 5 hurricane")
        >>> bm25_results = bm25_retriever.invoke("Category 5 hurricane")
        >>> combined = reciprocal_rank_fusion([vector_results, bm25_results], weights=[0.7, 0.3])
    """
    # Default to equal weights if not provided
    if weights is None:
        weights = [1.0 / len(results_list)] * len(results_list)

    # Normalize weights to sum to 1.0
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    # Track scores for each unique document
    doc_scores: dict[str, float] = {}
    doc_objects: dict[str, Document] = {}

    # Calculate weighted RRF scores
    for retriever_idx, docs in enumerate(results_list):
        weight = weights[retriever_idx]
        for rank, doc in enumerate(docs, 1):
            # Use page_content as unique key
            doc_key = doc.page_content

            # Weighted RRF formula: weight × (1 / (k + rank))
            score = weight * (1.0 / (k + rank))

            # Accumulate scores from different retrievers
            if doc_key in doc_scores:
                doc_scores[doc_key] += score
            else:
                doc_scores[doc_key] = score
                doc_objects[doc_key] = doc

    # Sort by RRF score (descending)
    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return top K documents
    return [doc_objects[doc_key] for doc_key, _ in sorted_docs[:top_k]]


class HybridRetrieverManager:
    """✅ v1.x: Class-based retriever management (replaces global singletons).

    Benefits over global state:
    - Test isolation (each test gets independent instance)
    - No race conditions in concurrent environments
    - Easier to mock/stub for testing
    - Follows SOLID principles
    """

    def __init__(self):
        """Initialize retriever manager with lazy loading."""
        self._vector_retriever = None
        self._bm25_retriever = None
        self._initialized = False

    def _initialize_retrievers(self, k: int = 10, fetch_k: int = 20, lambda_mult: float = 0.7):
        """Initialize vector and BM25 retrievers (called once on first use).

        Args:
            k: Number of documents to retrieve
            fetch_k: Number of documents to fetch before MMR reranking
            lambda_mult: MMR diversity parameter (0.7 = 70% relevance, 30% diversity)
        """
        if not self._initialized:
            logger.info("🔧 Initializing hybrid search retrievers (first call)")

            vectorstore = get_vector_store()

            # Dense retriever (semantic) with Maximum Marginal Relevance
            # MMR balances relevance with diversity to avoid redundant results
            self._vector_retriever = vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={
                    "k": k,  # Return k final documents
                    "fetch_k": fetch_k,  # Fetch more, then re-rank to k (diversity)
                    "lambda_mult": lambda_mult  # Balance relevance vs diversity
                }
            )

            logger.info(f"✅ Dense retriever: MMR (k={k}, fetch_k={fetch_k}, lambda={lambda_mult})")

            # Sparse retriever (keyword) with BM25
            # BM25 is excellent for exact term matching (e.g., "Saffir-Simpson", "Cat 5")
            all_docs = get_all_documents_from_vectorstore()

            self._bm25_retriever = BM25Retriever.from_documents(all_docs)
            self._bm25_retriever.k = k  # Retrieve top k

            logger.info(f"✅ Sparse retriever: BM25 (k={k}, indexed {len(all_docs)} docs)")

            self._initialized = True

    def get_retrievers(self, k: int = 10, fetch_k: int = 20, lambda_mult: float = 0.7):
        """Get or initialize retrievers.

        Args:
            k: Number of documents to retrieve
            fetch_k: Number of documents to fetch before MMR reranking
            lambda_mult: MMR diversity parameter

        Returns:
            tuple: (vector_retriever, bm25_retriever)
        """
        self._initialize_retrievers(k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
        return self._vector_retriever, self._bm25_retriever


# ⚠️ Module-level singleton for backward compatibility (Level 2 simplicity)
# Prefer get_hybrid_retriever_manager() in production for test isolation
_default_manager = HybridRetrieverManager()


def get_hybrid_retriever_manager() -> HybridRetrieverManager:
    """✅ v1.x: Factory function to create HybridRetrieverManager instance.

    Benefits over module-level singleton:
    - Test isolation (each test gets independent instance)
    - No race conditions in concurrent environments
    - Easier to mock/stub for testing
    - Follows SOLID principles (dependency injection)

    Returns:
        HybridRetrieverManager: New manager instance

    Example:
        >>> # Production code (preferred)
        >>> manager = get_hybrid_retriever_manager()
        >>> docs = hybrid_search_with_rrf("query", manager=manager)
        >>>
        >>> # Testing code
        >>> manager = get_hybrid_retriever_manager()
        >>> # Mock manager._vector_retriever as needed
    """
    return HybridRetrieverManager()


def hybrid_search_with_rrf(
    query: str,
    vector_weight: float | None = None,
    bm25_weight: float | None = None,
    top_k: int = 10,
    rrf_k: int = 60,
    manager: HybridRetrieverManager | None = None
) -> list[Document]:
    """Hybrid search using RRF fusion (LangChain 1.x compatible).

    Combines vector search (semantic) and BM25 (keyword) using
    Reciprocal Rank Fusion algorithm.

    Args:
        query: Search query
        vector_weight: Weight for vector search (None = use env HYBRID_SEARCH_VECTOR_WEIGHT, default: 0.7)
        bm25_weight: Weight for BM25 search (None = use env HYBRID_SEARCH_BM25_WEIGHT, default: 0.3)
        top_k: Number of final documents to return
        rrf_k: RRF constant (default: 60)
        manager: Optional HybridRetrieverManager instance (for testing/DI)

    Returns:
        list[Document]: Combined and re-ranked documents

    Example:
        >>> # Use environment defaults (70% semantic, 30% keyword)
        >>> docs = hybrid_search_with_rrf("Category 5 hurricane", top_k=5)
        >>>
        >>> # Override weights for specific query
        >>> docs = hybrid_search_with_rrf("Category 5 hurricane", vector_weight=0.5, bm25_weight=0.5, top_k=5)
        >>>
        >>> # Use custom manager for testing
        >>> manager = HybridRetrieverManager()
        >>> docs = hybrid_search_with_rrf("query", manager=manager)
    """
    # Use settings if weights not provided
    vector_weight = vector_weight if vector_weight is not None else settings.HYBRID_SEARCH_VECTOR_WEIGHT
    bm25_weight = bm25_weight if bm25_weight is not None else settings.HYBRID_SEARCH_BM25_WEIGHT

    # Use provided manager or default
    if manager is None:
        manager = _default_manager

    # Initialize retrievers on first call
    vector_retriever, bm25_retriever = manager.get_retrievers(k=top_k * 2, fetch_k=top_k * 4)

    # Get results from both retrievers
    vector_results = vector_retriever.invoke(query)
    bm25_results = bm25_retriever.invoke(query)

    logger.info(
        f"🔍 Hybrid search components | "
        f"vector: {len(vector_results)} docs, "
        f"bm25: {len(bm25_results)} docs"
    )

    # Combine using RRF with specified weights
    combined_docs = reciprocal_rank_fusion(
        results_list=[vector_results, bm25_results],
        weights=[vector_weight, bm25_weight],
        k=rrf_k,
        top_k=top_k
    )

    logger.info(
        f"✅ Hybrid search: {int(vector_weight*100)}% semantic + "
        f"{int(bm25_weight*100)}% keyword | "
        f"retrieved {len(combined_docs)} docs "
        f"(weights from: {'env' if vector_weight == settings.HYBRID_SEARCH_VECTOR_WEIGHT else 'override'})"
    )

    return combined_docs


async def hybrid_search(query: str, k: int = 10) -> list[Document]:
    """Perform hybrid search combining semantic and keyword retrieval.

    Uses custom RRF (Reciprocal Rank Fusion) implementation for LangChain v1.x compatibility.
    Weights are configured via environment variables (HYBRID_SEARCH_VECTOR_WEIGHT, HYBRID_SEARCH_BM25_WEIGHT).

    Args:
        query: Search query
        k: Number of documents to return (default: 10)

    Returns:
        list[Document]: Retrieved documents ranked by hybrid score

    Example:
        >>> # Uses env defaults (HYBRID_SEARCH_VECTOR_WEIGHT=0.7, HYBRID_SEARCH_BM25_WEIGHT=0.3)
        >>> docs = await hybrid_search("What is a Category 5 hurricane?")
        >>> for doc in docs:
        ...     print(doc.page_content[:100])

    Configuration:
        Set in .env file:
        HYBRID_SEARCH_VECTOR_WEIGHT=0.7  # 70% semantic (default)
        HYBRID_SEARCH_BM25_WEIGHT=0.3    # 30% keyword (default)

    Note:
        Uses asyncio.to_thread() to run blocking Qdrant/OpenAI calls in a separate thread,
        preventing event loop blocking in ASGI servers (FastAPI, LangGraph).
    """
    import asyncio

    # Run blocking hybrid search in a separate thread to avoid blocking the event loop
    # This prevents "Blocking call to socket.socket.connect" warnings in LangGraph
    docs = await asyncio.to_thread(
        hybrid_search_with_rrf,
        query=query,
        vector_weight=None,  # Use env HYBRID_SEARCH_VECTOR_WEIGHT
        bm25_weight=None,    # Use env HYBRID_SEARCH_BM25_WEIGHT
        top_k=k,
        rrf_k=60             # RRF constant
    )

    logger.info(
        f"🔍 Hybrid search complete | query='{query[:50]}...' | retrieved {len(docs)} docs"
    )

    return docs


def compare_search_methods(query: str, k: int = 5) -> dict:
    """Compare dense-only vs sparse-only vs hybrid search for analysis.

    Useful for debugging and understanding which search method works best
    for different query types.

    Args:
        query: Search query
        k: Number of results per method

    Returns:
        dict: Results from each search method with overlap analysis

    Example:
        >>> results = compare_search_methods("Saffir-Simpson scale")
        >>> print(f"Hybrid found {results['hybrid']['count']} unique docs")
    """
    vectorstore = get_vector_store()

    # Dense-only (semantic)
    dense_docs = vectorstore.similarity_search(query, k=k)

    # Sparse-only (keyword)
    all_docs = get_all_documents_from_vectorstore()
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k
    sparse_docs = bm25_retriever.invoke(query)

    # Hybrid (RRF combined) - LangChain v1.x compatible
    hybrid_docs = hybrid_search_with_rrf(query=query, top_k=k)

    # Calculate overlap
    dense_ids = set(doc.page_content[:100] for doc in dense_docs)
    sparse_ids = set(doc.page_content[:100] for doc in sparse_docs)
    hybrid_ids = set(doc.page_content[:100] for doc in hybrid_docs)

    overlap_dense_sparse = len(dense_ids & sparse_ids)
    overlap_hybrid_dense = len(hybrid_ids & dense_ids)
    overlap_hybrid_sparse = len(hybrid_ids & sparse_ids)

    return {
        "query": query,
        "dense": {
            "count": len(dense_docs),
            "docs": [doc.page_content[:200] for doc in dense_docs]
        },
        "sparse": {
            "count": len(sparse_docs),
            "docs": [doc.page_content[:200] for doc in sparse_docs]
        },
        "hybrid": {
            "count": len(hybrid_docs),
            "docs": [doc.page_content[:200] for doc in hybrid_docs]
        },
        "overlap": {
            "dense_sparse": overlap_dense_sparse,
            "hybrid_dense": overlap_hybrid_dense,
            "hybrid_sparse": overlap_hybrid_sparse
        }
    }
