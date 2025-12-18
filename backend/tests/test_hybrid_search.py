"""
Unit tests for hybrid search functionality (Level 2).

Tests cover:
- Hybrid search initialization
- Semantic search
- Keyword search
- Score fusion (RRF)
- Error handling
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from backend.src.rag.hybrid_search import (
    get_all_documents_from_vectorstore,
    hybrid_search,
    reciprocal_rank_fusion,
)


@pytest.fixture
def sample_semantic_results():
    """Sample semantic search results."""
    return [
        Document(page_content="Category 5 hurricanes have winds 157+ mph", metadata={"source": "hurricane_facts.txt"}),
        Document(page_content="Saffir-Simpson scale defines 5 categories", metadata={"source": "scales.txt"}),
    ]


@pytest.fixture
def sample_keyword_results():
    """Sample keyword search results."""
    return [
        Document(page_content="Category 5 hurricanes have winds 157+ mph", metadata={"source": "hurricane_facts.txt"}),
        Document(page_content="Hurricane wind speeds determine category", metadata={"source": "wind.txt"}),
    ]


def test_rrf_with_overlapping_docs(sample_semantic_results, sample_keyword_results):
    """Test RRF fusion with overlapping documents."""
    fused = reciprocal_rank_fusion(
        semantic_results=sample_semantic_results,
        keyword_results=sample_keyword_results,
        k=60
    )

    # Should have 3 unique documents (1 overlaps)
    assert len(fused) == 3

    # Overlapping doc should rank highest
    assert fused[0].page_content == "Category 5 hurricanes have winds 157+ mph"

    # All docs should have RRF scores
    for doc in fused:
        assert "rrf_score" in doc.metadata
        assert isinstance(doc.metadata["rrf_score"], float)
        assert doc.metadata["rrf_score"] > 0


def test_rrf_with_no_overlap():
    """Test RRF fusion with no overlapping documents."""
    semantic_results = [
        Document(page_content="Doc A", metadata={"source": "a.txt"}),
        Document(page_content="Doc B", metadata={"source": "b.txt"}),
    ]
    keyword_results = [
        Document(page_content="Doc C", metadata={"source": "c.txt"}),
        Document(page_content="Doc D", metadata={"source": "d.txt"}),
    ]

    fused = reciprocal_rank_fusion(semantic_results, keyword_results, k=60)

    # Should have all 4 documents
    assert len(fused) == 4

    # All docs should have scores
    for doc in fused:
        assert "rrf_score" in doc.metadata


def test_rrf_with_empty_semantic():
    """Test RRF when semantic results are empty."""
    keyword_results = [
        Document(page_content="Doc C", metadata={"source": "c.txt"}),
    ]

    fused = reciprocal_rank_fusion([], keyword_results, k=60)

    assert len(fused) == 1
    assert fused[0].page_content == "Doc C"


def test_rrf_with_empty_keyword():
    """Test RRF when keyword results are empty."""
    semantic_results = [
        Document(page_content="Doc A", metadata={"source": "a.txt"}),
    ]

    fused = reciprocal_rank_fusion(semantic_results, [], k=60)

    assert len(fused) == 1
    assert fused[0].page_content == "Doc A"


def test_rrf_with_both_empty():
    """Test RRF when both result lists are empty."""
    fused = reciprocal_rank_fusion([], [], k=60)
    assert len(fused) == 0


def test_rrf_score_decreases_with_rank():
    """Test that RRF scores decrease as rank increases."""
    semantic_results = [
        Document(page_content=f"Doc {i}", metadata={"source": f"{i}.txt"})
        for i in range(5)
    ]

    fused = reciprocal_rank_fusion(semantic_results, [], k=60)

    # Scores should decrease
    for i in range(len(fused) - 1):
        assert fused[i].metadata["rrf_score"] >= fused[i+1].metadata["rrf_score"]


def test_rrf_k_parameter():
    """Test that RRF k parameter affects scores."""
    semantic_results = [Document(page_content="Doc A", metadata={"source": "a.txt"})]

    # Higher k should result in lower scores (more documents in denominator)
    fused_k60 = reciprocal_rank_fusion(semantic_results, [], k=60)
    fused_k100 = reciprocal_rank_fusion(semantic_results, [], k=100)

    assert fused_k60[0].metadata["rrf_score"] > fused_k100[0].metadata["rrf_score"]


@pytest.mark.asyncio
@patch("backend.src.rag.hybrid_search.get_vector_store")
@patch("backend.src.rag.hybrid_search.get_all_documents_from_vectorstore")
async def test_hybrid_search_success(mock_get_docs, mock_get_vector_store):
    """Test successful hybrid search execution."""
    # Mock vector store
    mock_vectorstore = Mock()
    mock_vectorstore.similarity_search_with_score = Mock(return_value=[
        (Document(page_content="Semantic result", metadata={"source": "sem.txt"}), 0.95),
    ])
    mock_get_vector_store.return_value = mock_vectorstore

    # Mock BM25 documents
    mock_get_docs.return_value = [
        Document(page_content="Keyword result", metadata={"source": "key.txt"}),
    ]

    # Patch BM25Retriever
    with patch("backend.src.rag.hybrid_search.BM25Retriever") as mock_bm25_class:
        mock_bm25 = Mock()
        mock_bm25.invoke = Mock(return_value=[
            Document(page_content="Keyword result", metadata={"source": "key.txt"}),
        ])
        mock_bm25_class.from_documents.return_value = mock_bm25

        results = await hybrid_search("test query", k=10)

        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc in results)
        assert all("rrf_score" in doc.metadata for doc in results)


@pytest.mark.asyncio
@patch("backend.src.rag.hybrid_search.get_vector_store")
@patch("backend.src.rag.hybrid_search.get_all_documents_from_vectorstore")
async def test_hybrid_search_empty_query(mock_get_docs, mock_get_vector_store):
    """Test hybrid search with empty query."""
    mock_vectorstore = Mock()
    mock_vectorstore.similarity_search_with_score = Mock(return_value=[])
    mock_get_vector_store.return_value = mock_vectorstore

    mock_get_docs.return_value = []

    with patch("backend.src.rag.hybrid_search.BM25Retriever") as mock_bm25_class:
        mock_bm25 = Mock()
        mock_bm25.invoke = Mock(return_value=[])
        mock_bm25_class.from_documents.return_value = mock_bm25

        results = await hybrid_search("", k=10)
        assert isinstance(results, list)


@patch("backend.src.rag.hybrid_search.get_vector_store")
@patch("backend.src.rag.hybrid_search.QdrantClient")
def test_get_all_documents_success(mock_qdrant_client, mock_get_vector_store):
    """Test successful retrieval of all documents from vector store."""
    # Mock Qdrant client
    mock_client = Mock()
    mock_point = Mock()
    mock_point.id = "doc1"
    mock_point.payload = {
        "page_content": "Test document",
        "metadata": {"source": "test.txt"}
    }
    mock_client.scroll.return_value = ([mock_point], None)
    mock_qdrant_client.return_value = mock_client

    # Mock vector store
    mock_vectorstore = Mock()
    mock_vectorstore.collection_name = "test_collection"
    mock_get_vector_store.return_value = mock_vectorstore

    with patch("backend.src.rag.hybrid_search.settings") as mock_settings:
        mock_settings.QDRANT_URL = "http://localhost:6333"

        docs = get_all_documents_from_vectorstore()

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "Test document"
        assert docs[0].metadata["source"] == "test.txt"


@patch("backend.src.rag.hybrid_search.get_vector_store")
@patch("backend.src.rag.hybrid_search.QdrantClient")
def test_get_all_documents_empty(mock_qdrant_client, mock_get_vector_store):
    """Test retrieval when vector store is empty."""
    mock_client = Mock()
    mock_client.scroll.return_value = ([], None)
    mock_qdrant_client.return_value = mock_client

    mock_vectorstore = Mock()
    mock_vectorstore.collection_name = "test_collection"
    mock_get_vector_store.return_value = mock_vectorstore

    with patch("backend.src.rag.hybrid_search.settings") as mock_settings:
        mock_settings.QDRANT_URL = "http://localhost:6333"

        docs = get_all_documents_from_vectorstore()
        assert len(docs) == 0


def test_rrf_deduplication_by_content():
    """Test that RRF deduplicates documents with identical content."""
    duplicate_content = "Category 5 hurricanes have winds 157+ mph"

    semantic_results = [
        Document(page_content=duplicate_content, metadata={"source": "file1.txt"}),
    ]
    keyword_results = [
        Document(page_content=duplicate_content, metadata={"source": "file2.txt"}),  # Different metadata
    ]

    fused = reciprocal_rank_fusion(semantic_results, keyword_results, k=60)

    # Should have only 1 document (deduplicated by content)
    assert len(fused) == 1
    assert fused[0].page_content == duplicate_content


def test_rrf_preserves_original_metadata():
    """Test that RRF preserves original document metadata."""
    semantic_results = [
        Document(
            page_content="Test doc",
            metadata={"source": "test.txt", "custom_field": "value123"}
        ),
    ]

    fused = reciprocal_rank_fusion(semantic_results, [], k=60)

    assert fused[0].metadata["source"] == "test.txt"
    assert fused[0].metadata["custom_field"] == "value123"
    assert "rrf_score" in fused[0].metadata
