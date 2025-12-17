"""Shared test fixtures and mocks for unit tests."""

import pytest
from unittest.mock import patch


class MockOpenAIEmbeddings:
    """Mock OpenAI embeddings that behaves like the real thing."""

    def __init__(self, *args, **kwargs):
        """Initialize mock embeddings."""
        pass

    def __call__(self, texts):
        """Make the embeddings object callable (for direct invocation)."""
        return self.embed_documents(texts)

    def embed_documents(self, texts):
        """Return one embedding vector per input text."""
        if isinstance(texts, str):
            return [[0.1] * 1536]
        return [[0.1] * 1536 for _ in texts]

    def embed_query(self, text):
        """Return single embedding vector."""
        return [0.1] * 1536

    async def aembed_documents(self, texts):
        """Async version of embed_documents."""
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        """Async version of embed_query."""
        return self.embed_query(text)


@pytest.fixture(autouse=True)
def mock_openai_embeddings():
    """Mock OpenAI embeddings for all tests to avoid API calls."""
    with patch("backend.src.registry.bigtool_registry.OpenAIEmbeddings", MockOpenAIEmbeddings):
        yield


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset BigtoolRegistry singleton before each test for proper isolation."""
    from backend.src.registry import reset_bigtool_registry

    # Reset before each test
    reset_bigtool_registry()
    yield
    # Reset after each test (cleanup)
    reset_bigtool_registry()
