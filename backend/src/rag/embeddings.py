"""OpenAI embeddings configuration for RAG system.

Uses text-embedding-3-small for cost-effective semantic search:
- Model: text-embedding-3-small (1536 dimensions)
- Cost: $0.02 per 1M tokens (62% cheaper than ada-002)
- Performance: Comparable to ada-002 on MTEB benchmarks
- Use case: Weather knowledge base (500+ docs, <$1/month)

Level 2 Configuration:
- Default model: text-embedding-3-small
- No dimensionality reduction (full 1536-d vectors)
- Standard embedding parameters

Future Optimizations (L5a):
- Dimensionality reduction to 512-d (50% cost reduction)
- Batch embedding for large document sets
- Cache frequently embedded queries
"""

from langchain_openai import OpenAIEmbeddings

from backend.config.settings import settings


def create_embeddings(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """Create OpenAI embeddings instance for RAG.

    Args:
        model: OpenAI embedding model name (default: text-embedding-3-small)
            - text-embedding-3-small: 1536-d, $0.02/1M tokens (RECOMMENDED for L2)
            - text-embedding-3-large: 3072-d, $0.13/1M tokens (defer to L5a)
            - text-embedding-ada-002: 1536-d, $0.10/1M tokens (legacy, avoid)

    Returns:
        OpenAIEmbeddings: Configured embeddings instance

    Example:
        >>> embeddings = create_embeddings()
        >>> vector = embeddings.embed_query("What is a Category 5 hurricane?")
        >>> len(vector)  # 1536
        1536

    Cost Analysis (500 docs @ 400 tokens avg = 200K tokens):
        - text-embedding-3-small: $0.004 (RECOMMENDED)
        - text-embedding-3-large: $0.026
        - ada-002: $0.020

    Note:
        Requires OPENAI_API_KEY in .env file.
        Uses centralized settings from backend.config.settings.
        Uses default OpenAI client configuration (no custom parameters for L2).
    """
    # API key loaded automatically from settings
    # (Pydantic Settings validates and loads from .env)

    # Create embeddings instance with L2 configuration
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=settings.OPENAI_API_KEY,
        # Level 2: No custom parameters
        # Level 5a will add:
        # - dimensions=512 (dimensionality reduction)
        # - show_progress_bar=True (for large batches)
        # - chunk_size=100 (batch optimization)
    )
