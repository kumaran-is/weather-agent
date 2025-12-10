"""Memory system configuration for Level 3a.

CRITICAL RULE: ALL config in backend/config/ (NOT backend/src/config/)

This configuration module centralizes all memory-related settings:
- Redis (short-term memory)
- Graphiti + Neo4j (long-term memory)
- Memory behavior settings

Environment Variables Required:
- REDIS_URL (default: redis://localhost:6379/0)
- GRAPHITI_URL (default: bolt://localhost:7687)
- GRAPHITI_USER (default: neo4j)
- GRAPHITI_PASSWORD (required)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class MemoryConfig(BaseSettings):
    """Memory system configuration from environment variables.

    Example .env:
        REDIS_URL=redis://localhost:6379/0
        REDIS_TTL_SECONDS=1800
        GRAPHITI_URL=bolt://localhost:7687
        GRAPHITI_USER=neo4j
        GRAPHITI_PASSWORD=your_password
        GRAPHITI_DATABASE=weather_ai
        ENABLE_MEMORY=true
    """

    # Redis (Short-term memory)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL_SECONDS: int = 1800  # 30 minutes
    REDIS_MAX_CONVERSATION_TURNS: int = 10
    REDIS_MAX_CONNECTIONS: int = 50

    # Graphiti (Long-term memory)
    GRAPHITI_URL: str = "bolt://localhost:7687"
    GRAPHITI_USER: str = "neo4j"
    GRAPHITI_PASSWORD: str = "password"
    GRAPHITI_DATABASE: str = "weather_ai"

    # Memory behavior
    ENABLE_MEMORY: bool = True
    ENABLE_ENTITY_TRACKING: bool = True
    ENABLE_PRONOUN_RESOLUTION: bool = True
    ENABLE_USER_PROFILES: bool = True

    # Performance tuning
    MEMORY_CONTEXT_WINDOW_SIZE: int = 10  # Number of turns to keep
    MEMORY_TOKEN_BUDGET: int = 1000  # Max tokens for memory context

    # 🆕 P1 FIX: Parallel retrieval configuration (2-3x faster memory operations)
    MEMORY_PARALLEL_RETRIEVAL: bool = True  # Enable parallel layer retrieval
    MEMORY_LAYER_TIMEOUT: float = 5.0  # Timeout per layer in seconds (prevents hangs)

    # 🆕 Future: Multi-layer memory configuration (Level 3c - 7 layers)
    # These flags control which memory layers are active
    # Currently only Layers 1-3 are implemented:
    #   - Layer 1: Short-term (Redis) - always enabled
    #   - Layer 2: Long-term user profiles (Graphiti) - ENABLE_USER_PROFILES
    #   - Layer 3: Episodic memory (Graphiti) - ENABLE_USER_PROFILES
    #
    # Future layers (to be implemented):
    ENABLE_SEMANTIC_MEMORY: bool = False  # Layer 4: Domain knowledge (Qdrant RAG)
    ENABLE_PROCEDURAL_MEMORY: bool = False  # Layer 5: Tool usage patterns (PostgreSQL)
    ENABLE_EMOTIONAL_MEMORY: bool = False  # Layer 6: Sentiment/anxiety (Redis)
    ENABLE_REFLECTIVE_MEMORY: bool = False  # Layer 7: Self-improvement (PostgreSQL)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Global configuration instance
memory_config = MemoryConfig()
