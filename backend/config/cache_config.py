"""Cache configuration for Weather AI Agent.

This module provides configuration for all three cache layers:
- L1: In-process LRU cache
- L2: Redis distributed cache
- L3: Anthropic prompt cache

Configuration is loaded from environment variables with sensible defaults.
"""

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class CacheConfig(BaseSettings):
    """Cache configuration from environment variables.

    Environment Variables:
    - CACHE_ENABLED: Enable/disable all caching (default: True)
    - L1_CACHE_ENABLED: Enable/disable L1 cache (default: True)
    - L1_CACHE_MAX_SIZE: L1 cache max entries (default: 1000)
    - L1_CACHE_TTL_SECONDS: L1 cache TTL (default: 300 = 5 minutes)
    - L2_CACHE_ENABLED: Enable/disable L2 cache (default: True)
    - L2_CACHE_REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
    - L2_CACHE_TTL_SECONDS: L2 cache TTL (default: 1800 = 30 minutes)
    - L2_CACHE_KEY_PREFIX: Redis key prefix (default: weather:cache:)
    - L3_CACHE_ENABLED: Enable/disable L3 cache (default: True)
    - L3_CACHE_RECENT_TURNS: Cache recent N conversation turns (default: 0)
    """

    # Global cache toggle
    CACHE_ENABLED: bool = True

    # L1: In-process LRU cache
    L1_CACHE_ENABLED: bool = True
    L1_CACHE_MAX_SIZE: int = 1000
    L1_CACHE_TTL_SECONDS: int = 300  # 5 minutes

    # L2: Redis distributed cache
    L2_CACHE_ENABLED: bool = True
    L2_CACHE_REDIS_URL: str = "redis://localhost:6379/0"
    L2_CACHE_TTL_SECONDS: int = 1800  # 30 minutes
    L2_CACHE_KEY_PREFIX: str = "weather:cache:"

    # L3: Anthropic prompt cache
    L3_CACHE_ENABLED: bool = True
    L3_CACHE_RECENT_TURNS: int = 0  # 0 = don't cache messages, only system/tools

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    def model_post_init(self, __context):
        """Log configuration after initialization."""
        logger.info("✅ Cache configuration loaded:")
        logger.info(f"  - Global: {'ENABLED' if self.CACHE_ENABLED else 'DISABLED'}")
        logger.info(f"  - L1 (Memory): {'ENABLED' if self.L1_CACHE_ENABLED else 'DISABLED'} (max_size={self.L1_CACHE_MAX_SIZE}, ttl={self.L1_CACHE_TTL_SECONDS}s)")
        logger.info(f"  - L2 (Redis): {'ENABLED' if self.L2_CACHE_ENABLED else 'DISABLED'} (url={self.L2_CACHE_REDIS_URL}, ttl={self.L2_CACHE_TTL_SECONDS}s)")
        logger.info(f"  - L3 (Anthropic): {'ENABLED' if self.L3_CACHE_ENABLED else 'DISABLED'}")


# Global cache configuration instance
cache_config = CacheConfig()
