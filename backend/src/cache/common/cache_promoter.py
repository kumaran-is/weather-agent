"""Cache Promoter for Tier 2 → Tier 1 backfill.

Level 9a: Handles promotion of semantic cache hits to exact match cache.

Architecture:
    When a Tier 2 (semantic) hit occurs, we "promote" the result to Tier 1 (exact)
    so that identical future queries get the faster exact match response.

Benefits:
    - Tier 2 hit → Tier 1 backfill → Future queries hit Tier 1
    - Reduces embedding API calls over time
    - Improves latency for repeated queries

Usage:
    from backend.src.cache.common import CachePromoter

    promoter = CachePromoter()

    # On Tier 2 hit, promote to Tier 1
    await promoter.promote(
        redis=redis_client,
        cache_key="query:abc123",
        value="serialized_response",
        ttl=1800,
    )
"""

import logging

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class CachePromoter:
    """Handles promotion from Tier 2 (semantic) to Tier 1 (exact match).

    When a semantic cache hit occurs, the result is backfilled to the
    exact match cache for faster future lookups on identical queries.

    Promotion Strategy:
        1. Tier 2 hit detected (semantic match)
        2. Extract value from Qdrant payload
        3. Write to Redis with original cache key
        4. Future identical queries hit Tier 1 directly

    Metrics:
        - promotions_count: Total successful promotions
        - promotion_errors: Failed promotion attempts
    """

    def __init__(self):
        """Initialize cache promoter."""
        self.promotions_count = 0
        self.promotion_errors = 0

        logger.info("✅ CachePromoter initialized")

    async def promote(
        self,
        redis: redis.Redis,
        cache_key: str,
        value: str,
        ttl: int = 1800,
    ) -> bool:
        """Promote value from Tier 2 to Tier 1.

        Args:
            redis: Async Redis client
            cache_key: Tier 1 cache key
            value: Serialized value to store
            ttl: Time-to-live in seconds

        Returns:
            True if promotion successful, False otherwise
        """
        try:
            await redis.setex(cache_key, ttl, value)
            self.promotions_count += 1

            logger.debug(
                f"⬆️ Promoted to Tier 1 | key={cache_key[:16]}... | ttl={ttl}s"
            )

            return True

        except Exception as e:
            self.promotion_errors += 1
            logger.warning(f"⚠️ Promotion failed | key={cache_key[:16]}... | error={e}")
            return False

    async def promote_batch(
        self,
        redis: redis.Redis,
        entries: list[tuple[str, str, int]],
    ) -> dict[str, bool]:
        """Promote multiple entries in a batch.

        Args:
            redis: Async Redis client
            entries: List of (cache_key, value, ttl) tuples

        Returns:
            Dict mapping cache_key to success status
        """
        results = {}

        async with redis.pipeline(transaction=True) as pipe:
            for cache_key, value, ttl in entries:
                pipe.setex(cache_key, ttl, value)

            try:
                await pipe.execute()
                for cache_key, _, _ in entries:
                    results[cache_key] = True
                    self.promotions_count += 1

                logger.debug(f"⬆️ Batch promoted {len(entries)} entries to Tier 1")

            except Exception as e:
                for cache_key, _, _ in entries:
                    results[cache_key] = False
                    self.promotion_errors += 1

                logger.warning(f"⚠️ Batch promotion failed | error={e}")

        return results

    def get_stats(self) -> dict[str, int]:
        """Get promotion statistics.

        Returns:
            Dict with promotion counts
        """
        return {
            "promotions_count": self.promotions_count,
            "promotion_errors": self.promotion_errors,
            "promotion_success_rate": (
                self.promotions_count / max(1, self.promotions_count + self.promotion_errors)
            ),
        }

    def reset_stats(self) -> None:
        """Reset promotion statistics."""
        self.promotions_count = 0
        self.promotion_errors = 0
