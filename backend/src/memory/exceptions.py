"""Custom exceptions for memory operations.

Provides specific exception types for Redis and Graphiti errors.
"""


class MemoryStoreError(Exception):
    """Base exception for memory store operations.

    Raised when Redis or Graphiti operations fail.

    Examples:
        >>> raise MemoryStoreError("Redis connection failed")
        >>> raise MemoryStoreError("Graphiti episode ingestion failed")
    """

    pass


class RedisMemoryError(MemoryStoreError):
    """Exception for Redis-specific failures.

    Raised when Redis operations fail (connection, get, set, delete).

    Examples:
        >>> raise RedisMemoryError("Failed to save context: Connection timeout")
        >>> raise RedisMemoryError("Failed to retrieve entity: Key not found")
    """

    pass


class GraphitiMemoryError(MemoryStoreError):
    """Exception for Graphiti-specific failures.

    Raised when Graphiti operations fail (add_episode, search, initialization).

    Examples:
        >>> raise GraphitiMemoryError("Failed to add episode: Neo4j connection lost")
        >>> raise GraphitiMemoryError("Failed to search: Invalid query syntax")
    """

    pass
