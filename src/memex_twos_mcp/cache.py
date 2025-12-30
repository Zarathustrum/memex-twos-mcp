"""
Query result cache with TTL (Time-To-Live) support.

Provides in-memory caching for expensive database queries with automatic expiration.
"""

import hashlib
import json
import threading
import time
from typing import Any, Dict, Optional, Tuple


class QueryCache:
    """
    TTL-based cache for query results.

    Thread-safe caching with automatic expiration for database query results.
    Primarily used for caching search candidate results to reduce repeated FTS queries.
    """

    def __init__(self, ttl_seconds: int = 900):
        """
        Initialize query cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default: 900 = 15 minutes)
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, **kwargs) -> str:
        """
        Create deterministic cache key from query and parameters.

        Args:
            query: Search query string
            **kwargs: Additional parameters (limit, filters, etc.)

        Returns:
            SHA256 hash of normalized query + params
        """
        # Normalize query to lowercase and strip whitespace
        normalized = {"q": query.lower().strip(), **kwargs}
        # Sort keys for deterministic ordering
        key_string = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, query: str, **kwargs) -> Optional[Any]:
        """
        Retrieve cached result if not expired.

        Args:
            query: Search query string
            **kwargs: Additional parameters that were used in original query

        Returns:
            Cached result if found and not expired, None otherwise
        """
        key = self._make_key(query, **kwargs)

        with self._lock:
            if key in self._cache:
                timestamp, result = self._cache[key]

                # Check if expired
                if time.time() - timestamp < self.ttl_seconds:
                    self._hits += 1
                    return result
                else:
                    # Expired - remove from cache
                    del self._cache[key]
                    self._misses += 1
            else:
                self._misses += 1

        return None

    def set(self, query: str, result: Any, **kwargs):
        """
        Cache query result with current timestamp.

        Args:
            query: Search query string
            result: Query result to cache
            **kwargs: Additional parameters that were used in query
        """
        key = self._make_key(query, **kwargs)

        with self._lock:
            self._cache[key] = (time.time(), result)

    def invalidate_all(self):
        """
        Clear entire cache.

        Call this when data changes (insert, update, delete) to ensure cache consistency.
        """
        with self._lock:
            self._cache.clear()
            # Reset hit/miss counters on full invalidation
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache size, hit rate, and timing info
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "cache_size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2),
                "ttl_seconds": self.ttl_seconds,
            }
