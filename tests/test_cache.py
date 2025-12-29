"""Tests for query result caching."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from memex_twos_mcp.cache import QueryCache
from memex_twos_mcp.database import TwosDatabase


def test_cache_hit_miss():
    """Test basic cache hit and miss behavior."""
    cache = QueryCache(ttl_seconds=60)

    # First access - cache miss
    result = cache.get("test query", limit=10)
    assert result is None

    # Set value
    cache.set("test query", ["result1", "result2"], limit=10)

    # Second access - cache hit
    result = cache.get("test query", limit=10)
    assert result == ["result1", "result2"]

    # Stats should show 1 hit, 1 miss
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["cache_size"] == 1


def test_cache_ttl_expiration():
    """Test that cache entries expire after TTL."""
    cache = QueryCache(ttl_seconds=1)  # 1 second TTL

    cache.set("test query", ["result1"], limit=10)

    # Immediate access - should hit
    result = cache.get("test query", limit=10)
    assert result == ["result1"]

    # Wait for expiration
    time.sleep(1.1)

    # Access after expiration - should miss
    result = cache.get("test query", limit=10)
    assert result is None

    # Cache should be empty (expired entry removed)
    stats = cache.get_stats()
    assert stats["cache_size"] == 0


def test_cache_key_normalization():
    """Test that cache keys are normalized correctly."""
    cache = QueryCache(ttl_seconds=60)

    # Different whitespace/case should produce same key
    cache.set("  TEST Query  ", ["result1"], limit=10)

    result = cache.get("test query", limit=10)
    assert result == ["result1"]


def test_cache_invalidation():
    """Test that cache can be fully invalidated."""
    cache = QueryCache(ttl_seconds=60)

    cache.set("query1", ["result1"], limit=10)
    cache.set("query2", ["result2"], limit=20)

    assert cache.get_stats()["cache_size"] == 2

    # Invalidate all
    cache.invalidate_all()

    assert cache.get_stats()["cache_size"] == 0
    assert cache.get("query1", limit=10) is None
    assert cache.get("query2", limit=20) is None

    # Stats should be reset
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 2


def _init_test_db(db_path: Path, schema_path: Path) -> None:
    """Initialize test database with sample data."""
    schema_sql = schema_path.read_text(encoding="utf-8")
    conn = sqlite3.connect(db_path)
    conn.executescript(schema_sql)

    # Insert test things
    for i in range(5):
        conn.execute(
            """
            INSERT INTO things (
                id, timestamp, content, section_header,
                bullet_type, is_completed, is_pending, is_strikethrough
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"task_{i:05d}",
                f"2024-01-{i+1:02d}T10:00:00",
                f"Thing {i} with doctor keyword",
                f"Day {i}",
                "bullet",
                0,
                0,
                0,
            ),
        )

    conn.commit()
    conn.close()


def test_database_cache_integration(tmp_path: Path) -> None:
    """Test that database caching works end-to-end."""
    db_path = tmp_path / "twos.db"
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "schema.sql"

    _init_test_db(db_path, schema_path)

    db = TwosDatabase(db_path, cache_ttl_seconds=60)

    # First search - cache miss
    results1 = db.search_candidates("doctor", limit=5)
    assert len(results1) == 5

    # Check cache stats
    stats = db.get_cache_stats()
    assert stats["cache_size"] == 1
    assert stats["hits"] == 0
    assert stats["misses"] == 1

    # Second identical search - cache hit
    results2 = db.search_candidates("doctor", limit=5)
    assert results2 == results1  # Exact same results

    # Check cache stats
    stats = db.get_cache_stats()
    assert stats["cache_size"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate_percent"] == 50.0

    # Different limit - different cache key - cache miss
    results3 = db.search_candidates("doctor", limit=3)
    assert len(results3) == 3

    # Check cache stats
    stats = db.get_cache_stats()
    assert stats["cache_size"] == 2  # Two different cache entries
    assert stats["hits"] == 1
    assert stats["misses"] == 2

    # Clean up
    db.close()


def test_connection_pooling(tmp_path: Path) -> None:
    """Test that connection pooling reuses connections."""
    db_path = tmp_path / "twos.db"
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "schema.sql"

    _init_test_db(db_path, schema_path)

    db = TwosDatabase(db_path)

    # Get connection multiple times
    conn1 = db._get_connection()
    conn2 = db._get_connection()

    # Should be the same connection object
    assert conn1 is conn2

    # Connection should persist across queries
    db.search_candidates("doctor", limit=5)
    conn3 = db._get_connection()
    assert conn1 is conn3

    # Close should None out the connection
    db.close()
    assert db._connection is None

    # Next query should create new connection
    db.search_candidates("doctor", limit=5)
    conn4 = db._get_connection()
    assert conn4 is not None
    assert conn4 is not conn1  # New connection after close

    db.close()
