"""
Database wrapper for Twos SQLite database.
Provides safe query methods for MCP tools.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cache import QueryCache


class TwosDatabase:
    """Wrapper for querying the Twos SQLite database."""

    def __init__(self, db_path: Path, cache_ttl_seconds: int = 900):
        """
        Initialize database wrapper with connection pooling and caching.

        Args:
            db_path: Path to the SQLite database file on disk.
            cache_ttl_seconds: TTL for query cache in seconds (default: 900 = 15 minutes)

        Raises:
            FileNotFoundError: If the database file does not exist.
        """
        self.db_path = db_path
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        # Connection pooling (single persistent connection per instance)
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

        # Query result cache
        self.cache = QueryCache(ttl_seconds=cache_ttl_seconds)

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a persistent database connection with row factory (connection pooling).

        Returns a single shared connection instead of creating new connections per query.
        Thread-safe via locking.

        Returns:
            A sqlite3.Connection that yields rows as dict-like objects.

        Side effects:
            Opens a file-backed SQLite connection on first call.
        """
        with self._lock:
            if self._connection is None:
                self._connection = sqlite3.connect(
                    self.db_path, check_same_thread=False  # Allow multi-thread with lock
                )
                self._connection.row_factory = sqlite3.Row
            return self._connection

    def close(self):
        """
        Close the persistent database connection.

        Call this for graceful shutdown. Connection will be reopened
        automatically on next query if needed.
        """
        with self._lock:
            if self._connection:
                self._connection.close()
                self._connection = None

    def query_tasks_by_date(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query things within a date range.

        Args:
            start_date: ISO format date string (YYYY-MM-DD)
            end_date: ISO format date string (YYYY-MM-DD)
            limit: Maximum number of results

        Returns:
            List of thing dictionaries
        """
        # Open a new connection for this query to avoid cross-request state.
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build the SQL query incrementally based on optional filters.
        query = "SELECT * FROM things WHERE 1=1"
        params: list[object] = []

        if start_date:
            # SQLite DATE(...) extracts date-only for comparison.
            query += " AND DATE(timestamp) >= ?"
            params.append(start_date)

        if end_date:
            query += " AND DATE(timestamp) <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        # Parameterized query avoids SQL injection risks.
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        # Note: Connection is persistent, no need to close

        return results

    def query_tasks_by_date_candidates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query things within a date range and return minimal candidate previews.

        This returns only essential fields for LLM preview (two-phase retrieval):
        - id, timestamp, is_completed
        - content_preview (truncated to 100 chars)
        - tags, people

        Results are ~75% smaller than full records. Use get_things_by_ids() to
        fetch full content for selected candidates.

        Args:
            start_date: ISO format date string (YYYY-MM-DD)
            end_date: ISO format date string (YYYY-MM-DD)
            limit: Maximum number of candidates to return

        Returns:
            List of candidate dictionaries with minimal fields
        """
        # Check cache first
        cache_key_parts = {"start": start_date, "end": end_date, "limit": limit}
        cached = self.cache.get(
            f"date_query:{start_date}:{end_date}", **cache_key_parts
        )
        if cached is not None:
            return cached

        conn = self._get_connection()
        cursor = conn.cursor()

        # Build query with minimal SELECT fields
        query = """
            SELECT
                id,
                timestamp,
                is_completed,
                SUBSTR(content, 1, 100) AS content_preview
            FROM things
            WHERE 1=1
        """
        params: list[object] = []

        if start_date:
            query += " AND DATE(timestamp) >= ?"
            params.append(start_date)

        if end_date:
            query += " AND DATE(timestamp) <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            thing_id = result["id"]

            # Fetch tags for this thing
            cursor.execute(
                """
                SELECT t.name
                FROM tags t
                JOIN thing_tags tt ON t.id = tt.tag_id
                WHERE tt.thing_id = ?
            """,
                (thing_id,),
            )
            result["tags"] = [r[0] for r in cursor.fetchall()]

            # Fetch people for this thing
            cursor.execute(
                """
                SELECT p.name
                FROM people p
                JOIN thing_people tp ON p.id = tp.person_id
                WHERE tp.thing_id = ?
            """,
                (thing_id,),
            )
            result["people"] = [r[0] for r in cursor.fetchall()]

            results.append(result)

        # Cache the results
        self.cache.set(f"date_query:{start_date}:{end_date}", results, **cache_key_parts)

        return results

    def search_content(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Full-text search across thing content with BM25 relevance ranking.

        Args:
            query: Search query (FTS5 syntax)
            limit: Maximum number of results

        Returns:
            List of matching thing dictionaries with relevance_score and snippet

        Raises:
            ValueError: If the FTS5 query syntax is invalid
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Use BM25 ranking (bm25() returns negative scores, lower = better)
            # Also extract highlighted snippets showing match context
            cursor.execute(
                """
                SELECT t.*,
                       bm25(things_fts) AS relevance_score,
                       snippet(things_fts, 1, '<b>', '</b>', '...', 32) AS snippet
                FROM things t
                JOIN things_fts fts ON t.id = fts.thing_id
                WHERE things_fts MATCH ?
                ORDER BY bm25(things_fts)
                LIMIT ?
            """,
                (query, limit),
            )

            results = [dict(row) for row in cursor.fetchall()]
            # Note: Connection is persistent, no need to close

            return results

        except sqlite3.OperationalError as e:
            # FTS5 query syntax error - provide helpful error message
            raise ValueError(
                f"Invalid FTS5 query syntax: {query}. "
                f"Error: {str(e)}. "
                f"Tip: Use AND, OR, NOT operators, or quote phrases."
            ) from e

    def search_candidates(
        self, query: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search for things and return minimal candidate previews (two-phase retrieval).

        This returns only essential fields for LLM preview:
        - id, relevance_score, snippet
        - timestamp, tags, people
        - is_completed

        Results are cached for 15 minutes (default TTL) to speed up repeated queries.
        Use get_things_by_ids() to fetch full content for selected candidates.

        Args:
            query: Search query (FTS5 syntax)
            limit: Maximum number of candidates to return

        Returns:
            List of candidate dictionaries with minimal fields (~75% smaller than full records)

        Raises:
            ValueError: If the FTS5 query syntax is invalid
        """
        # Check cache first
        cached = self.cache.get(query, limit=limit)
        if cached is not None:
            return cached

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # First, get matching thing IDs with BM25 scores and snippets
            cursor.execute(
                """
                SELECT
                    t.id,
                    bm25(things_fts) AS relevance_score,
                    snippet(things_fts, 1, '<b>', '</b>', '...', 32) AS snippet,
                    t.timestamp,
                    t.is_completed
                FROM things t
                JOIN things_fts fts ON t.id = fts.thing_id
                WHERE things_fts MATCH ?
                ORDER BY bm25(things_fts)
                LIMIT ?
            """,
                (query, limit),
            )

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                thing_id = result["id"]

                # Fetch tags for this thing
                cursor.execute(
                    """
                    SELECT t.name
                    FROM tags t
                    JOIN thing_tags tt ON t.id = tt.tag_id
                    WHERE tt.thing_id = ?
                """,
                    (thing_id,),
                )
                result["tags"] = [r[0] for r in cursor.fetchall()]

                # Fetch people for this thing
                cursor.execute(
                    """
                    SELECT p.name
                    FROM people p
                    JOIN thing_people tp ON p.id = tp.person_id
                    WHERE tp.thing_id = ?
                """,
                    (thing_id,),
                )
                result["people"] = [r[0] for r in cursor.fetchall()]

                results.append(result)

            # Cache the results
            self.cache.set(query, results, limit=limit)

            return results

        except sqlite3.OperationalError as e:
            raise ValueError(
                f"Invalid FTS5 query syntax: {query}. "
                f"Error: {str(e)}. "
                f"Tip: Use AND, OR, NOT operators, or quote phrases."
            ) from e

    def get_thing_by_id(self, thing_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single thing by ID with full details.

        Args:
            thing_id: The thing ID to fetch

        Returns:
            Thing dictionary with all fields, or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM things WHERE id = ?", (thing_id,))
        row = cursor.fetchone()

        if row is None:
            # Note: Connection is persistent, no need to close
            return None

        result = dict(row)

        # Fetch related entities
        cursor.execute(
            """
            SELECT p.name
            FROM people p
            JOIN thing_people tp ON p.id = tp.person_id
            WHERE tp.thing_id = ?
        """,
            (thing_id,),
        )
        result["people_mentioned"] = [row[0] for row in cursor.fetchall()]

        cursor.execute(
            """
            SELECT t.name
            FROM tags t
            JOIN thing_tags tt ON t.id = tt.tag_id
            WHERE tt.thing_id = ?
        """,
            (thing_id,),
        )
        result["tags"] = [row[0] for row in cursor.fetchall()]

        cursor.execute(
            "SELECT link_text, url FROM links WHERE thing_id = ?", (thing_id,)
        )
        result["links"] = [
            {"text": row[0], "url": row[1]} for row in cursor.fetchall()
        ]

        # Note: Connection is persistent, no need to close
        return result

    def get_things_by_ids(self, thing_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Batch fetch things by IDs with full details.

        Args:
            thing_ids: List of thing IDs to fetch

        Returns:
            List of thing dictionaries with all fields, in same order as input IDs
        """
        if not thing_ids:
            return []

        results = []
        for thing_id in thing_ids:
            thing = self.get_thing_by_id(thing_id)
            if thing:
                results.append(thing)

        return results

    def get_tasks_by_person(
        self, person_name: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all things mentioning a specific person.

        Args:
            person_name: Name of person to search for
            limit: Maximum number of results

        Returns:
            List of thing dictionaries
        """
        # Uses LIKE for partial matches; this is case-insensitive by default in SQLite.
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT t.*, p.name as person_name
            FROM things t
            JOIN thing_people tp ON t.id = tp.thing_id
            JOIN people p ON tp.person_id = p.id
            WHERE p.name LIKE ?
            ORDER BY t.timestamp DESC
            LIMIT ?
        """,
            (f"%{person_name}%", limit),
        )

        results = [dict(row) for row in cursor.fetchall()]
        # Note: Connection is persistent, no need to close

        return results

    def get_tasks_by_tag(self, tag_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all things with a specific tag.

        Args:
            tag_name: Tag to search for
            limit: Maximum number of results

        Returns:
            List of thing dictionaries
        """
        # Tags are stored normalized (lowercase), so normalize input.
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT t.*, tag.name as tag_name
            FROM things t
            JOIN thing_tags tt ON t.id = tt.thing_id
            JOIN tags tag ON tt.tag_id = tag.id
            WHERE tag.name = ?
            ORDER BY t.timestamp DESC
            LIMIT ?
        """,
            (tag_name.lower(), limit),
        )

        results = [dict(row) for row in cursor.fetchall()]
        # Note: Connection is persistent, no need to close

        return results

    def get_people_list(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get list of all people with mention counts.

        Args:
            limit: Maximum number of people to return.

        Returns:
            List of people with counts
        """
        # LEFT JOIN keeps people even if they have zero mentions.
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT p.name, p.category, COUNT(tp.thing_id) as mention_count
            FROM people p
            LEFT JOIN thing_people tp ON p.id = tp.person_id
            GROUP BY p.id
            ORDER BY mention_count DESC
            LIMIT ?
        """,
            (limit,),
        )

        results = [dict(row) for row in cursor.fetchall()]
        # Note: Connection is persistent, no need to close

        return results

    def get_tags_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all tags with usage counts.

        Returns:
            List of tags with counts
        """
        # Aggregates usage counts from the junction table.
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT t.name, COUNT(tt.thing_id) as use_count
            FROM tags t
            LEFT JOIN thing_tags tt ON t.id = tt.tag_id
            GROUP BY t.id
            ORDER BY use_count DESC
        """
        )

        results = [dict(row) for row in cursor.fetchall()]
        # Note: Connection is persistent, no need to close

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary of statistics
        """
        # Each query reads from the database and does not modify data.
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}

        # Task counts
        cursor.execute("SELECT COUNT(*) as total FROM things")
        stats["total_things"] = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) FROM things WHERE is_completed = 1")
        stats["completed_things"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM things WHERE is_strikethrough = 1")
        stats["strikethrough_things"] = cursor.fetchone()[0]

        # Date range
        cursor.execute(
            "SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest FROM things"
        )
        row = cursor.fetchone()
        stats["date_range"] = {"earliest": row["earliest"], "latest": row["latest"]}

        # Entity counts
        cursor.execute("SELECT COUNT(*) FROM people")
        stats["total_people"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM tags")
        stats["total_tags"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM links")
        stats["total_links"] = cursor.fetchone()[0]

        # Note: Connection is persistent, no need to close

        return stats

    def get_count_info(self) -> Dict[str, Any]:
        """
        Return a minimal health check payload for the database.

        Returns:
            A dict containing the database path, total thing count,
            and optional metadata like source file and load time.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM things")
        total_things = cursor.fetchone()[0]

        metadata: Dict[str, Any] = {
            "db_path": str(self.db_path),
            "total_things": total_things,
            "source_file": None,
            "json_file": None,
            "last_loaded": None,
        }

        cursor.execute("SELECT key, value FROM metadata")
        for key, value in cursor.fetchall():
            if key in metadata:
                metadata[key] = value

        # Note: Connection is persistent, no need to close
        return metadata

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get query cache performance statistics.

        Returns:
            Dictionary with cache size, hit rate, and timing info
        """
        return self.cache.get_stats()
