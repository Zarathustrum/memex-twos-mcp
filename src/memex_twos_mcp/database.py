"""
Database wrapper for Twos SQLite database.
Provides safe query methods for MCP tools.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import sqlite_vec  # type: ignore

    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

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

        # Initialize embedding generator (graceful degradation)
        self.embeddings_enabled = False
        self.embedding_gen = None
        try:
            from .embeddings import EmbeddingGenerator

            self.embedding_gen = EmbeddingGenerator()
            if self.embedding_gen.available:
                self.embeddings_enabled = True
                # Initialize vector search extension
                self._init_vector_search()
        except Exception as e:
            print(f"⚠️  Embeddings disabled: {e}")

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
                    self.db_path,
                    check_same_thread=False,  # Allow multi-thread with lock
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

    def _init_vector_search(self):
        """Initialize sqlite-vec extension for vector similarity search."""
        if not SQLITE_VEC_AVAILABLE:
            print("⚠️  sqlite-vec not installed. Vector search unavailable.")
            self.embeddings_enabled = False
            return

        conn = self._get_connection()

        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

            # Create virtual table for vector search
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0(
                    thing_id TEXT PRIMARY KEY,
                    embedding float[384]
                )
            """
            )
            conn.commit()
        except Exception as e:
            print(f"⚠️  Vector search unavailable: {e}")
            self.embeddings_enabled = False

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
        - id, timestamp, is_completed, is_strikethrough, is_pending
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
                is_strikethrough,
                is_pending,
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
        self.cache.set(
            f"date_query:{start_date}:{end_date}", results, **cache_key_parts
        )

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

    def search_candidates(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search for things and return minimal candidate previews (two-phase retrieval).

        This returns only essential fields for LLM preview:
        - id, relevance_score, snippet
        - timestamp, tags, people
        - is_completed, is_strikethrough, is_pending

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
        if not query or not query.strip():
            return []

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
                    t.is_completed,
                    t.is_strikethrough,
                    t.is_pending
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
        result["links"] = [{"text": row[0], "url": row[1]} for row in cursor.fetchall()]

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

    def hybrid_search(
        self,
        query: str,
        limit: int = 50,
        lexical_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60,
        enable_semantic: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining BM25 (lexical) + vector similarity (semantic).

        Uses Reciprocal Rank Fusion (RRF) to merge rankings:
            RRF_score(doc) = sum over all rankings ( weight / (k + rank) )

        Args:
            query: Search query text
            limit: Number of results to return
            lexical_weight: Weight for BM25 scores (0-1)
            semantic_weight: Weight for vector similarity (0-1)
            rrf_k: RRF constant (default 60 per literature)
            enable_semantic: If False, fall back to lexical-only

        Returns:
            List of candidate dictionaries with hybrid_score field
        """
        # Phase 1: BM25 lexical search
        lexical_results = self.search_candidates(query, limit=limit * 2)

        # Phase 2: Vector semantic search (if enabled)
        if enable_semantic and self.embeddings_enabled:
            try:
                semantic_results = self._vector_search(query, limit=limit * 2)
            except Exception as e:
                print(f"⚠️  Semantic search failed, falling back to lexical: {e}")
                semantic_results = []
        else:
            semantic_results = []

        # Phase 3: Reciprocal Rank Fusion
        thing_scores: Dict[str, float] = {}
        thing_data: Dict[str, Dict] = {}

        # Add lexical scores
        for rank, result in enumerate(lexical_results, start=1):
            thing_id = result["id"]
            thing_scores[thing_id] = lexical_weight / (rrf_k + rank)
            thing_data[thing_id] = result

        # Add semantic scores
        for rank, result in enumerate(semantic_results, start=1):
            thing_id = result["id"]
            thing_scores[thing_id] = thing_scores.get(thing_id, 0) + (
                semantic_weight / (rrf_k + rank)
            )
            if thing_id not in thing_data:
                thing_data[thing_id] = result

        # Sort by combined score and return top N
        ranked = sorted(thing_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        results = []
        for thing_id, hybrid_score in ranked:
            result = thing_data[thing_id].copy()
            result["hybrid_score"] = hybrid_score
            results.append(result)

        return results

    def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Vector similarity search using cosine similarity.

        Args:
            query: Search query text
            limit: Number of results

        Returns:
            List of candidates with cosine_similarity scores
        """
        if not self.embeddings_enabled or self.embedding_gen is None:
            raise RuntimeError("Embeddings not available")

        # Generate query embedding
        query_embedding = self.embedding_gen.encode_single(query)

        # Search using sqlite-vec
        conn = self._get_connection()
        cursor = conn.cursor()

        # Convert embedding to bytes for SQLite storage
        embedding_bytes = query_embedding.tobytes()

        cursor.execute(
            """
            SELECT
                v.thing_id,
                vec_distance_cosine(v.embedding, ?) AS distance
            FROM vec_index v
            ORDER BY distance ASC
            LIMIT ?
        """,
            (embedding_bytes, limit),
        )

        # Fetch thing metadata for results
        results = []
        for row in cursor.fetchall():
            thing_id, distance = row
            thing = self.get_thing_by_id(thing_id)
            if thing:
                # Convert distance to similarity (cosine distance = 1 - similarity)
                thing["cosine_similarity"] = 1 - distance
                results.append(thing)

        return results

    # ========================================================================
    # List-Scoped Queries (Phase 6: List Semantics)
    # ========================================================================

    def get_list_by_date(
        self, date: str, include_non_substantive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all items on the list for a specific date.

        Args:
            date: ISO date string (YYYY-MM-DD) or 'today'
            include_non_substantive: Include dividers/headers (default: False)

        Returns:
            List of thing dictionaries ordered by line_number
        """
        if date == "today":
            from datetime import datetime

            date = datetime.now().date().isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()

        # Build query
        query = """
            SELECT t.*
            FROM things t
            WHERE t.list_id = ?
        """
        params = [f"date_{date}"]

        if not include_non_substantive:
            query += " AND t.item_type = 'content'"

        query += " ORDER BY t.line_number"

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]

        return results

    def get_list_by_name(
        self,
        name: str,
        list_type: Optional[str] = None,
        include_non_substantive: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get all items on a named list (e.g., 'Tech Projects').

        Args:
            name: List name (case-insensitive)
            list_type: Optional filter ('topic', 'date', 'category')
            include_non_substantive: Include dividers/headers (default: False)

        Returns:
            List of thing dictionaries ordered by line_number
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build query
        query = """
            SELECT t.*
            FROM things t
            JOIN lists l ON t.list_id = l.list_id
            WHERE LOWER(l.list_name) = LOWER(?)
        """
        params = [name]

        if list_type:
            query += " AND l.list_type = ?"
            params.append(list_type)

        if not include_non_substantive:
            query += " AND t.item_type = 'content'"

        query += " ORDER BY t.line_number"

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]

        return results

    def get_all_lists(
        self, list_type: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get all lists with summary statistics.

        Args:
            list_type: Optional filter ('date', 'topic', 'category')
            limit: Max results (default 50)

        Returns:
            List of list dictionaries with stats
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                l.list_id,
                l.list_type,
                l.list_name,
                l.list_name_raw,
                l.list_date,
                l.substantive_count AS items,
                COUNT(CASE WHEN t.is_completed = 0 AND t.item_type = 'content' THEN 1 END) AS open_items,
                COUNT(CASE WHEN t.is_completed = 1 AND t.item_type = 'content' THEN 1 END) AS completed_items
            FROM lists l
            LEFT JOIN things t ON l.list_id = t.list_id
        """
        params = []

        if list_type:
            query += " WHERE l.list_type = ?"
            params.append(list_type)

        query += """
            GROUP BY l.list_id
            ORDER BY l.list_date DESC NULLS LAST, l.list_name
            LIMIT ?
        """
        params.append(limit)

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]

        return results

    def get_list_metadata(self, list_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific list.

        Args:
            list_id: List identifier

        Returns:
            List metadata dictionary or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM lists WHERE list_id = ?", (list_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def search_within_list(
        self,
        query: str,
        list_id: Optional[str] = None,
        list_date: Optional[str] = None,
        list_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for items within a specific list.

        Must provide one of: list_id, list_date, or list_name.

        Args:
            query: Search query (FTS5 syntax)
            list_id: Exact list ID
            list_date: ISO date for date-based lists
            list_name: List name (case-insensitive)
            limit: Max results

        Returns:
            List of matching thing dictionaries with relevance scores

        Raises:
            ValueError: If no list identifier provided or invalid query
        """
        if not (list_id or list_date or list_name):
            raise ValueError(
                "Must provide one of: list_id, list_date, or list_name"
            )

        conn = self._get_connection()
        cursor = conn.cursor()

        # Determine list_id from other parameters if needed
        target_list_id = list_id

        if not target_list_id and list_date:
            target_list_id = f"date_{list_date}"

        if not target_list_id and list_name:
            # Look up list_id from list_name
            cursor.execute(
                "SELECT list_id FROM lists WHERE LOWER(list_name) = LOWER(?) LIMIT 1",
                (list_name,),
            )
            row = cursor.fetchone()
            if row:
                target_list_id = row[0]
            else:
                # List not found
                return []

        try:
            # Search within list using FTS
            cursor.execute(
                """
                SELECT t.*,
                       bm25(things_fts) AS relevance_score,
                       snippet(things_fts, 1, '<b>', '</b>', '...', 32) AS snippet
                FROM things t
                JOIN things_fts fts ON t.id = fts.thing_id
                WHERE things_fts MATCH ?
                  AND t.list_id = ?
                  AND t.item_type = 'content'
                ORDER BY bm25(things_fts)
                LIMIT ?
            """,
                (query, target_list_id, limit),
            )

            results = [dict(row) for row in cursor.fetchall()]
            return results

        except Exception as e:
            raise ValueError(
                f"Invalid FTS5 query syntax: {query}. Error: {str(e)}"
            ) from e

    # ========================================================================
    # TimePacks: Rollup Queries (Phase 7)
    # ========================================================================

    def get_rollups(
        self,
        kind: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get rollups with optional filtering.

        Args:
            kind: Filter by kind ('d', 'w', 'm')
            start_date: Filter by start_date >= value (ISO date)
            end_date: Filter by start_date <= value (ISO date)
            limit: Maximum results

        Returns:
            List of rollup dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM rollups WHERE 1=1"
        params: List[Any] = []

        if kind:
            query += " AND kind = ?"
            params.append(kind)

        if start_date:
            query += " AND start_date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND start_date <= ?"
            params.append(end_date)

        query += " ORDER BY start_date DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]

        return results

    def get_rollup(self, rollup_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single rollup by ID.

        Args:
            rollup_id: Rollup identifier (e.g., 'd:2025-12-30')

        Returns:
            Rollup dictionary or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM rollups WHERE rollup_id = ?", (rollup_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def get_rollup_highlights(self, rollup_id: str) -> List[Dict[str, Any]]:
        """
        Get full thing objects for rollup highlights.

        Args:
            rollup_id: Rollup identifier

        Returns:
            List of thing dictionaries (highlights only)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Fetch highlight thing_ids with rank
        cursor.execute(
            """
            SELECT thing_id, rank
            FROM rollup_evidence
            WHERE rollup_id = ? AND role = 'hi'
            ORDER BY rank
            """,
            (rollup_id,)
        )

        highlight_ids = [(row[0], row[1]) for row in cursor.fetchall()]

        # Fetch full thing objects
        results = []
        for thing_id, rank in highlight_ids:
            thing = self.get_thing_by_id(thing_id)
            if thing:
                thing["highlight_rank"] = rank
                results.append(thing)

        return results

    def search_rollups(
        self,
        keyword: str,
        kind: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search rollups by keyword (searches kw column).

        Args:
            keyword: Search keyword
            kind: Optional kind filter ('d', 'w', 'm')
            limit: Maximum results

        Returns:
            List of matching rollup dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM rollups WHERE kw LIKE ?"
        params: List[Any] = [f"%{keyword}%"]

        if kind:
            query += " AND kind = ?"
            params.append(kind)

        query += " ORDER BY start_date DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]

        return results

    # ========================================================================
    # MonthlySummaries: LLM-Powered Semantic Framing (Phase 8)
    # ========================================================================

    def get_month_summary(
        self,
        month_id: Optional[str] = None,
        offset: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Get a monthly summary by ID or offset.

        Args:
            month_id: Specific month ID (YYYY-MM) or None for current/offset
            offset: Months back from current (0=current, 1=last month, etc.)

        Returns:
            Month summary dictionary or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if month_id:
            # Fetch specific month
            cursor.execute(
                "SELECT * FROM month_summaries WHERE month_id = ?",
                (month_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            result = dict(row)

            # Parse suggested_questions JSON
            if result.get("suggested_questions"):
                try:
                    import json
                    result["suggested_questions"] = json.loads(result["suggested_questions"])
                except json.JSONDecodeError:
                    result["suggested_questions"] = {"questions": []}

            return result

        else:
            # Fetch by offset
            cursor.execute(
                """
                SELECT * FROM month_summaries
                ORDER BY start_date DESC
                LIMIT 1 OFFSET ?
                """,
                (offset,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            result = dict(row)

            # Parse suggested_questions JSON
            if result.get("suggested_questions"):
                try:
                    import json
                    result["suggested_questions"] = json.loads(result["suggested_questions"])
                except json.JSONDecodeError:
                    result["suggested_questions"] = {"questions": []}

            return result

    def list_month_summaries(self, limit: int = 12) -> List[Dict[str, Any]]:
        """
        Get list of monthly summaries.

        Args:
            limit: Maximum results (default 12)

        Returns:
            List of month summary dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM month_summaries
            ORDER BY start_date DESC
            LIMIT ?
            """,
            (limit,)
        )

        results = []
        for row in cursor.fetchall():
            result = dict(row)

            # Parse suggested_questions JSON
            if result.get("suggested_questions"):
                try:
                    import json
                    result["suggested_questions"] = json.loads(result["suggested_questions"])
                except json.JSONDecodeError:
                    result["suggested_questions"] = {"questions": []}

            results.append(result)

        return results

    def get_month_summary_highlights(self, month_id: str) -> List[Dict[str, Any]]:
        """
        Get full thing objects for month summary highlights.

        Args:
            month_id: Month identifier (YYYY-MM)

        Returns:
            List of thing dictionaries (highlights only)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Fetch highlight thing_ids with rank
        cursor.execute(
            """
            SELECT thing_id, rank
            FROM month_summary_evidence
            WHERE month_id = ? AND role = 'hi'
            ORDER BY rank
            """,
            (month_id,)
        )

        highlight_ids = [(row[0], row[1]) for row in cursor.fetchall()]

        # Fetch full thing objects
        results = []
        for thing_id, rank in highlight_ids:
            thing = self.get_thing_by_id(thing_id)
            if thing:
                thing["highlight_rank"] = rank
                results.append(thing)

        return results

    def get_month_summary_questions(self, month_id: str) -> List[Dict[str, Any]]:
        """
        Get suggested questions for a month summary.

        Args:
            month_id: Month identifier (YYYY-MM)

        Returns:
            List of question dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT suggested_questions FROM month_summaries WHERE month_id = ?",
            (month_id,)
        )

        row = cursor.fetchone()

        if row is None or not row[0]:
            return []

        try:
            import json
            questions_data = json.loads(row[0])
            return questions_data.get("questions", [])
        except json.JSONDecodeError:
            return []

    # ========================================================================
    # ThreadPacks: Active Tag/Person Thread Indices (Phase 9)
    # ========================================================================

    def list_threads(
        self,
        status: str = "active",
        kind: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List threads filtered by status and kind.

        Args:
            status: Filter by status ('active', 'stale', 'archived', or 'all')
            kind: Filter by kind ('tag', 'person', or None for all)
            limit: Maximum results

        Returns:
            List of thread dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM threads WHERE 1=1"
        params: List[Any] = []

        if status != "all":
            query += " AND status = ?"
            params.append(status)

        if kind:
            query += " AND kind = ?"
            params.append(kind)

        query += " ORDER BY last_ts DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]

        return results

    def search_threads(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search threads using FTS.

        Args:
            query: Search query (FTS5 syntax)
            limit: Maximum results

        Returns:
            List of matching thread dictionaries

        Raises:
            ValueError: If FTS5 query syntax is invalid
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Search threads_fts and join with threads table
            cursor.execute(
                """
                SELECT t.*,
                       bm25(threads_fts) AS relevance_score
                FROM threads t
                JOIN threads_fts fts ON t.thread_id = fts.thread_id
                WHERE threads_fts MATCH ?
                ORDER BY bm25(threads_fts)
                LIMIT ?
                """,
                (query, limit)
            )

            results = [dict(row) for row in cursor.fetchall()]
            return results

        except sqlite3.OperationalError as e:
            raise ValueError(
                f"Invalid FTS5 query syntax: {query}. Error: {str(e)}"
            ) from e

    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single thread by ID.

        Args:
            thread_id: Thread identifier (e.g., 'thr:tag:work')

        Returns:
            Thread dictionary or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def get_thread_highlights(
        self,
        thread_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get full thing objects for thread highlights.

        Args:
            thread_id: Thread identifier
            limit: Maximum highlights to return

        Returns:
            List of thing dictionaries (highlights only)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Fetch highlight thing_ids with rank
        cursor.execute(
            """
            SELECT thing_id, rank
            FROM thread_evidence
            WHERE thread_id = ? AND role = 'hi'
            ORDER BY rank
            LIMIT ?
            """,
            (thread_id, limit)
        )

        highlight_ids = [(row[0], row[1]) for row in cursor.fetchall()]

        # Fetch full thing objects
        results = []
        for thing_id, rank in highlight_ids:
            thing = self.get_thing_by_id(thing_id)
            if thing:
                thing["highlight_rank"] = rank
                results.append(thing)

        return results
