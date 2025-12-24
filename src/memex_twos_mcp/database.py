"""
Database wrapper for Twos SQLite database.
Provides safe query methods for MCP tools.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class TwosDatabase:
    """Wrapper for querying the Twos SQLite database."""

    def __init__(self, db_path: Path):
        """
        Initialize database wrapper.

        Args:
            db_path: Path to the SQLite database file on disk.

        Raises:
            FileNotFoundError: If the database file does not exist.
        """
        self.db_path = db_path
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with row factory.

        Returns:
            A sqlite3.Connection that yields rows as dict-like objects.

        Side effects:
            Opens a file-backed SQLite connection.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn

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
        conn.close()

        return results

    def search_content(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Full-text search across thing content.

        Args:
            query: Search query (FTS5 syntax)
            limit: Maximum number of results

        Returns:
            List of matching thing dictionaries
        """
        # FTS5 queries can raise sqlite3.OperationalError for invalid syntax.
        conn = self._get_connection()
        cursor = conn.cursor()

        # Join FTS results with full thing data
        cursor.execute(
            """
            SELECT t.* FROM things t
            JOIN things_fts fts ON t.id = fts.thing_id
            WHERE things_fts MATCH ?
            ORDER BY t.timestamp DESC
            LIMIT ?
        """,
            (query, limit),
        )

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

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
        conn.close()

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
        conn.close()

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
        conn.close()

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
        conn.close()

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

        conn.close()

        return stats

    def get_count_info(self) -> Dict[str, Any]:
        """
        Return a minimal health check payload for the database.

        Returns:
            A dict containing the database path, total thing count,
            and optional metadata like source file and last load time.
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

        conn.close()
        return metadata
