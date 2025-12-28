"""Tests for the TwosDatabase wrapper."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from memex_twos_mcp.database import TwosDatabase


def _init_db(db_path: Path, schema_path: Path) -> None:
    """
    Initialize a sqlite database with the project schema.

    Args:
        db_path: Path to the temporary SQLite database file.
        schema_path: Path to the SQL schema file.

    This creates a minimal dataset used by tests in this file.

    Returns:
        None.
    """
    # I/O boundary: read schema SQL from disk.
    schema_sql = schema_path.read_text(encoding="utf-8")
    # I/O boundary: create a temporary SQLite file for testing.
    conn = sqlite3.connect(db_path)
    conn.executescript(schema_sql)

    conn.execute(
        """
        INSERT INTO things (
            id, timestamp, timestamp_raw, content, content_raw,
            section_header, section_date, line_number, indent_level,
            parent_task_id, bullet_type, is_completed, is_pending,
            is_strikethrough
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task_00001",
            "2024-01-01T10:00:00",
            "01/01/24 10:00 am",
            "Meet Alice about project",
            "• Meet Alice about project 01/01/24 10:00 am",
            "Mon, Jan 1, 2024",
            "01/01/24 10:00 am",
            1,
            0,
            None,
            "bullet",
            0,
            0,
            0,
        ),
    )

    conn.execute(
        "INSERT INTO people (name, normalized_name) VALUES (?, ?)", ("Alice", "alice")
    )
    conn.execute("INSERT INTO tags (name) VALUES (?)", ("work",))
    conn.execute(
        "INSERT INTO links (thing_id, link_text, url) VALUES (?, ?, ?)",
        ("task_00001", "Link", "https://example.com"),
    )

    person_id = conn.execute("SELECT id FROM people WHERE name = 'Alice'").fetchone()[0]
    tag_id = conn.execute("SELECT id FROM tags WHERE name = 'work'").fetchone()[0]

    conn.execute(
        "INSERT INTO thing_people (thing_id, person_id) VALUES (?, ?)",
        ("task_00001", person_id),
    )
    conn.execute(
        "INSERT INTO thing_tags (thing_id, tag_id) VALUES (?, ?)",
        ("task_00001", tag_id),
    )

    conn.commit()
    conn.close()


def test_database_stats(tmp_path: Path) -> None:
    """
    TwosDatabase returns expected stats and lists.

    Args:
        tmp_path: Pytest-provided temporary directory for file I/O.

    Uses a temporary database populated by _init_db.

    Returns:
        None.
    """
    db_path = tmp_path / "twos.db"
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "schema.sql"

    _init_db(db_path, schema_path)

    db = TwosDatabase(db_path)
    stats = db.get_stats()
    assert stats["total_things"] == 1
    assert stats["total_people"] == 1
    assert stats["total_tags"] == 1

    people = db.get_people_list()
    assert people[0]["name"] == "Alice"

    tags = db.get_tags_list()
    assert tags[0]["name"] == "work"


def test_search_content(tmp_path: Path) -> None:
    """
    Full-text search returns matching things with BM25 ranking and snippets.

    Args:
        tmp_path: Pytest-provided temporary directory for file I/O.

    Uses SQLite FTS5 data created in the schema.

    Returns:
        None.
    """
    db_path = tmp_path / "twos.db"
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "schema.sql"

    _init_db(db_path, schema_path)

    db = TwosDatabase(db_path)
    results = db.search_content("Alice")
    assert len(results) == 1
    assert results[0]["id"] == "task_00001"

    # Verify BM25 ranking fields are present
    assert "relevance_score" in results[0]
    assert "snippet" in results[0]

    # Verify snippet contains highlighted search term
    assert "<b>" in results[0]["snippet"]
    assert "</b>" in results[0]["snippet"]
    assert "Alice" in results[0]["snippet"] or "alice" in results[0]["snippet"]


def test_search_content_bm25_ranking(tmp_path: Path) -> None:
    """
    Test that BM25 ranking orders results by relevance, not timestamp.

    Creates multiple things where the most relevant result has an earlier timestamp.
    Verifies that relevance ordering takes precedence.

    Args:
        tmp_path: Pytest-provided temporary directory for file I/O.

    Returns:
        None.
    """
    db_path = tmp_path / "twos.db"
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "schema.sql"

    # Initialize database with multiple things
    schema_sql = schema_path.read_text(encoding="utf-8")
    conn = sqlite3.connect(db_path)
    conn.executescript(schema_sql)

    # Insert thing 1: Contains "doctor" once (older timestamp)
    conn.execute(
        """
        INSERT INTO things (
            id, timestamp, content, section_header,
            bullet_type, is_completed, is_pending, is_strikethrough
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task_00001",
            "2024-01-01T10:00:00",
            "Call doctor about appointment",
            "Mon, Jan 1, 2024",
            "bullet",
            0,
            0,
            0,
        ),
    )

    # Insert thing 2: Contains "doctor" multiple times (newer timestamp, higher relevance)
    conn.execute(
        """
        INSERT INTO things (
            id, timestamp, content, section_header,
            bullet_type, is_completed, is_pending, is_strikethrough
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task_00002",
            "2024-01-05T10:00:00",
            "Doctor appointment confirmed with doctor office, doctor said to bring records",
            "Fri, Jan 5, 2024",
            "bullet",
            0,
            0,
            0,
        ),
    )

    # Insert thing 3: Unrelated content (newer timestamp but no match)
    conn.execute(
        """
        INSERT INTO things (
            id, timestamp, content, section_header,
            bullet_type, is_completed, is_pending, is_strikethrough
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task_00003",
            "2024-01-10T10:00:00",
            "Buy groceries and prepare dinner",
            "Wed, Jan 10, 2024",
            "bullet",
            0,
            0,
            0,
        ),
    )

    conn.commit()
    conn.close()

    # Search for "doctor"
    db = TwosDatabase(db_path)
    results = db.search_content("doctor")

    # Should return 2 results (task_00001 and task_00002)
    assert len(results) == 2

    # BM25 should rank task_00002 higher (contains "doctor" 3 times)
    # even though it has a later timestamp
    assert results[0]["id"] == "task_00002", "BM25 should rank most relevant first"
    assert results[1]["id"] == "task_00001"

    # Verify both have relevance scores
    assert "relevance_score" in results[0]
    assert "relevance_score" in results[1]

    # Verify first result has better (lower/more negative) BM25 score
    # Note: BM25 in SQLite FTS5 returns negative scores, lower = better
    assert results[0]["relevance_score"] < results[1]["relevance_score"]


def test_search_content_invalid_query(tmp_path: Path) -> None:
    """
    Test that invalid FTS5 query syntax raises ValueError with helpful message.

    Args:
        tmp_path: Pytest-provided temporary directory for file I/O.

    Returns:
        None.
    """
    db_path = tmp_path / "twos.db"
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "schema.sql"

    _init_db(db_path, schema_path)

    db = TwosDatabase(db_path)

    # Test invalid query syntax (unmatched quote)
    try:
        db.search_content('invalid"query')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        # Verify error message is helpful
        assert "Invalid FTS5 query syntax" in str(e)
        assert "Tip:" in str(e)


def test_search_candidates(tmp_path: Path) -> None:
    """
    Test two-phase retrieval: search_candidates returns minimal preview fields.

    Args:
        tmp_path: Pytest-provided temporary directory for file I/O.

    Returns:
        None.
    """
    db_path = tmp_path / "twos.db"
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "schema.sql"

    _init_db(db_path, schema_path)

    db = TwosDatabase(db_path)
    candidates = db.search_candidates("Alice")

    assert len(candidates) == 1
    candidate = candidates[0]

    # Verify minimal fields are present
    required_fields = ["id", "relevance_score", "snippet", "timestamp", "is_completed", "tags", "people"]
    for field in required_fields:
        assert field in candidate, f"Missing required field: {field}"

    # Verify heavy fields are NOT present
    excluded_fields = ["content_raw", "line_number", "indent_level", "parent_task_id", "bullet_type"]
    for field in excluded_fields:
        assert field not in candidate, f"Candidate should not include: {field}"

    # Verify tags and people are lists
    assert isinstance(candidate["tags"], list)
    assert isinstance(candidate["people"], list)


def test_get_thing_by_id(tmp_path: Path) -> None:
    """
    Test fetching a single thing by ID with full details.

    Args:
        tmp_path: Pytest-provided temporary directory for file I/O.

    Returns:
        None.
    """
    db_path = tmp_path / "twos.db"
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "schema.sql"

    _init_db(db_path, schema_path)

    db = TwosDatabase(db_path)

    # Fetch existing thing
    thing = db.get_thing_by_id("task_00001")
    assert thing is not None
    assert thing["id"] == "task_00001"
    assert thing["content"] == "Meet Alice about project"

    # Verify related entities are included
    assert "tags" in thing
    assert "people_mentioned" in thing
    assert "links" in thing

    assert "work" in thing["tags"]
    assert "Alice" in thing["people_mentioned"]
    assert len(thing["links"]) == 1
    assert thing["links"][0]["url"] == "https://example.com"

    # Fetch non-existent thing
    missing = db.get_thing_by_id("task_99999")
    assert missing is None


def test_get_things_by_ids(tmp_path: Path) -> None:
    """
    Test batch fetching things by IDs.

    Args:
        tmp_path: Pytest-provided temporary directory for file I/O.

    Returns:
        None.
    """
    db_path = tmp_path / "twos.db"
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "schema.sql"

    # Initialize with multiple things
    schema_sql = schema_path.read_text(encoding="utf-8")
    conn = sqlite3.connect(db_path)
    conn.executescript(schema_sql)

    # Insert two things
    conn.execute(
        """
        INSERT INTO things (
            id, timestamp, content, section_header,
            bullet_type, is_completed, is_pending, is_strikethrough
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("task_00001", "2024-01-01T10:00:00", "First thing", "Mon, Jan 1, 2024", "bullet", 0, 0, 0),
    )

    conn.execute(
        """
        INSERT INTO things (
            id, timestamp, content, section_header,
            bullet_type, is_completed, is_pending, is_strikethrough
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("task_00002", "2024-01-02T10:00:00", "Second thing", "Tue, Jan 2, 2024", "bullet", 0, 0, 0),
    )

    conn.commit()
    conn.close()

    db = TwosDatabase(db_path)

    # Batch fetch
    things = db.get_things_by_ids(["task_00001", "task_00002"])
    assert len(things) == 2
    assert things[0]["id"] == "task_00001"
    assert things[1]["id"] == "task_00002"

    # Fetch with some missing IDs
    things = db.get_things_by_ids(["task_00001", "task_99999"])
    assert len(things) == 1
    assert things[0]["id"] == "task_00001"

    # Empty list
    things = db.get_things_by_ids([])
    assert len(things) == 0


def test_two_phase_retrieval_workflow(tmp_path: Path) -> None:
    """
    Test complete two-phase retrieval workflow: search_candidates → get_things_by_ids.

    Args:
        tmp_path: Pytest-provided temporary directory for file I/O.

    Returns:
        None.
    """
    db_path = tmp_path / "twos.db"
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "schema.sql"

    # Initialize with multiple matching things
    schema_sql = schema_path.read_text(encoding="utf-8")
    conn = sqlite3.connect(db_path)
    conn.executescript(schema_sql)

    conn.execute(
        """
        INSERT INTO things (
            id, timestamp, content, section_header,
            bullet_type, is_completed, is_pending, is_strikethrough
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("task_00001", "2024-01-01T10:00:00", "Call doctor about checkup", "Mon, Jan 1, 2024", "bullet", 0, 0, 0),
    )

    conn.execute(
        """
        INSERT INTO things (
            id, timestamp, content, section_header,
            bullet_type, is_completed, is_pending, is_strikethrough
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("task_00002", "2024-01-02T10:00:00", "Pick up prescription from doctor", "Tue, Jan 2, 2024", "bullet", 0, 0, 0),
    )

    conn.commit()
    conn.close()

    db = TwosDatabase(db_path)

    # Phase 1: Search for candidates
    candidates = db.search_candidates("doctor", limit=10)
    assert len(candidates) == 2

    # Verify candidates have minimal fields
    for candidate in candidates:
        assert "snippet" in candidate
        assert "doctor" in candidate["snippet"].lower()

    # Phase 2: Select top candidate and fetch full details
    top_id = candidates[0]["id"]
    full_thing = db.get_thing_by_id(top_id)

    assert full_thing is not None
    assert full_thing["id"] == top_id
    assert "content" in full_thing  # Full record has content field
    assert "content_raw" in full_thing  # Full record has all fields
