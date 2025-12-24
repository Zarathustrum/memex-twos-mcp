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
    Full-text search returns matching things.

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
