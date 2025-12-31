"""
Tests for ThreadPacks builder (scripts/build_threads.py)

Covers:
- TH1 pack format validation
- Thread status logic (active/stale)
- Highlight scoring (deterministic)
- Keyword extraction (deterministic)
- Incremental rebuild (src_hash matching)
- FTS search
"""

import re
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Import builder functions
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_threads import (  # noqa: E402
    build,
    build_th1_pack,
    build_thread,
    compute_src_hash,
    extract_keywords,
    score_thread_highlight,
)


@pytest.fixture
def temp_db():
    """Create a temporary test database with schema."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Create schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Minimal schema for testing
    cursor.execute(
        """
        CREATE TABLE things (
            id TEXT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT,
            is_completed BOOLEAN DEFAULT 0,
            is_pending BOOLEAN DEFAULT 0,
            is_strikethrough BOOLEAN DEFAULT 0
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE thing_tags (
            thing_id TEXT,
            tag_id INTEGER,
            PRIMARY KEY (thing_id, tag_id),
            FOREIGN KEY (thing_id) REFERENCES things(id),
            FOREIGN KEY (tag_id) REFERENCES tags(id)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE thing_people (
            thing_id TEXT,
            person_id INTEGER,
            PRIMARY KEY (thing_id, person_id),
            FOREIGN KEY (thing_id) REFERENCES things(id),
            FOREIGN KEY (person_id) REFERENCES people(id)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE threads (
            thread_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK(kind IN ('tag','person')),
            label TEXT NOT NULL,
            label_norm TEXT NOT NULL,
            start_ts TEXT,
            last_ts TEXT,
            thing_count INTEGER NOT NULL,
            thing_count_90d INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('active','stale','archived')),
            archived_at TEXT,
            pack_v INTEGER NOT NULL,
            pack TEXT NOT NULL,
            kw TEXT NOT NULL,
            src_hash TEXT NOT NULL,
            builder_v TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE thread_evidence (
            thread_id TEXT NOT NULL,
            thing_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('hi','ev')),
            rank INTEGER,
            PRIMARY KEY (thread_id, thing_id, role),
            FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE,
            FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE
        )
    """
    )

    cursor.execute(
        """
        CREATE VIRTUAL TABLE threads_fts USING fts5(
            thread_id UNINDEXED,
            label,
            kw,
            tokenize='porter unicode61'
        )
    """
    )

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    db_path.unlink()


def test_compute_src_hash():
    """Test deterministic src_hash computation."""
    things1 = [
        {"id": "task_001", "content_hash": "abc123"},
        {"id": "task_002", "content_hash": "def456"},
    ]

    things2 = [
        {"id": "task_002", "content_hash": "def456"},
        {"id": "task_001", "content_hash": "abc123"},
    ]

    # Should be identical (order-independent)
    hash1 = compute_src_hash(things1)
    hash2 = compute_src_hash(things2)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest


def test_score_thread_highlight():
    """Test deterministic highlight scoring."""
    now = datetime.now()

    # Recent, high entity density, long content
    thing1 = {
        "id": "task_001",
        "timestamp": (now - timedelta(days=1)).isoformat(),
        "content": "A" * 250,  # Long content
        "tags": ["work", "urgent"],
        "people_mentioned": ["Alice"],
    }

    # Old, no entities, short content
    thing2 = {
        "id": "task_002",
        "timestamp": (now - timedelta(days=100)).isoformat(),
        "content": "Short",
        "tags": [],
        "people_mentioned": [],
    }

    score1 = score_thread_highlight(thing1, now, active_days=90)
    score2 = score_thread_highlight(thing2, now, active_days=90)

    # thing1 should score much higher
    assert score1 > score2
    assert 0.0 <= score1 <= 1.0
    assert 0.0 <= score2 <= 1.0


def test_extract_keywords():
    """Test deterministic keyword extraction."""
    things = [
        {
            "id": "task_001",
            "content": "Review the quarterly budget and prepare invoice for client meeting",
        },
        {
            "id": "task_002",
            "content": "Schedule dentist appointment and review insurance coverage",
        },
    ]

    keywords = extract_keywords(things, top_n=10)

    # Should be sorted alphabetically
    assert keywords == sorted(keywords)

    # Should contain relevant keywords
    assert "review" in keywords or "budget" in keywords or "dentist" in keywords

    # Should exclude stopwords
    assert "the" not in keywords
    assert "and" not in keywords


def test_build_th1_pack_format():
    """Test TH1 pack format structure."""
    highlights = [
        {"id": "task_001", "label": "q4_review", "content": "Q4 review"},
        {"id": "task_002", "label": "budget_check", "content": "Budget check"},
    ]

    pack = build_th1_pack(
        thread_id="thr:tag:work",
        status="active",
        last_ts="2025-12-30T10:00:00",
        thing_count=150,
        thing_count_90d=25,
        keywords=["budget", "invoice", "meeting"],
        highlights=highlights,
    )

    # Validate format
    assert pack.startswith("TH1|")
    assert "id=thr:tag:work" in pack
    assert "st=active" in pack
    assert "last=2025-12-30" in pack
    assert "n=150" in pack
    assert "a90=25" in pack
    assert "kw=budget,invoice,meeting" in pack
    assert "hi=task_001~q4_review;task_002~budget_check" in pack

    # Should be bounded
    assert len(pack) <= 600


def test_build_th1_pack_regex():
    """Test TH1 pack format with regex."""
    pack = build_th1_pack(
        "thr:tag:test",
        "active",
        "2025-12-30T10:00:00",
        100,
        10,
        ["test"],
        [{"id": "task_001", "label": "test", "content": "test"}],
    )

    # Regex pattern for TH1 format
    pattern = (
        r"^TH1\|id=thr:(tag|person):[a-z0-9_]+\|st=(active|stale)\|"
        r"last=\d{4}-\d{2}-\d{2}\|n=\d+\|a90=\d+\|kw=.*\|hi=.*$"
    )

    assert re.match(pattern, pack), f"Pack doesn't match TH1 format: {pack}"


def test_thread_status_active():
    """Test thread status logic: active when thing_count_90d > 0."""
    now = datetime.now()
    recent = (now - timedelta(days=10)).isoformat()

    highlights = [{"id": "task_001", "label": "test", "content": "test"}]

    pack = build_th1_pack("thr:tag:work", "active", recent, 100, 15, [], highlights)

    assert "st=active" in pack


def test_thread_status_stale():
    """Test thread status logic: stale when thing_count_90d == 0."""
    old = (datetime.now() - timedelta(days=100)).isoformat()

    pack = build_th1_pack("thr:tag:work", "stale", old, 100, 0, [], [])

    assert "st=stale" in pack


def test_build_thread_with_tag(temp_db):
    """Test building a tag thread."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert tag
    cursor.execute("INSERT INTO tags (name) VALUES ('work')")
    tag_id = cursor.lastrowid

    # Insert things
    now = datetime.now()
    for i in range(5):
        ts = (now - timedelta(days=i * 20)).isoformat()
        cursor.execute(
            "INSERT INTO things (id, timestamp, content, content_hash) VALUES (?, ?, ?, ?)",
            (f"task_{i:03d}", ts, f"Work thing {i}", f"hash{i}"),
        )
        cursor.execute(
            "INSERT INTO thing_tags (thing_id, tag_id) VALUES (?, ?)",
            (f"task_{i:03d}", tag_id),
        )

    conn.commit()

    # Build thread
    was_built, error = build_thread(
        conn, "tag", "work", "work", active_days=90, force=False
    )

    assert was_built
    assert error is None

    # Check thread was created
    cursor.execute("SELECT * FROM threads WHERE thread_id = 'thr:tag:work'")
    thread = cursor.fetchone()

    assert thread is not None
    assert thread[1] == "tag"  # kind
    assert thread[2] == "work"  # label
    assert thread[6] == 5  # thing_count

    # Check FTS entry
    cursor.execute("SELECT * FROM threads_fts WHERE thread_id = 'thr:tag:work'")
    fts_row = cursor.fetchone()
    assert fts_row is not None

    conn.close()


def test_build_thread_incremental(temp_db):
    """Test incremental rebuild with src_hash matching."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert tag and thing
    cursor.execute("INSERT INTO tags (name) VALUES ('test')")
    tag_id = cursor.lastrowid

    cursor.execute(
        "INSERT INTO things (id, timestamp, content, content_hash) VALUES (?, ?, ?, ?)",
        ("task_001", "2025-12-30 10:00:00", "Test", "hash1"),
    )
    cursor.execute(
        "INSERT INTO thing_tags (thing_id, tag_id) VALUES (?, ?)", ("task_001", tag_id)
    )
    conn.commit()

    # First build
    was_built1, _ = build_thread(conn, "tag", "test", "test", force=False)
    assert was_built1

    # Second build (no changes)
    was_built2, _ = build_thread(conn, "tag", "test", "test", force=False)
    assert not was_built2  # Should skip

    # Third build with force
    was_built3, _ = build_thread(conn, "tag", "test", "test", force=True)
    assert was_built3  # Should rebuild

    conn.close()


def test_build_function(temp_db):
    """Test main build() function."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert tags and people
    cursor.execute("INSERT INTO tags (name) VALUES ('work'), ('health')")
    cursor.execute("INSERT INTO people (name) VALUES ('Alice'), ('Bob')")

    # Insert things
    now = datetime.now()
    for i in range(10):
        ts = (now - timedelta(days=i * 10)).isoformat()
        cursor.execute(
            "INSERT INTO things (id, timestamp, content, content_hash) VALUES (?, ?, ?, ?)",
            (f"task_{i:03d}", ts, f"Thing {i}", f"hash{i}"),
        )

        # Tag first 5 with 'work', last 5 with 'health'
        tag_id = 1 if i < 5 else 2
        cursor.execute(
            "INSERT INTO thing_tags (thing_id, tag_id) VALUES (?, ?)",
            (f"task_{i:03d}", tag_id),
        )

        # Mention Alice in first 3, Bob in last 3
        person_id = 1 if i < 3 else (2 if i >= 7 else None)
        if person_id:
            cursor.execute(
                "INSERT INTO thing_people (thing_id, person_id) VALUES (?, ?)",
                (f"task_{i:03d}", person_id),
            )

    conn.commit()
    conn.close()

    # Build threads
    result = build(db_path=temp_db, force=False, active_days=90, kinds="tag,person")

    assert result["success"]
    assert result["stats"]["tag_threads"] == 2
    assert result["stats"]["person_threads"] == 2
    assert result["stats"]["thread_count"] == 4
    assert result["stats"]["active_count"] > 0


def test_fts_search(temp_db):
    """Test FTS search on threads."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert thread manually for testing
    cursor.execute(
        """
        INSERT INTO threads (
            thread_id, kind, label, label_norm, start_ts, last_ts,
            thing_count, thing_count_90d, status, archived_at,
            pack_v, pack, kw, src_hash, builder_v
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            "thr:tag:health",
            "tag",
            "health",
            "health",
            "2025-01-01",
            "2025-12-30",
            50,
            10,
            "active",
            None,
            1,
            "TH1|...",
            "dentist doctor medical",
            "hash123",
            "1.0",
        ),
    )

    cursor.execute(
        """
        INSERT INTO threads_fts (thread_id, label, kw)
        VALUES (?, ?, ?)
    """,
        ("thr:tag:health", "health", "dentist doctor medical"),
    )

    conn.commit()

    # Search using FTS
    cursor.execute(
        """
        SELECT t.*
        FROM threads t
        JOIN threads_fts fts ON t.thread_id = fts.thread_id
        WHERE threads_fts MATCH ?
    """,
        ("dentist",),
    )

    results = cursor.fetchall()

    assert len(results) == 1
    assert results[0][0] == "thr:tag:health"

    conn.close()


def test_keyword_extraction_deterministic():
    """Test that keyword extraction is deterministic."""
    things = [
        {"id": "task_001", "content": "dentist appointment dental checkup"},
        {"id": "task_002", "content": "budget review financial planning"},
    ]

    # Extract multiple times
    kw1 = extract_keywords(things, top_n=10)
    kw2 = extract_keywords(things, top_n=10)
    kw3 = extract_keywords(things, top_n=10)

    # Should be identical
    assert kw1 == kw2 == kw3

    # Should be sorted
    assert kw1 == sorted(kw1)


def test_empty_thread_skipped(temp_db):
    """Test that threads with no things are skipped."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert tag but no things
    cursor.execute("INSERT INTO tags (name) VALUES ('empty')")
    conn.commit()

    # Try to build thread
    was_built, error = build_thread(conn, "tag", "empty", "empty", force=False)

    assert not was_built
    assert error is None

    # Check no thread was created
    cursor.execute("SELECT * FROM threads WHERE thread_id = 'thr:tag:empty'")
    assert cursor.fetchone() is None

    conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
