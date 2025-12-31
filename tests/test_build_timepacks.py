"""
Tests for TimePacks builder (scripts/build_timepacks.py)

Covers:
- TP1 pack format validation
- Highlight scoring (deterministic)
- Keyword extraction (deterministic)
- Incremental rebuild (src_hash matching)
- Week boundary handling (ISO 8601)
"""

import re
import sqlite3
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

# Import builder functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_timepacks import (
    build,
    build_rollup,
    build_tp1_pack,
    compute_src_hash,
    extract_keywords,
    generate_day_windows,
    generate_month_windows,
    generate_week_windows,
    make_label,
    score_highlight,
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
    cursor.execute("""
        CREATE TABLE things (
            id TEXT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT,
            is_completed BOOLEAN DEFAULT 0,
            is_pending BOOLEAN DEFAULT 0,
            is_strikethrough BOOLEAN DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE thing_tags (
            thing_id TEXT,
            tag_id INTEGER,
            PRIMARY KEY (thing_id, tag_id),
            FOREIGN KEY (thing_id) REFERENCES things(id),
            FOREIGN KEY (tag_id) REFERENCES tags(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE thing_people (
            thing_id TEXT,
            person_id INTEGER,
            PRIMARY KEY (thing_id, person_id),
            FOREIGN KEY (thing_id) REFERENCES things(id),
            FOREIGN KEY (person_id) REFERENCES people(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE imports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            imported_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE rollups (
            rollup_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK(kind IN ('d','w','m')),
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            thing_count INTEGER NOT NULL,
            completed_count INTEGER NOT NULL,
            pending_count INTEGER NOT NULL,
            strikethrough_count INTEGER NOT NULL,
            pack_v INTEGER NOT NULL,
            pack TEXT NOT NULL,
            kw TEXT NOT NULL,
            src_hash TEXT NOT NULL,
            build_import_id INTEGER,
            builder_v TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (build_import_id) REFERENCES imports(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE rollup_evidence (
            rollup_id TEXT NOT NULL,
            thing_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('hi','ev')),
            rank INTEGER,
            PRIMARY KEY (rollup_id, thing_id, role),
            FOREIGN KEY (rollup_id) REFERENCES rollups(rollup_id) ON DELETE CASCADE,
            FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE
        )
    """)

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


def test_score_highlight():
    """Test deterministic highlight scoring."""
    end_date = date(2025, 12, 30)

    # Recent, high entity density, long content, incomplete
    thing1 = {
        "id": "task_001",
        "timestamp": "2025-12-29T10:00:00",
        "content": "A" * 250,  # Long content
        "tags": ["work", "urgent", "review"],
        "people_mentioned": ["Alice", "Bob"],
        "is_completed": False,
        "is_pending": False,
    }

    # Old, no entities, short content, completed
    thing2 = {
        "id": "task_002",
        "timestamp": "2025-12-20T10:00:00",
        "content": "Short",
        "tags": [],
        "people_mentioned": [],
        "is_completed": True,
        "is_pending": False,
    }

    score1 = score_highlight(thing1, end_date, window_days=7)
    score2 = score_highlight(thing2, end_date, window_days=7)

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
            "tags": ["work", "finance"],
        },
        {
            "id": "task_002",
            "content": "Schedule dentist appointment and review insurance coverage",
            "tags": ["health"],
        },
    ]

    keywords = extract_keywords(things, top_n=10)

    # Should be sorted alphabetically
    assert keywords == sorted(keywords)

    # Should contain relevant keywords (tags + content tokens)
    assert "work" in keywords or "finance" in keywords or "health" in keywords
    assert "review" in keywords or "budget" in keywords or "dentist" in keywords

    # Should exclude stopwords
    assert "the" not in keywords
    assert "and" not in keywords


def test_make_label():
    """Test label generation from content."""
    # Normal content
    label1 = make_label("Schedule dentist appointment for next week")
    assert label1 == "schedule_dentist_appointment_for"
    assert len(label1) <= 32

    # Long content (truncate)
    label2 = make_label("A" * 100)
    assert len(label2) <= 32

    # Empty content
    label3 = make_label("")
    assert label3 == "item"


def test_tp1_pack_format():
    """Test TP1 pack format validation."""
    things = [
        {
            "id": "task_001",
            "content": "Review quarterly budget",
            "tags": ["work", "finance"],
            "people_mentioned": ["Alice"],
            "is_completed": False,
            "is_pending": True,
            "is_strikethrough": False,
        },
        {
            "id": "task_002",
            "content": "Dentist appointment",
            "tags": ["health"],
            "people_mentioned": [],
            "is_completed": True,
            "is_pending": False,
            "is_strikethrough": False,
        },
    ]

    highlights = [things[0]]

    pack = build_tp1_pack(
        kind="d",
        start_date="2025-12-30",
        end_date="2025-12-30",
        things=things,
        highlights=highlights,
    )

    # Validate format
    assert pack.startswith("TP1|")
    assert "k=d" in pack
    assert "s=2025-12-30" in pack
    assert "e=2025-12-30" in pack
    assert "n=2" in pack
    assert "cx=1" in pack  # 1 completed
    assert "pn=1" in pack  # 1 pending
    assert "st=0" in pack  # 0 strikethrough

    # Should contain tags and people
    assert "tg=" in pack
    assert "pp=" in pack

    # Should contain highlights
    assert "hi=task_001~" in pack

    # Should be bounded
    assert len(pack) <= 800


def test_tp1_pack_format_regex():
    """Test TP1 pack format with regex."""
    things = [
        {
            "id": "task_001",
            "content": "Test content",
            "tags": ["test"],
            "people_mentioned": ["Bob"],
            "is_completed": False,
            "is_pending": False,
            "is_strikethrough": False,
        }
    ]

    pack = build_tp1_pack("d", "2025-12-30", "2025-12-30", things, things)

    # Regex pattern for TP1 format
    pattern = r"^TP1\|k=[dwm]\|s=\d{4}-\d{2}-\d{2}\|e=\d{4}-\d{2}-\d{2}\|n=\d+\|cx=\d+\|pn=\d+\|st=\d+\|tg=.*\|pp=.*\|kw=.*\|hi=.*$"

    assert re.match(pattern, pack), f"Pack doesn't match TP1 format: {pack}"


def test_generate_day_windows():
    """Test day window generation."""
    start = date(2025, 12, 28)
    end = date(2025, 12, 30)

    windows = generate_day_windows(start, end)

    assert len(windows) == 3
    assert windows[0] == (date(2025, 12, 28), date(2025, 12, 28))
    assert windows[1] == (date(2025, 12, 29), date(2025, 12, 29))
    assert windows[2] == (date(2025, 12, 30), date(2025, 12, 30))


def test_generate_week_windows_iso8601():
    """Test ISO 8601 week window generation (Monday-based)."""
    # Dec 30, 2025 is a Tuesday
    # Previous Monday is Dec 29, 2025
    start = date(2025, 12, 28)  # Sunday
    end = date(2026, 1, 5)  # Next week Monday

    windows = generate_week_windows(start, end)

    # Should start from Monday Dec 22, 2025
    assert windows[0][0].weekday() == 0  # Monday
    assert windows[1][0].weekday() == 0  # Monday

    # Each window should be 7 days (Monday-Sunday)
    for week_start, week_end in windows:
        assert (week_end - week_start).days == 6
        assert week_start.weekday() == 0  # Monday
        assert week_end.weekday() == 6  # Sunday


def test_generate_month_windows():
    """Test month window generation."""
    start = date(2025, 11, 15)
    end = date(2026, 1, 10)

    windows = generate_month_windows(start, end)

    # Should generate Nov 2025, Dec 2025, Jan 2026
    assert len(windows) == 3

    # November
    assert windows[0][0] == date(2025, 11, 1)
    assert windows[0][1] == date(2025, 11, 30)

    # December
    assert windows[1][0] == date(2025, 12, 1)
    assert windows[1][1] == date(2025, 12, 31)

    # January
    assert windows[2][0] == date(2026, 1, 1)
    assert windows[2][1] == date(2026, 1, 31)


def test_build_rollup_incremental(temp_db):
    """Test incremental rebuild with src_hash matching."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert test data
    cursor.execute(
        """
        INSERT INTO things (id, timestamp, content, content_hash, is_completed)
        VALUES ('task_001', '2025-12-30 10:00:00', 'Test thing', 'hash123', 0)
        """
    )
    conn.commit()

    # First build
    was_built = build_rollup(
        conn,
        kind="d",
        start_date=date(2025, 12, 30),
        end_date=date(2025, 12, 30),
        force=False,
    )

    assert was_built  # Should build on first run

    # Check rollup exists
    cursor.execute("SELECT * FROM rollups WHERE rollup_id = 'd:2025-12-30'")
    rollup = cursor.fetchone()
    assert rollup is not None

    # Second build (no changes)
    was_built2 = build_rollup(
        conn,
        kind="d",
        start_date=date(2025, 12, 30),
        end_date=date(2025, 12, 30),
        force=False,
    )

    assert not was_built2  # Should skip (src_hash matches)

    # Third build with force
    was_built3 = build_rollup(
        conn,
        kind="d",
        start_date=date(2025, 12, 30),
        end_date=date(2025, 12, 30),
        force=True,
    )

    assert was_built3  # Should rebuild with force=True

    conn.close()


def test_build_function(temp_db):
    """Test main build() function."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert test data spanning 2 months
    for i in range(10):
        ts = datetime(2025, 11, i + 1, 10, 0, 0)
        cursor.execute(
            """
            INSERT INTO things (id, timestamp, content, content_hash)
            VALUES (?, ?, ?, ?)
            """,
            (f"task_{i:03d}", ts.isoformat(), f"Test thing {i}", f"hash{i}"),
        )

    for i in range(10, 20):
        ts = datetime(2025, 12, i - 9, 10, 0, 0)
        cursor.execute(
            """
            INSERT INTO things (id, timestamp, content, content_hash)
            VALUES (?, ?, ?, ?)
            """,
            (f"task_{i:03d}", ts.isoformat(), f"Test thing {i}", f"hash{i}"),
        )

    conn.commit()
    conn.close()

    # Build rollups (last 2 months, all kinds)
    result = build(db_path=temp_db, force=False, months=2, kinds="d,w,m")

    assert result["success"]
    assert result["stats"]["rollup_count"] > 0
    assert result["stats"]["day_count"] > 0
    assert result["stats"]["week_count"] > 0
    assert result["stats"]["month_count"] > 0
    assert result["duration_seconds"] > 0


def test_build_empty_window(temp_db):
    """Test that empty windows are skipped."""
    conn = sqlite3.connect(temp_db)

    # No data in database
    was_built = build_rollup(
        conn,
        kind="d",
        start_date=date(2025, 12, 30),
        end_date=date(2025, 12, 30),
        force=False,
    )

    assert not was_built  # Should skip empty windows

    conn.close()


def test_highlight_evidence_storage(temp_db):
    """Test that highlights and evidence are stored correctly."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert 15 things (to get 10 highlights + 5 evidence)
    for i in range(15):
        ts = datetime(2025, 12, 30 - i // 3, 10, i, 0)  # Spread across dates
        cursor.execute(
            """
            INSERT INTO things (id, timestamp, content, content_hash)
            VALUES (?, ?, ?, ?)
            """,
            (f"task_{i:03d}", ts.isoformat(), f"Test content {i}" * 10, f"hash{i}"),
        )

    conn.commit()

    # Build rollup
    build_rollup(
        conn,
        kind="d",
        start_date=date(2025, 12, 28),
        end_date=date(2025, 12, 30),
        force=False,
    )

    # Check highlights (should be top 10)
    cursor.execute(
        """
        SELECT COUNT(*) FROM rollup_evidence
        WHERE rollup_id = 'd:2025-12-28' AND role = 'hi'
        """
    )
    hi_count = cursor.fetchone()[0]
    assert hi_count <= 10

    # Check evidence (should be next 20, but we only have 5 more)
    cursor.execute(
        """
        SELECT COUNT(*) FROM rollup_evidence
        WHERE rollup_id = 'd:2025-12-28' AND role = 'ev'
        """
    )
    ev_count = cursor.fetchone()[0]
    assert ev_count <= 5

    conn.close()


def test_week_id_format():
    """Test that week rollup_id uses Monday date."""
    start = date(2025, 12, 29)  # Monday
    end = date(2026, 1, 4)  # Sunday

    windows = generate_week_windows(start, end)

    # First window should start on Monday
    week_start = windows[0][0]
    assert week_start.weekday() == 0

    # Rollup ID format
    rollup_id = f"w:{week_start.isoformat()}"
    assert rollup_id == "w:2025-12-29"


def test_keyword_extraction_deterministic():
    """Test that keyword extraction is deterministic."""
    things = [
        {
            "id": "task_001",
            "content": "dentist appointment dental checkup",
            "tags": ["health"],
        },
        {
            "id": "task_002",
            "content": "budget review financial planning",
            "tags": ["work", "finance"],
        },
    ]

    # Extract multiple times
    kw1 = extract_keywords(things, top_n=10)
    kw2 = extract_keywords(things, top_n=10)
    kw3 = extract_keywords(things, top_n=10)

    # Should be identical
    assert kw1 == kw2 == kw3

    # Should be sorted
    assert kw1 == sorted(kw1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
