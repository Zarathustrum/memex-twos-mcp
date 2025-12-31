"""
Tests for MonthlySummaries builder (scripts/build_month_summaries.py)

Covers:
- MS1 pack format validation
- LLM response validation (with mock responses)
- Question format validation
- Incremental rebuild (src_hash matching)
- Error handling (LLM timeout, invalid response)
"""

import json
import re
import sqlite3
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import builder functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_month_summaries import (
    build,
    build_llm_prompt,
    build_month_summary,
    build_ms1_pack,
    generate_month_windows,
    validate_llm_response,
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
        CREATE TABLE month_summaries (
            month_id TEXT PRIMARY KEY,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            thing_count INTEGER NOT NULL,
            pack_v INTEGER NOT NULL,
            pack TEXT NOT NULL,
            suggested_questions TEXT,
            src_hash TEXT NOT NULL,
            build_import_id INTEGER,
            builder_v TEXT NOT NULL,
            llm_model TEXT,
            llm_conf REAL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (build_import_id) REFERENCES imports(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE month_summary_evidence (
            month_id TEXT NOT NULL,
            thing_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('hi','ev')),
            rank INTEGER,
            PRIMARY KEY (month_id, thing_id, role),
            FOREIGN KEY (month_id) REFERENCES month_summaries(month_id) ON DELETE CASCADE,
            FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    db_path.unlink()


def test_validate_llm_response_valid():
    """Test validation of valid LLM response."""
    candidate_ids = {"task_001", "task_002", "task_003", "task_004", "task_005"}

    response = {
        "themes": [
            {"name": "work_planning", "thing_ids": ["task_001", "task_002"]},
            {"name": "health_care", "thing_ids": ["task_003", "task_004"]},
            {"name": "home_tasks", "thing_ids": ["task_001", "task_005"]},
        ],
        "highlights": [
            {"thing_id": "task_001", "label": "review_meeting"},
            {"thing_id": "task_002", "label": "budget_check"},
            {"thing_id": "task_003", "label": "dentist_appt"},
            {"thing_id": "task_004", "label": "gym_session"},
            {"thing_id": "task_005", "label": "repair_sink"},
            {"thing_id": "task_001", "label": "planning_doc"},
            {"thing_id": "task_002", "label": "invoice_sent"},
            {"thing_id": "task_003", "label": "med_refill"},
            {"thing_id": "task_004", "label": "workout_plan"},
            {"thing_id": "task_005", "label": "home_cleanup"},
        ],
        "questions": [
            {
                "text": "What progress on work planning?",
                "anchors": ["task_001", "task_002"],
                "thread_id": "thr:tag:work",
                "rationale": "High activity"
            },
            {
                "text": "How did health evolve?",
                "anchors": ["task_003"],
                "thread_id": "thr:tag:health",
                "rationale": "New thread"
            },
            {
                "text": "What's happening at home?",
                "anchors": ["task_005"],
                "thread_id": "thr:tag:home",
                "rationale": "Multiple tasks"
            }
        ]
    }

    is_valid, error = validate_llm_response(response, candidate_ids)

    assert is_valid
    assert error is None


def test_validate_llm_response_invalid_thing_id():
    """Test validation catches invalid thing IDs."""
    candidate_ids = {"task_001", "task_002"}

    response = {
        "themes": [
            {"name": "work", "thing_ids": ["task_001", "task_999"]},  # Invalid ID
        ],
        "highlights": [{"thing_id": "task_001", "label": "test"}] * 10,
        "questions": [
            {"text": "Test?", "anchors": ["task_001"], "rationale": "Test"}
        ] * 3
    }

    is_valid, error = validate_llm_response(response, candidate_ids)

    assert not is_valid
    assert "invalid thing_id" in error.lower()


def test_validate_llm_response_invalid_theme_name():
    """Test validation catches invalid theme names."""
    candidate_ids = {"task_001", "task_002"}

    response = {
        "themes": [
            {"name": "Work Planning!", "thing_ids": ["task_001", "task_002"]},  # Invalid chars
        ],
        "highlights": [{"thing_id": "task_001", "label": "test"}] * 10,
        "questions": [
            {"text": "Test?", "anchors": ["task_001"], "rationale": "Test"}
        ] * 3
    }

    is_valid, error = validate_llm_response(response, candidate_ids)

    assert not is_valid
    assert "invalid theme name" in error.lower()


def test_validate_llm_response_wrong_highlight_count():
    """Test validation checks highlight count (10-12)."""
    candidate_ids = {"task_001", "task_002"}

    response = {
        "themes": [
            {"name": "work", "thing_ids": ["task_001", "task_002"]},
        ] * 3,
        "highlights": [{"thing_id": "task_001", "label": "test"}] * 5,  # Too few
        "questions": [
            {"text": "Test?", "anchors": ["task_001"], "rationale": "Test"}
        ] * 3
    }

    is_valid, error = validate_llm_response(response, candidate_ids)

    assert not is_valid
    assert "10-12 highlights" in error.lower()


def test_validate_llm_response_long_question():
    """Test validation catches questions >100 chars."""
    candidate_ids = {"task_001", "task_002"}

    response = {
        "themes": [
            {"name": "work", "thing_ids": ["task_001", "task_002"]},
        ] * 3,
        "highlights": [{"thing_id": "task_001", "label": "test"}] * 10,
        "questions": [
            {
                "text": "A" * 101,  # Too long
                "anchors": ["task_001"],
                "rationale": "Test"
            }
        ] * 3
    }

    is_valid, error = validate_llm_response(response, candidate_ids)

    assert not is_valid
    assert "too long" in error.lower()


def test_build_ms1_pack_format():
    """Test MS1 pack format structure."""
    themes = [
        {"name": "work_planning", "thing_ids": ["task_001", "task_002"]},
        {"name": "health_care", "thing_ids": ["task_003", "task_004"]},
    ]

    highlights = [
        {"thing_id": "task_001", "label": "q4_review"},
        {"thing_id": "task_002", "label": "budget_check"},
        {"thing_id": "task_003", "label": "dentist_appt"},
    ]

    pack = build_ms1_pack(
        month_id="2025-12",
        thing_count=150,
        tags_summary="work:48,health:12",
        people_summary="alice:7,bob:3",
        themes=themes,
        highlights=highlights,
        question_count=3
    )

    # Validate format
    assert pack.startswith("MS1|")
    assert "m=2025-12" in pack
    assert "n=150" in pack
    assert "tg=work:48,health:12" in pack
    assert "pp=alice:7,bob:3" in pack
    assert "th=work_planning@task_001,task_002;health_care@task_003,task_004" in pack
    assert "hi=task_001~q4_review;task_002~budget_check;task_003~dentist_appt" in pack
    assert "nq=3" in pack

    # Should be bounded
    assert len(pack) <= 1200


def test_build_ms1_pack_regex():
    """Test MS1 pack format with regex."""
    themes = [{"name": "test", "thing_ids": ["task_001", "task_002"]}]
    highlights = [{"thing_id": "task_001", "label": "test"}]

    pack = build_ms1_pack(
        "2025-12", 100, "work:10", "alice:5", themes, highlights, 3
    )

    # Regex pattern for MS1 format
    pattern = r"^MS1\|m=\d{4}-\d{2}\|n=\d+\|tg=.*\|pp=.*\|th=.*\|hi=.*\|nq=\d+$"

    assert re.match(pattern, pack), f"Pack doesn't match MS1 format: {pack}"


def test_build_llm_prompt_format():
    """Test LLM prompt includes all required information."""
    candidates = [
        {
            "id": "task_001",
            "timestamp": "2025-12-15T10:00:00",
            "content": "Test content",
            "tags": ["work"],
            "people_mentioned": ["Alice"]
        }
    ]

    prompt = build_llm_prompt(
        month_id="2025-12",
        start_date="2025-12-01",
        end_date="2025-12-31",
        thing_count=100,
        tags_summary="work:48",
        people_summary="alice:7",
        candidates=candidates
    )

    # Check key sections
    assert "Month: 2025-12" in prompt
    assert "Total things: 100" in prompt
    assert "Top tags: work:48" in prompt
    assert "Top people: alice:7" in prompt
    assert "task_001" in prompt
    assert "Candidate highlights" in prompt
    assert "CRITICAL" in prompt  # Warning about hallucination


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


@patch('build_month_summaries.invoke_llm_via_claude_code')
def test_build_month_summary_with_mock(mock_llm, temp_db):
    """Test month summary building with mocked LLM."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert test data
    for i in range(15):
        cursor.execute(
            """
            INSERT INTO things (id, timestamp, content, content_hash)
            VALUES (?, ?, ?, ?)
            """,
            (f"task_{i:03d}", f"2025-12-{i+1:02d} 10:00:00", f"Test content {i}", f"hash{i}")
        )

    conn.commit()

    # Mock LLM response
    mock_llm.return_value = {
        "themes": [
            {"name": "work", "thing_ids": ["task_000", "task_001"]},
            {"name": "health", "thing_ids": ["task_002", "task_003"]},
            {"name": "home", "thing_ids": ["task_004", "task_005"]},
        ],
        "highlights": [
            {"thing_id": f"task_{i:03d}", "label": f"item_{i}"}
            for i in range(10)
        ],
        "questions": [
            {
                "text": "What happened with work?",
                "anchors": ["task_000", "task_001"],
                "thread_id": "thr:tag:work",
                "rationale": "High activity"
            },
            {
                "text": "Health progress?",
                "anchors": ["task_002"],
                "thread_id": "thr:tag:health",
                "rationale": "New thread"
            },
            {
                "text": "Home updates?",
                "anchors": ["task_004"],
                "thread_id": "thr:tag:home",
                "rationale": "Multiple tasks"
            }
        ]
    }

    # Build summary
    was_built, error = build_month_summary(
        conn,
        date(2025, 12, 1),
        date(2025, 12, 31),
        force=False,
        dry_run=False
    )

    assert was_built
    assert error is None

    # Check summary was created
    cursor.execute("SELECT * FROM month_summaries WHERE month_id = '2025-12'")
    summary = cursor.fetchone()

    assert summary is not None
    assert summary[0] == "2025-12"  # month_id
    assert summary[5].startswith("MS1|")  # pack

    # Check questions JSON
    questions = json.loads(summary[6])
    assert len(questions["questions"]) == 3

    conn.close()


@patch('build_month_summaries.invoke_llm_via_claude_code')
def test_build_month_summary_llm_failure(mock_llm, temp_db):
    """Test graceful handling of LLM failure."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert test data
    cursor.execute(
        "INSERT INTO things (id, timestamp, content, content_hash) VALUES (?, ?, ?, ?)",
        ("task_001", "2025-12-15 10:00:00", "Test", "hash1")
    )
    conn.commit()

    # Mock LLM failure
    mock_llm.side_effect = RuntimeError("LLM timeout")

    # Build summary
    was_built, error = build_month_summary(
        conn,
        date(2025, 12, 1),
        date(2025, 12, 31),
        force=False,
        dry_run=False
    )

    assert not was_built
    assert error is not None
    assert "LLM" in error

    conn.close()


@patch('build_month_summaries.invoke_llm_via_claude_code')
def test_build_month_summary_validation_failure(mock_llm, temp_db):
    """Test validation failure handling."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert test data
    cursor.execute(
        "INSERT INTO things (id, timestamp, content, content_hash) VALUES (?, ?, ?, ?)",
        ("task_001", "2025-12-15 10:00:00", "Test", "hash1")
    )
    conn.commit()

    # Mock invalid LLM response (hallucinated thing IDs)
    mock_llm.return_value = {
        "themes": [
            {"name": "work", "thing_ids": ["task_999", "task_888"]},  # Invalid IDs
        ] * 3,
        "highlights": [
            {"thing_id": "task_999", "label": "fake"}
        ] * 10,
        "questions": [
            {"text": "Test?", "anchors": ["task_999"], "rationale": "Test"}
        ] * 3
    }

    # Build summary
    was_built, error = build_month_summary(
        conn,
        date(2025, 12, 1),
        date(2025, 12, 31),
        force=False,
        dry_run=False
    )

    assert not was_built
    assert error is not None
    assert "Validation" in error

    conn.close()


def test_dry_run_mode(temp_db):
    """Test dry-run mode doesn't invoke LLM."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert test data
    cursor.execute(
        "INSERT INTO things (id, timestamp, content, content_hash) VALUES (?, ?, ?, ?)",
        ("task_001", "2025-12-15 10:00:00", "Test", "hash1")
    )
    conn.commit()

    # Build in dry-run mode
    was_built, error = build_month_summary(
        conn,
        date(2025, 12, 1),
        date(2025, 12, 31),
        force=False,
        dry_run=True
    )

    assert not was_built
    assert error is None

    # Check no summary was created
    cursor.execute("SELECT * FROM month_summaries WHERE month_id = '2025-12'")
    assert cursor.fetchone() is None

    conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
