"""Tests for incremental ingestion (Phase 5)."""

import pytest
import sqlite3
import tempfile
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_to_sqlite import (
    compute_content_hash,
    incremental_load,
    create_database,
    load_tasks,
    load_people,
    load_tags,
    load_links,
)


@pytest.fixture
def schema_path():
    """Get path to schema file."""
    return Path(__file__).parent.parent / "schema" / "schema.sql"


@pytest.fixture
def temp_db(schema_path):
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = create_database(db_path, schema_path)
    conn.close()

    yield db_path

    # Cleanup
    if db_path.exists():
        db_path.unlink()


class TestContentHash:
    """Tests for content hash computation."""

    def test_compute_content_hash_deterministic(self):
        """Test hash stability."""
        task = {
            'timestamp': '2024-01-01T10:00:00',
            'content': 'Test task',
            'section_header': 'Mon, Jan 1',
            'line_number': 42  # Should not affect hash
        }

        hash1 = compute_content_hash(task)
        hash2 = compute_content_hash(task)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex

    def test_content_hash_ignores_formatting(self):
        """Test that formatting fields don't affect hash."""
        task1 = {
            'timestamp': '2024-01-01T10:00:00',
            'content': 'Test task',
            'section_header': 'Mon, Jan 1',
            'line_number': 42,
            'content_raw': '• Test task',
        }

        task2 = {
            'timestamp': '2024-01-01T10:00:00',
            'content': 'Test task',
            'section_header': 'Mon, Jan 1',
            'line_number': 100,  # Different line number
            'content_raw': '- Test task',  # Different raw format
        }

        assert compute_content_hash(task1) == compute_content_hash(task2)

    def test_content_hash_detects_content_changes(self):
        """Test that content changes produce different hashes."""
        task1 = {
            'timestamp': '2024-01-01T10:00:00',
            'content': 'Test task',
            'section_header': 'Mon, Jan 1',
        }

        task2 = {
            'timestamp': '2024-01-01T10:00:00',
            'content': 'Modified task',  # Different content
            'section_header': 'Mon, Jan 1',
        }

        assert compute_content_hash(task1) != compute_content_hash(task2)

    def test_content_hash_strips_whitespace(self):
        """Test that whitespace differences don't affect hash."""
        task1 = {
            'timestamp': '2024-01-01T10:00:00',
            'content': 'Test task',
            'section_header': 'Mon, Jan 1',
        }

        task2 = {
            'timestamp': '2024-01-01T10:00:00',
            'content': '  Test task  ',  # Extra whitespace
            'section_header': '  Mon, Jan 1  ',
        }

        assert compute_content_hash(task1) == compute_content_hash(task2)


class TestIncrementalLoad:
    """Tests for incremental loading."""

    def _create_test_db(self, db_path, tasks):
        """Helper to create test database with initial data."""
        conn = sqlite3.connect(db_path)
        load_tasks(conn, tasks)
        load_people(conn, tasks)
        load_tags(conn, tasks)
        load_links(conn, tasks)
        conn.commit()
        conn.close()

    def test_incremental_insert_new(self, temp_db):
        """Test inserting new things in incremental mode."""
        # Initial load with 2 things
        tasks1 = [
            {
                'id': 'task_001',
                'timestamp': '2024-01-01T10:00:00',
                'content': 'Task 1',
                'section_header': 'Day 1',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
            {
                'id': 'task_002',
                'timestamp': '2024-01-02T10:00:00',
                'content': 'Task 2',
                'section_header': 'Day 2',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
        ]
        self._create_test_db(temp_db, tasks1)

        # Incremental load with 1 new task
        tasks2 = tasks1 + [
            {
                'id': 'task_003',
                'timestamp': '2024-01-03T10:00:00',
                'content': 'Task 3',
                'section_header': 'Day 3',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
        ]

        conn = sqlite3.connect(temp_db)
        stats = incremental_load(conn, tasks2, 'test.md', 'test.json', mode='append')
        conn.close()

        assert stats['new_count'] == 1
        assert stats['updated_count'] == 0
        assert stats['deleted_count'] == 0

        # Verify database has 3 things
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM things")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 3

    def test_incremental_update_changed(self, temp_db):
        """Test updating changed things."""
        # Initial load
        tasks1 = [
            {
                'id': 'task_001',
                'timestamp': '2024-01-01T10:00:00',
                'content': 'Original content',
                'section_header': 'Day 1',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
        ]
        self._create_test_db(temp_db, tasks1)

        # Change content
        tasks2 = [
            {
                'id': 'task_001',
                'timestamp': '2024-01-01T10:00:00',
                'content': 'Updated content',  # Changed!
                'section_header': 'Day 1',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
        ]

        conn = sqlite3.connect(temp_db)
        stats = incremental_load(conn, tasks2, 'test.md', 'test.json', mode='sync')
        conn.close()

        assert stats['new_count'] == 0
        assert stats['updated_count'] == 1
        assert stats['deleted_count'] == 0

        # Verify content updated
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM things WHERE id = 'task_001'")
        content = cursor.fetchone()[0]
        conn.close()

        assert content == 'Updated content'

    def test_incremental_delete_sync_mode(self, temp_db):
        """Test deleting removed things in sync mode."""
        # Initial load with 3 things
        tasks1 = [
            {
                'id': 'task_001',
                'content': 'Task 1',
                'timestamp': '2024-01-01T10:00:00',
                'section_header': 'Day 1',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
            {
                'id': 'task_002',
                'content': 'Task 2',
                'timestamp': '2024-01-02T10:00:00',
                'section_header': 'Day 2',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
            {
                'id': 'task_003',
                'content': 'Task 3',
                'timestamp': '2024-01-03T10:00:00',
                'section_header': 'Day 3',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
        ]
        self._create_test_db(temp_db, tasks1)

        # Load with only 2 things (task_003 removed)
        tasks2 = tasks1[:2]

        conn = sqlite3.connect(temp_db)
        stats = incremental_load(conn, tasks2, 'test.md', 'test.json', mode='sync')
        conn.close()

        assert stats['deleted_count'] == 1

        # Verify task_003 is gone
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM things WHERE id = 'task_003'")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 0

    def test_append_mode_does_not_delete(self, temp_db):
        """Test that append mode doesn't delete removed things."""
        # Initial load with 3 things
        tasks1 = [
            {
                'id': 'task_001',
                'content': 'Task 1',
                'timestamp': '2024-01-01T10:00:00',
                'section_header': 'Day 1',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
            {
                'id': 'task_002',
                'content': 'Task 2',
                'timestamp': '2024-01-02T10:00:00',
                'section_header': 'Day 2',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
            {
                'id': 'task_003',
                'content': 'Task 3',
                'timestamp': '2024-01-03T10:00:00',
                'section_header': 'Day 3',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
        ]
        self._create_test_db(temp_db, tasks1)

        # Load with only 2 things (task_003 removed from file)
        tasks2 = tasks1[:2]

        conn = sqlite3.connect(temp_db)
        stats = incremental_load(conn, tasks2, 'test.md', 'test.json', mode='append')
        conn.close()

        # In append mode, nothing should be deleted
        assert stats['deleted_count'] == 0

        # Verify task_003 still exists
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM things WHERE id = 'task_003'")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_fts_stays_in_sync(self, temp_db):
        """Test that FTS index updates with incremental changes."""
        # Initial load
        tasks1 = [
            {
                'id': 'task_001',
                'content': 'doctor appointment',
                'timestamp': '2024-01-01T10:00:00',
                'section_header': 'Day 1',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
        ]
        self._create_test_db(temp_db, tasks1)

        # Update content
        tasks2 = [
            {
                'id': 'task_001',
                'content': 'dentist appointment',  # Changed!
                'timestamp': '2024-01-01T10:00:00',
                'section_header': 'Day 1',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
        ]

        conn = sqlite3.connect(temp_db)
        incremental_load(conn, tasks2, 'test.md', 'test.json', mode='sync')

        # Search should find new content, not old
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM things_fts WHERE things_fts MATCH 'dentist'")
        dentist_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM things_fts WHERE things_fts MATCH 'doctor'")
        doctor_count = cursor.fetchone()[0]

        conn.close()

        assert dentist_count == 1  # New content in FTS
        assert doctor_count == 0   # Old content gone from FTS

    def test_import_audit_trail(self, temp_db):
        """Test that import runs are tracked in imports table."""
        tasks = [
            {
                'id': 'task_001',
                'content': 'Task 1',
                'timestamp': '2024-01-01T10:00:00',
                'section_header': 'Day 1',
                'tags': [],
                'people_mentioned': [],
                'links': []
            },
        ]

        conn = sqlite3.connect(temp_db)
        incremental_load(conn, tasks, 'test.md', 'test.json', mode='append')

        # Check imports table
        cursor = conn.cursor()
        cursor.execute("SELECT mode, new_count FROM imports ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == 'append'
        assert row[1] == 1  # 1 new thing

    def test_tags_update_incrementally(self, temp_db):
        """Test that tags are updated for changed things."""
        # Initial load
        tasks1 = [
            {
                'id': 'task_001',
                'content': 'Task 1',
                'timestamp': '2024-01-01T10:00:00',
                'section_header': 'Day 1',
                'tags': ['work'],
                'people_mentioned': [],
                'links': []
            },
        ]
        self._create_test_db(temp_db, tasks1)

        # Update with different tags AND content (content must change for hash to detect change)
        tasks2 = [
            {
                'id': 'task_001',
                'content': 'Task 1 updated',  # Changed content!
                'timestamp': '2024-01-01T10:00:00',
                'section_header': 'Day 1',
                'tags': ['personal', 'urgent'],  # Changed tags!
                'people_mentioned': [],
                'links': []
            },
        ]

        conn = sqlite3.connect(temp_db)
        incremental_load(conn, tasks2, 'test.md', 'test.json', mode='sync')

        # Check tags
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.name FROM tags t
            JOIN thing_tags tt ON t.id = tt.tag_id
            WHERE tt.thing_id = 'task_001'
            ORDER BY t.name
        """)
        tags = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert tags == ['personal', 'urgent']
        assert 'work' not in tags

    def test_people_update_incrementally(self, temp_db):
        """Test that people are updated for changed things."""
        # Initial load
        tasks1 = [
            {
                'id': 'task_001',
                'content': 'Task 1',
                'timestamp': '2024-01-01T10:00:00',
                'section_header': 'Day 1',
                'tags': [],
                'people_mentioned': ['Alice'],
                'links': []
            },
        ]
        self._create_test_db(temp_db, tasks1)

        # Update with different people AND content (content must change for hash to detect change)
        tasks2 = [
            {
                'id': 'task_001',
                'content': 'Task 1 updated',  # Changed content!
                'timestamp': '2024-01-01T10:00:00',
                'section_header': 'Day 1',
                'tags': [],
                'people_mentioned': ['Bob', 'Charlie'],  # Changed people!
                'links': []
            },
        ]

        conn = sqlite3.connect(temp_db)
        incremental_load(conn, tasks2, 'test.md', 'test.json', mode='sync')

        # Check people
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.name FROM people p
            JOIN thing_people tp ON p.id = tp.person_id
            WHERE tp.thing_id = 'task_001'
            ORDER BY p.name
        """)
        people = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert people == ['Bob', 'Charlie']
        assert 'Alice' not in people


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
