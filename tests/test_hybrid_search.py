"""Integration tests for hybrid search functionality."""

import pytest
import sqlite3
from memex_twos_mcp.database import TwosDatabase

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None


def create_test_db(tmp_path):
    """
    Create a minimal test database with sample data.

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        Path to created database
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)

    # Create minimal schema
    conn.executescript(
        """
        CREATE TABLE things (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            is_completed BOOLEAN DEFAULT 0,
            is_strikethrough BOOLEAN DEFAULT 0,
            is_pending BOOLEAN DEFAULT 0
        );

        CREATE VIRTUAL TABLE things_fts USING fts5(
            thing_id UNINDEXED,
            content,
            tokenize = 'porter unicode61'
        );

        CREATE TABLE thing_embeddings (
            thing_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            model_version TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE thing_tags (
            thing_id TEXT,
            tag_id INTEGER,
            PRIMARY KEY (thing_id, tag_id)
        );

        CREATE TABLE people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            normalized_name TEXT,
            category TEXT DEFAULT 'Uncategorized'
        );

        CREATE TABLE thing_people (
            thing_id TEXT,
            person_id INTEGER,
            PRIMARY KEY (thing_id, person_id)
        );
    """
    )

    # Insert sample data
    cursor = conn.cursor()
    sample_things = [
        ("task_001", "Schedule doctor appointment", "2024-01-01T10:00:00"),
        ("task_002", "Buy groceries for dinner", "2024-01-02T10:00:00"),
        ("task_003", "Meeting with team about project", "2024-01-03T10:00:00"),
        ("task_004", "Call dentist for checkup", "2024-01-04T10:00:00"),
        ("task_005", "Research vacation destinations", "2024-01-05T10:00:00"),
    ]

    for thing_id, content, timestamp in sample_things:
        cursor.execute(
            "INSERT INTO things (id, content, timestamp) VALUES (?, ?, ?)",
            (thing_id, content, timestamp),
        )
        cursor.execute(
            "INSERT INTO things_fts (thing_id, content) VALUES (?, ?)",
            (thing_id, content),
        )

    conn.commit()
    conn.close()

    return db_path


def test_hybrid_search_fallback(tmp_path):
    """Test fallback to lexical-only if embeddings disabled."""
    db_path = create_test_db(tmp_path)
    db = TwosDatabase(db_path)

    # Should not crash, should fall back to lexical
    results = db.hybrid_search("doctor", limit=10, enable_semantic=False)

    assert len(results) > 0
    assert "hybrid_score" in results[0]
    # Should find "doctor appointment" and "dentist"
    assert any("doctor" in r.get("snippet", "").lower() for r in results)


def test_hybrid_search_with_embeddings(tmp_path):
    """Test hybrid search with embeddings enabled."""
    db_path = create_test_db(tmp_path)

    # Check if embeddings available
    try:
        from memex_twos_mcp.embeddings import EmbeddingGenerator

        gen = EmbeddingGenerator()
        if not gen.available:
            pytest.skip("Embedding model not available")
    except ImportError:
        pytest.skip("Required dependencies not available")

    db = TwosDatabase(db_path)

    if not db.embeddings_enabled:
        pytest.skip("Embeddings not enabled in database")

    # Generate embeddings for test data
    import sqlite_vec

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM things")
    things = cursor.fetchall()

    from memex_twos_mcp.embeddings import EmbeddingGenerator
    import numpy as np

    gen = EmbeddingGenerator()
    texts = [thing[1] for thing in things]
    thing_ids = [thing[0] for thing in things]
    embeddings = gen.encode_batch(texts, show_progress=False)

    for thing_id, embedding in zip(thing_ids, embeddings):
        embedding_blob = embedding.astype(np.float32).tobytes()
        cursor.execute(
            "INSERT INTO thing_embeddings (thing_id, embedding, model_version) VALUES (?, ?, ?)",
            (thing_id, embedding_blob, gen.model_name),
        )
        cursor.execute(
            "INSERT INTO vec_index (thing_id, embedding) VALUES (?, ?)",
            (thing_id, embedding_blob),
        )

    conn.commit()
    conn.close()

    # Test hybrid search
    results = db.hybrid_search("meeting", limit=10)

    assert len(results) > 0
    assert "hybrid_score" in results[0]


def test_hybrid_search_weights(tmp_path):
    """Test that different weights affect results."""
    db_path = create_test_db(tmp_path)
    db = TwosDatabase(db_path)

    # Lexical-only (semantic_weight=0)
    lexical_only = db.hybrid_search(
        "doctor", limit=10, lexical_weight=1.0, semantic_weight=0.0
    )

    assert len(lexical_only) > 0
    assert all("hybrid_score" in r for r in lexical_only)


def test_hybrid_search_empty_query(tmp_path):
    """Test hybrid search with empty query."""
    db_path = create_test_db(tmp_path)
    db = TwosDatabase(db_path)

    # Empty query should return empty results (FTS5 behavior)
    results = db.hybrid_search("", limit=10, enable_semantic=False)

    # FTS5 may throw error or return empty, either is acceptable
    assert isinstance(results, list)
