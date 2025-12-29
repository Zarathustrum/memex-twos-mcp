#!/usr/bin/env python3
"""
Migrate existing database to add embeddings support.

This script adds embeddings to an existing database that was created
before Phase 4. It:
- Creates the thing_embeddings table
- Initializes the vec_index virtual table
- Generates embeddings for all existing things
- Stores embeddings in both tables
"""

import sqlite3
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def migrate_database(db_path: Path):
    """Add embeddings to existing database."""

    print(f"📊 Migrating database: {db_path}")

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 1. Add embeddings table
    print("  Creating thing_embeddings table...")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS thing_embeddings (
            thing_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            model_version TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_thing ON thing_embeddings(thing_id)")
    conn.commit()

    # 2. Initialize sqlite-vec extension and create vec_index
    print("  Initializing vector search extension...")
    try:
        import sqlite_vec
        import numpy as np

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0(
                thing_id TEXT PRIMARY KEY,
                embedding float[384]
            )
        """)
        conn.commit()
    except ImportError:
        print("❌ sqlite-vec not installed. Install with: pip install sqlite-vec")
        conn.close()
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to initialize vector search: {e}")
        conn.close()
        sys.exit(1)

    # 3. Fetch all things
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM things")
    things = cursor.fetchall()

    print(f"  Found {len(things)} things to embed")

    # 4. Generate embeddings
    try:
        from memex_twos_mcp.embeddings import EmbeddingGenerator

        gen = EmbeddingGenerator()
        if not gen.available:
            print("❌ Embedding model not available. Install with: pip install sentence-transformers")
            conn.close()
            sys.exit(1)

        texts = [thing['content'] for thing in things]
        thing_ids = [thing['id'] for thing in things]

        print("  Generating embeddings...")
        embeddings = gen.encode_batch(texts, show_progress=True)

        # 5. Store embeddings
        print("  Storing embeddings...")
        for thing_id, embedding in zip(thing_ids, embeddings):
            embedding_blob = embedding.astype(np.float32).tobytes()

            # Insert into thing_embeddings table
            cursor.execute("""
                INSERT OR REPLACE INTO thing_embeddings (thing_id, embedding, model_version)
                VALUES (?, ?, ?)
            """, (thing_id, embedding_blob, gen.model_name))

            # Insert into vec_index for fast search
            cursor.execute("""
                INSERT OR REPLACE INTO vec_index (thing_id, embedding)
                VALUES (?, ?)
            """, (thing_id, embedding_blob))

        conn.commit()
        print("✅ Migration complete!")
        print(f"   Generated {len(embeddings)} embeddings")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python migrate_add_embeddings.py <db_path>")
        print("Example: python migrate_add_embeddings.py data/processed/twos.db")
        sys.exit(1)

    migrate_database(Path(sys.argv[1]))
