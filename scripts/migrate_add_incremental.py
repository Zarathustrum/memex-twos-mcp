#!/usr/bin/env python3
"""
Migrate existing database to add incremental ingestion support.

This script adds:
- content_hash column to things table
- imports table for audit trail
- Backfills content hashes for existing things

Usage:
    python scripts/migrate_add_incremental.py data/processed/twos.db
"""

import sqlite3
import sys
from pathlib import Path
import hashlib
import json


def compute_content_hash(task):
    """Compute content hash (same as in load_to_sqlite.py)."""
    canonical = {
        "timestamp": task.get("timestamp"),
        "content": task.get("content", "").strip(),
        "section_header": task.get("section_header", "").strip(),
    }
    hash_input = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(hash_input.encode()).hexdigest()


def migrate_database(db_path: Path):
    """Add incremental ingestion support to existing database."""

    print(f"📊 Migrating database: {db_path}")

    if not db_path.exists():
        print(f"❌ Error: Database not found at {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Add content_hash column
    print("  Adding content_hash column...")
    try:
        cursor.execute("ALTER TABLE things ADD COLUMN content_hash TEXT")
        print("    ✅ Column added")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower():
            print("    ℹ️  Column already exists, skipping")
        else:
            print(f"    ❌ Error: {e}")
            raise

    # 2. Create index
    print("  Creating index...")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_things_content_hash ON things(content_hash)"
    )
    print("    ✅ Index created")

    # 3. Backfill hashes
    print("  Backfilling content hashes...")
    cursor.execute(
        "SELECT id, timestamp, content, section_header FROM things WHERE content_hash IS NULL"
    )
    things = cursor.fetchall()

    if things:
        print(f"    Processing {len(things)} things...")
        batch_size = 1000
        for idx, row in enumerate(things, 1):
            task = {
                "timestamp": row["timestamp"],
                "content": row["content"],
                "section_header": row["section_header"],
            }
            content_hash = compute_content_hash(task)

            cursor.execute(
                "UPDATE things SET content_hash = ? WHERE id = ?",
                (content_hash, row["id"]),
            )

            if idx % batch_size == 0:
                conn.commit()
                print(f"      - Processed {idx}/{len(things)} things")

        conn.commit()
        print(f"    ✅ Updated {len(things)} hashes")
    else:
        print("    ℹ️  All things already have content hashes")

    # 4. Create imports table
    print("  Creating imports table...")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS imports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            json_file TEXT,
            mode TEXT NOT NULL,
            imported_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            thing_count INTEGER,
            new_count INTEGER,
            updated_count INTEGER,
            deleted_count INTEGER,
            duration_seconds REAL
        )
    """
    )
    print("    ✅ Table created")

    # 5. Add metadata
    print("  Adding metadata...")
    cursor.execute(
        """
        INSERT OR REPLACE INTO metadata (key, value)
        VALUES ('last_incremental_import', NULL)
    """
    )
    cursor.execute(
        """
        INSERT OR REPLACE INTO metadata (key, value)
        VALUES ('incremental_migration_completed', datetime('now'))
    """
    )
    conn.commit()
    print("    ✅ Metadata updated")

    print("\n✅ Migration complete! Database is now ready for incremental ingestion.")
    print("   You can now use: python scripts/load_to_sqlite.py <file> --mode=append")

    conn.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python migrate_add_incremental.py <db_path>")
        print(
            "Example: python scripts/migrate_add_incremental.py data/processed/twos.db"
        )
        sys.exit(1)

    exit(migrate_database(Path(sys.argv[1])))
