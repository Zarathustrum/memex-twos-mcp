#!/usr/bin/env python3
"""
Load converted Twos JSON data into SQLite database.

I/O Boundary: Reads JSON from disk, writes to SQLite database file.

This script operates in three modes:
1. rebuild (default): Delete all data, full reload (use for first import)
2. append: Insert new items only (use for daily updates, safest)
3. sync: Insert new, update changed, delete removed (use when export is authoritative)

Mode selection:
- Rebuild: Always safe, but slow for large datasets
- Append: Fast incremental, but won't detect changes/deletions
- Sync: Full incremental, requires stable IDs from convert_to_json.py

Incremental mode relies on:
- Stable content-hash based IDs (not sequential counters)
- Matching hash computation in convert_to_json.py
- Foreign key constraints enabled for CASCADE deletions

This script:
- Creates the database schema
- Loads all things
- Populates normalized entity tables (people, tags, links)
- Creates relationships via junction tables
- Populates FTS index
- Generates embeddings for semantic search (optional)
"""

import json
import sqlite3
import os
import hashlib
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, Any, List


def compute_content_hash(task: Dict[str, Any]) -> str:
    """
    Compute stable content hash for change detection.

    Hash includes: timestamp, content, section_header (canonical fields only).
    Excludes: line_number, content_raw (formatting/position don't matter).

    Args:
        task: Thing dictionary

    Returns:
        SHA256 hex digest (64 chars)
    """
    # Use only canonical fields that matter for content identity
    canonical = {
        "timestamp": task.get("timestamp"),
        "content": task.get("content", "").strip(),
        "section_header": task.get("section_header", "").strip(),
        # Note: Not including tags/people (they're junction tables, updated separately)
    }

    # Sort keys for deterministic JSON
    hash_input = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(hash_input.encode()).hexdigest()


def create_database(db_path: Path, schema_path: Path):
    """
    Create database and apply schema.

    Irreversible action: Overwrites existing database file at db_path.

    Foreign key enforcement:
    - SQLite disables foreign keys by default for backwards compatibility.
    - Without "PRAGMA foreign_keys = ON", CASCADE deletions don't work.
    - This would leave orphaned rows in junction tables (thing_tags, thing_people, etc).
    - Must be set per-connection, not per-database.

    Args:
        db_path: Path where the SQLite file will be created.
        schema_path: Path to a SQL schema file.

    Returns:
        A live sqlite3.Connection to the new database.
    """
    print(f"Creating database at {db_path}")

    # I/O boundary: read schema SQL from disk.
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    # I/O boundary: create or overwrite the database file on disk.
    conn = sqlite3.connect(db_path)

    # Enable foreign key constraints (required for proper CASCADE behavior)
    # Must be set before schema creation for best compatibility
    conn.execute("PRAGMA foreign_keys = ON")

    conn.executescript(schema_sql)
    conn.commit()

    print("Schema applied successfully")
    return conn


def load_tasks(conn: sqlite3.Connection, tasks: list) -> None:
    """
    Load things into the things table.

    Args:
        conn: Open SQLite connection.
        tasks: List of thing dictionaries from the JSON export.

    Returns:
        None.
    """
    print(f"Loading {len(tasks)} things...")

    cursor = conn.cursor()
    batch_size = 1000  # Commit every 1000 inserts for performance + progress tracking

    for index, task in enumerate(tasks, start=1):
        # Compute content hash for change detection
        content_hash = compute_content_hash(task)

        # Parameterized INSERT protects against SQL injection and quoting issues.
        cursor.execute(
            """
            INSERT INTO things (
                id, timestamp, timestamp_raw, content, content_raw, content_hash,
                section_header, section_date, line_number, indent_level,
                parent_task_id, bullet_type, is_completed, is_pending,
                is_strikethrough, item_type, list_id, is_substantive
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task["id"],
                task.get("timestamp"),
                task.get("timestamp_raw"),
                task.get("content"),
                task.get("content_raw"),
                content_hash,
                task.get("section_header"),
                task.get("section_date"),
                task.get("line_number"),
                task.get("indent_level", 0),
                task.get("parent_task_id"),
                task.get("bullet_type"),
                task.get("is_completed", False),
                task.get("is_pending", False),
                task.get("is_strikethrough", False),
                task.get("item_type", "content"),
                task.get("list_id"),
                task.get("is_substantive", True),
            ),
        )

        # Batch commit for better performance (10x faster than commit-per-insert)
        if index % batch_size == 0:
            conn.commit()
            print(f"  - Inserted {index} things (batch commit)")

    # Final commit for any remaining records
    conn.commit()
    print(f"Loaded {len(tasks)} things")


def load_people(conn: sqlite3.Connection, tasks: list):
    """
    Extract and load people, then create thing-people relationships.

    This populates the normalized `people` table and the junction table
    that links things to people.

    Args:
        conn: Open SQLite connection.
        tasks: List of thing dictionaries from the JSON export.

    Returns:
        None.
    """
    print("Loading people and relationships...")

    cursor = conn.cursor()

    # Collect all unique people
    people_set = set()
    for task in tasks:
        for person in task.get("people_mentioned", []):
            people_set.add(person)

    # Insert people
    for person in sorted(people_set):
        cursor.execute(
            """
            INSERT OR IGNORE INTO people (name, normalized_name)
            VALUES (?, ?)
        """,
            (person, person.lower()),
        )

    conn.commit()
    print(f"Loaded {len(people_set)} unique people")

    # Create thing-people relationships by mapping names to IDs.
    person_id_map = {}
    cursor.execute("SELECT id, name FROM people")
    for person_id, name in cursor.fetchall():
        person_id_map[name] = person_id

    relationship_count = 0
    for task in tasks:
        task_id = task["id"]
        for person in task.get("people_mentioned", []):
            person_id = person_id_map.get(person)
            if person_id:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO thing_people (thing_id, person_id)
                    VALUES (?, ?)
                """,
                    (task_id, person_id),
                )
                relationship_count += 1

    conn.commit()
    print(f"Created {relationship_count} thing-people relationships")


def load_tags(conn: sqlite3.Connection, tasks: list):
    """
    Extract and load tags, then create thing-tag relationships.

    Tags are normalized to lowercase during insert.

    Args:
        conn: Open SQLite connection.
        tasks: List of thing dictionaries from the JSON export.

    Returns:
        None.
    """
    print("Loading tags and relationships...")

    cursor = conn.cursor()

    # Collect all unique tags
    tags_set = set()
    for task in tasks:
        for tag in task.get("tags", []):
            tags_set.add(tag)

    # Insert tags
    for tag in sorted(tags_set):
        cursor.execute(
            """
            INSERT OR IGNORE INTO tags (name)
            VALUES (?)
        """,
            (tag,),
        )

    conn.commit()
    print(f"Loaded {len(tags_set)} unique tags")

    # Create thing-tag relationships by mapping tag names to IDs.
    tag_id_map = {}
    cursor.execute("SELECT id, name FROM tags")
    for tag_id, name in cursor.fetchall():
        tag_id_map[name] = tag_id

    relationship_count = 0
    for task in tasks:
        task_id = task["id"]
        for tag in task.get("tags", []):
            tag_id = tag_id_map.get(tag)
            if tag_id:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO thing_tags (thing_id, tag_id)
                    VALUES (?, ?)
                """,
                    (task_id, tag_id),
                )
                relationship_count += 1

    conn.commit()
    print(f"Created {relationship_count} thing-tag relationships")


def load_links(conn: sqlite3.Connection, tasks: list):
    """
    Extract and load links.

    Links are stored as-is and associated with their thing IDs.

    Args:
        conn: Open SQLite connection.
        tasks: List of thing dictionaries from the JSON export.

    Returns:
        None.
    """
    print("Loading links...")

    cursor = conn.cursor()

    link_count = 0
    for task in tasks:
        task_id = task["id"]
        for link in task.get("links", []):
            # Each link becomes one row in the links table.
            cursor.execute(
                """
            INSERT INTO links (thing_id, link_text, url)
            VALUES (?, ?, ?)
            """,
                (task_id, link.get("text"), link.get("url")),
            )
            link_count += 1

    conn.commit()
    print(f"Loaded {link_count} links")


def load_lists(conn: sqlite3.Connection, lists_metadata: list):
    """
    Load lists metadata into the lists table.

    Args:
        conn: Open SQLite connection.
        lists_metadata: List of list dictionaries.

    Returns:
        None.
    """
    if not lists_metadata:
        print("No lists metadata to load")
        return

    print(f"Loading {len(lists_metadata)} lists...")

    cursor = conn.cursor()

    for list_entry in lists_metadata:
        cursor.execute(
            """
            INSERT OR REPLACE INTO lists (
                list_id, list_type, list_name, list_name_raw, list_date,
                start_line, end_line, item_count, substantive_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                list_entry["list_id"],
                list_entry["list_type"],
                list_entry["list_name"],
                list_entry["list_name_raw"],
                list_entry.get("list_date"),
                list_entry["start_line"],
                list_entry["end_line"],
                list_entry.get("item_count", 0),
                list_entry.get("substantive_count", 0),
                list_entry.get("created_at", datetime.now().isoformat()),
            ),
        )

    conn.commit()
    print(f"Loaded {len(lists_metadata)} lists")


def load_thing_lists(conn: sqlite3.Connection, tasks: list):
    """
    Load thing-list relationships into the thing_lists junction table.

    Args:
        conn: Open SQLite connection.
        tasks: List of thing dictionaries with list_id assigned.

    Returns:
        None.
    """
    print("Loading thing-list relationships...")

    cursor = conn.cursor()

    # Group tasks by list_id to assign positions
    list_items = {}
    for task in tasks:
        list_id = task.get("list_id")
        if list_id:
            if list_id not in list_items:
                list_items[list_id] = []
            list_items[list_id].append(task)

    # Insert relationships with positions
    relationship_count = 0
    for list_id, items in list_items.items():
        # Sort by line_number within list
        items_sorted = sorted(items, key=lambda t: t.get("line_number", 0))

        for position, task in enumerate(items_sorted):
            cursor.execute(
                """
                INSERT OR IGNORE INTO thing_lists (thing_id, list_id, position_in_list)
                VALUES (?, ?, ?)
            """,
                (task["id"], list_id, position),
            )
            relationship_count += 1

    conn.commit()
    print(f"Created {relationship_count} thing-list relationships")


def update_metadata(
    conn: sqlite3.Connection,
    source_file: str | None,
    thing_count: int,
    json_file: str | None = None,
):
    """
    Update metadata table with load information.

    This stores the source filename, thing count, and load timestamp.

    Args:
        conn: Open SQLite connection.
        source_file: Path to the original Twos export file, if known.
        thing_count: Number of things loaded.
        json_file: Path to the JSON file used for loading, if known.

    Returns:
        None.
    """
    cursor = conn.cursor()

    if source_file:
        cursor.execute(
            """
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES ('source_file', ?)
        """,
            (source_file,),
        )

    if json_file:
        cursor.execute(
            """
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES ('json_file', ?)
        """,
            (json_file,),
        )

    cursor.execute(
        """
        INSERT OR REPLACE INTO metadata (key, value)
        VALUES ('thing_count', ?)
    """,
        (str(thing_count),),
    )

    cursor.execute(
        """
        INSERT OR REPLACE INTO metadata (key, value)
        VALUES ('last_loaded', ?)
    """,
        (datetime.now().isoformat(),),
    )

    conn.commit()
    print("Metadata updated")


def generate_embeddings(
    conn: sqlite3.Connection,
    tasks: list,
    batch_size: int = 64,
    show_progress: bool = True,
) -> None:
    """
    Generate and store embeddings for all things.

    Args:
        conn: Open SQLite connection
        tasks: List of thing dictionaries
        batch_size: Batch size for encoding
        show_progress: Show progress bar

    Returns:
        None
    """
    print(f"\n📊 Generating embeddings for {len(tasks)} things...")

    # Check if user wants embeddings (environment variable)
    if os.getenv("MEMEX_DISABLE_EMBEDDINGS") == "1":
        print("⚠️  Embeddings disabled via MEMEX_DISABLE_EMBEDDINGS=1")
        return

    try:
        import sys

        # Add src to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from memex_twos_mcp.embeddings import EmbeddingGenerator
        import numpy as np

        # Also need sqlite_vec for vec_index
        try:
            import sqlite_vec
        except ImportError:
            print("⚠️  sqlite-vec not installed, skipping embeddings")
            print("   Install with: pip install sqlite-vec")
            return

        embedding_gen = EmbeddingGenerator()
        if not embedding_gen.available:
            print("⚠️  Embedding model unavailable, skipping")
            print("   Install with: pip install sentence-transformers")
            return
    except Exception as e:
        print(f"⚠️  Could not initialize embeddings: {e}")
        print(
            "   Skipping embedding generation. Set MEMEX_DISABLE_EMBEDDINGS=1 to silence."
        )
        return

    # Initialize sqlite-vec extension
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        # Create vec_index virtual table
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0(
                thing_id TEXT PRIMARY KEY,
                embedding float[384]
            )
        """
        )
        conn.commit()
    except Exception as e:
        print(f"⚠️  Could not initialize vector search: {e}")
        return

    # Prepare texts (use content field)
    texts = [task.get("content", "") for task in tasks]
    thing_ids = [task["id"] for task in tasks]

    # Generate embeddings in batches
    try:
        embeddings = embedding_gen.encode_batch(
            texts, batch_size=batch_size, show_progress=show_progress
        )
    except Exception as e:
        print(f"⚠️  Embedding generation failed: {e}")
        return

    # Store in database
    cursor = conn.cursor()
    print("  Storing embeddings...")
    for thing_id, embedding in zip(thing_ids, embeddings):
        # Serialize as float32 bytes for thing_embeddings table
        embedding_blob = embedding.astype(np.float32).tobytes()

        cursor.execute(
            """
            INSERT INTO thing_embeddings (thing_id, embedding, model_version)
            VALUES (?, ?, ?)
        """,
            (thing_id, embedding_blob, embedding_gen.model_name),
        )

        # Also insert into vec_index for fast search
        # vec_index expects the raw bytes directly
        cursor.execute(
            """
            INSERT INTO vec_index (thing_id, embedding)
            VALUES (?, ?)
        """,
            (thing_id, embedding_blob),
        )

    conn.commit()
    print(f"✅ Generated and stored {len(embeddings)} embeddings")


def _insert_thing(cursor, task):
    """Insert single thing with all fields including content_hash and list metadata."""
    content_hash = compute_content_hash(task)

    cursor.execute(
        """
        INSERT INTO things (
            id, timestamp, content, content_hash,
            timestamp_raw, content_raw, section_header, section_date,
            line_number, indent_level, parent_task_id, bullet_type,
            is_completed, is_pending, is_strikethrough,
            item_type, list_id, is_substantive
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            task["id"],
            task.get("timestamp"),
            task.get("content"),
            content_hash,
            task.get("timestamp_raw"),
            task.get("content_raw"),
            task.get("section_header"),
            task.get("section_date"),
            task.get("line_number"),
            task.get("indent_level", 0),
            task.get("parent_task_id"),
            task.get("bullet_type"),
            task.get("is_completed", False),
            task.get("is_pending", False),
            task.get("is_strikethrough", False),
            task.get("item_type", "content"),
            task.get("list_id"),
            task.get("is_substantive", True),
        ),
    )


def _update_people(conn, tasks: List[Dict]):
    """Update people for a subset of tasks (used by incremental load)."""
    cursor = conn.cursor()

    # Clear existing relationships for these tasks
    task_ids = [t["id"] for t in tasks]
    if task_ids:
        placeholders = ",".join("?" * len(task_ids))
        cursor.execute(
            f"DELETE FROM thing_people WHERE thing_id IN ({placeholders})", task_ids
        )

    # Reinsert
    for task in tasks:
        for person in task.get("people_mentioned", []):
            # Insert person if not exists
            cursor.execute(
                "INSERT OR IGNORE INTO people (name, normalized_name) VALUES (?, ?)",
                (person, person.lower()),
            )
            # Get person ID
            cursor.execute("SELECT id FROM people WHERE name = ?", (person,))
            person_id = cursor.fetchone()[0]
            # Link to thing
            cursor.execute(
                "INSERT INTO thing_people (thing_id, person_id) VALUES (?, ?)",
                (task["id"], person_id),
            )

    conn.commit()


def _update_tags(conn, tasks: List[Dict]):
    """Update tags for a subset of tasks (used by incremental load)."""
    cursor = conn.cursor()

    # Clear existing relationships for these tasks
    task_ids = [t["id"] for t in tasks]
    if task_ids:
        placeholders = ",".join("?" * len(task_ids))
        cursor.execute(
            f"DELETE FROM thing_tags WHERE thing_id IN ({placeholders})", task_ids
        )

    # Reinsert
    for task in tasks:
        for tag in task.get("tags", []):
            # Insert tag if not exists
            cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
            # Get tag ID
            cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
            tag_id = cursor.fetchone()[0]
            # Link to thing
            cursor.execute(
                "INSERT INTO thing_tags (thing_id, tag_id) VALUES (?, ?)",
                (task["id"], tag_id),
            )

    conn.commit()


def _update_links(conn, tasks: List[Dict]):
    """Update links for a subset of tasks (used by incremental load)."""
    cursor = conn.cursor()

    # Clear existing links for these tasks
    task_ids = [t["id"] for t in tasks]
    if task_ids:
        placeholders = ",".join("?" * len(task_ids))
        cursor.execute(
            f"DELETE FROM links WHERE thing_id IN ({placeholders})", task_ids
        )

    # Reinsert
    for task in tasks:
        for link in task.get("links", []):
            cursor.execute(
                "INSERT INTO links (thing_id, link_text, url) VALUES (?, ?, ?)",
                (task["id"], link.get("text"), link.get("url")),
            )

    conn.commit()


def _update_embeddings_incremental(conn, tasks: List[Dict]):
    """
    Regenerate embeddings only for changed things (Phase 4 integration).

    Only runs if Phase 4 (embeddings) is implemented and available.
    """
    # Check if embeddings are available
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from memex_twos_mcp.embeddings import EmbeddingGenerator
        import numpy as np
    except ImportError:
        return  # Phase 4 not implemented

    # Check if disabled via environment
    if os.getenv("MEMEX_DISABLE_EMBEDDINGS") == "1":
        return

    print(f"  Updating embeddings for {len(tasks)} changed things...")

    try:
        embedding_gen = EmbeddingGenerator()
        if not embedding_gen.available:
            print("    ⚠️  Embedding model unavailable, skipping")
            return
    except Exception as e:
        print(f"    ⚠️  Could not initialize embeddings: {e}")
        return

    # Generate embeddings
    texts = [task["content"] for task in tasks]
    thing_ids = [task["id"] for task in tasks]

    try:
        embeddings = embedding_gen.encode_batch(
            texts, show_progress=True, batch_size=64
        )
    except Exception as e:
        print(f"    ⚠️  Embedding generation failed: {e}")
        return

    cursor = conn.cursor()
    for thing_id, embedding in zip(thing_ids, embeddings):
        embedding_blob = embedding.astype(np.float32).tobytes()

        # Upsert (replace if exists)
        cursor.execute(
            """
            INSERT OR REPLACE INTO thing_embeddings (thing_id, embedding, model_version)
            VALUES (?, ?, ?)
        """,
            (thing_id, embedding_blob, embedding_gen.model_name),
        )

        # Also update vec_index (if it exists)
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO vec_index (thing_id, embedding)
                VALUES (?, ?)
            """,
                (thing_id, embedding_blob),
            )
        except sqlite3.OperationalError:
            # vec_index might not exist, that's ok
            pass

    conn.commit()
    print(f"    ✅ Updated {len(embeddings)} embeddings")


def incremental_load(
    conn: sqlite3.Connection,
    tasks: list,
    source_file: str,
    json_file: str,
    mode: str = "append",
) -> Dict[str, Any]:
    """
    Incrementally load tasks using upsert logic.

    Modes:
    - 'append': Insert new only, don't delete (safest, default)
    - 'sync': Update changed, insert new, delete removed (full sync)
    - 'rebuild': Delete all and rebuild (current behavior)

    Args:
        conn: Open SQLite connection
        tasks: List of thing dictionaries
        source_file: Path to original Twos export
        json_file: Path to JSON file
        mode: Import mode ('append', 'sync', 'rebuild')

    Returns:
        Stats dict: {new_count, updated_count, deleted_count, duration_seconds}
    """
    import time

    start_time = time.time()

    print(f"\n📊 Incremental load mode: {mode}")

    cursor = conn.cursor()

    # Step 1: Compute hashes for incoming tasks
    print("  Computing content hashes...")
    incoming = {}
    for task in tasks:
        content_hash = compute_content_hash(task)
        task["content_hash"] = content_hash
        incoming[task["id"]] = (task, content_hash)

    # Step 2: Fetch existing hashes
    print("  Fetching existing data...")
    cursor.execute("SELECT id, content_hash FROM things")
    existing = {row[0]: row[1] for row in cursor.fetchall()}

    # Step 3: Categorize changes
    new_ids = set(incoming.keys()) - set(existing.keys())
    deleted_ids = (
        set(existing.keys()) - set(incoming.keys()) if mode == "sync" else set()
    )

    updated_ids = set()
    for thing_id in set(incoming.keys()) & set(existing.keys()):
        if incoming[thing_id][1] != existing[thing_id]:
            updated_ids.add(thing_id)

    print(
        f"  Changes detected: {len(new_ids)} new, {len(updated_ids)} updated, {len(deleted_ids)} deleted"
    )

    # Step 4: Apply changes

    # Insert new things
    if new_ids:
        print(f"  Inserting {len(new_ids)} new things...")
        for thing_id in new_ids:
            task, _ = incoming[thing_id]
            _insert_thing(cursor, task)

    # Update changed things (delete + reinsert to trigger FTS update)
    if updated_ids:
        print(f"  Updating {len(updated_ids)} changed things...")
        for thing_id in updated_ids:
            task, _ = incoming[thing_id]
            # Delete old version (cascades to FTS via trigger)
            cursor.execute("DELETE FROM things WHERE id = ?", (thing_id,))
            # Insert new version
            _insert_thing(cursor, task)

    # Delete removed things (sync mode only)
    if mode == "sync" and deleted_ids:
        print(f"  Deleting {len(deleted_ids)} removed things...")
        for thing_id in deleted_ids:
            cursor.execute("DELETE FROM things WHERE id = ?", (thing_id,))

    conn.commit()

    # Step 5: Update related entities (people, tags, links)
    # Only for new/updated things
    changed_things = [incoming[tid][0] for tid in (new_ids | updated_ids)]
    if changed_things:
        print(f"  Updating entities for {len(changed_things)} changed things...")
        _update_people(conn, changed_things)
        _update_tags(conn, changed_things)
        _update_links(conn, changed_things)

    # Step 6: Update embeddings incrementally (Phase 4 integration)
    if changed_things:
        _update_embeddings_incremental(conn, changed_things)

    # Step 7: Record import run
    duration = time.time() - start_time
    cursor.execute(
        """
        INSERT INTO imports (
            source_file, json_file, mode, thing_count,
            new_count, updated_count, deleted_count, duration_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            source_file,
            json_file,
            mode,
            len(tasks),
            len(new_ids),
            len(updated_ids),
            len(deleted_ids),
            duration,
        ),
    )

    cursor.execute(
        """
        INSERT OR REPLACE INTO metadata (key, value)
        VALUES ('last_incremental_import', ?)
    """,
        (datetime.now().isoformat(),),
    )

    conn.commit()

    print(f"✅ Incremental import completed in {duration:.2f}s")

    return {
        "new_count": len(new_ids),
        "updated_count": len(updated_ids),
        "deleted_count": len(deleted_ids),
        "duration_seconds": duration,
    }


def validate_database(conn: sqlite3.Connection):
    """
    Run validation queries on the loaded database.

    This prints summary counts and sample aggregates for sanity checks.

    Args:
        conn: Open SQLite connection.

    Returns:
        None.
    """
    print("\n=== Database Validation ===")

    cursor = conn.cursor()

    # Thing count
    cursor.execute("SELECT COUNT(*) FROM things")
    thing_count = cursor.fetchone()[0]
    print(f"Things: {thing_count}")

    # People count
    cursor.execute("SELECT COUNT(*) FROM people")
    people_count = cursor.fetchone()[0]
    print(f"People: {people_count}")

    # Tag count
    cursor.execute("SELECT COUNT(*) FROM tags")
    tag_count = cursor.fetchone()[0]
    print(f"Tags: {tag_count}")

    # Link count
    cursor.execute("SELECT COUNT(*) FROM links")
    link_count = cursor.fetchone()[0]
    print(f"Links: {link_count}")

    # Top 5 people by mentions
    cursor.execute(
        """
        SELECT p.name, COUNT(*) as mentions
        FROM people p
        JOIN thing_people tp ON p.id = tp.person_id
        GROUP BY p.id
        ORDER BY mentions DESC
        LIMIT 5
    """
    )
    print("\nTop 5 People:")
    for name, count in cursor.fetchall():
        print(f"  - {name}: {count}")

    # Top 5 tags
    cursor.execute(
        """
        SELECT t.name, COUNT(*) as uses
        FROM tags t
        JOIN thing_tags tt ON t.id = tt.tag_id
        GROUP BY t.id
        ORDER BY uses DESC
        LIMIT 5
    """
    )
    print("\nTop 5 Tags:")
    for name, count in cursor.fetchall():
        print(f"  - {name}: {count}")

    # FTS check
    cursor.execute("SELECT COUNT(*) FROM things_fts")
    fts_count = cursor.fetchone()[0]
    print(f"\nFTS Index: {fts_count} entries")

    # Test FTS search for a common token.
    cursor.execute(
        """
        SELECT thing_id FROM things_fts WHERE things_fts MATCH 'alice' LIMIT 5
    """
    )
    fts_results = cursor.fetchall()
    print(f"FTS Test ('alice'): {len(fts_results)} results")


def main():
    """
    CLI entry point for loading JSON data into SQLite.

    Returns:
        Exit code integer (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(description="Load Twos JSON data into SQLite")
    parser.add_argument("json_file", type=Path, help="Path to JSON data file")
    parser.add_argument("-o", "--output", type=Path, help="Output database file")
    parser.add_argument("-s", "--schema", type=Path, help="Path to schema file")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing database"
    )
    parser.add_argument(
        "--mode",
        choices=["rebuild", "sync", "append"],
        default="rebuild",
        help=(
            "Import mode: "
            "'rebuild' (delete all, full reload - default), "
            "'sync' (update changed, delete removed), "
            "'append' (insert new only, safest for incremental)"
        ),
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Enable incremental mode (shorthand for --mode=append)",
    )

    args = parser.parse_args()

    # Defaults
    if not args.output:
        args.output = Path(__file__).parent.parent / "data" / "processed" / "twos.db"

    if not args.schema:
        args.schema = Path(__file__).parent.parent / "schema" / "schema.sql"

    # Handle --incremental shorthand
    if args.incremental:
        args.mode = "append"

    # Check if database exists
    db_exists = args.output.exists()

    # Validate mode and database state
    if args.mode in ("sync", "append") and not db_exists:
        print(
            f"⚠️  Incremental mode '{args.mode}' requires existing database, switching to 'rebuild'"
        )
        args.mode = "rebuild"

    if db_exists and args.mode == "rebuild" and not args.force:
        print(
            f"Error: Database {args.output} already exists. Use --force to overwrite "
            "or use --mode=sync/append for incremental update."
        )
        return 1

    if not args.schema.exists():
        print(f"Error: Schema file not found: {args.schema}")
        print("Tip: Use -s to provide the schema path.")
        return 1

    if not args.json_file.exists():
        print(f"Error: JSON file not found: {args.json_file}")
        print("Tip: Run the converter first to generate JSON output.")
        return 1

    # I/O boundary: read JSON from disk.
    print(f"Loading JSON from {args.json_file}...")
    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both raw list and {metadata, tasks} structure.
    metadata = {}
    if isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
        metadata = (
            data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
        )
    else:
        tasks = data

    print(f"Found {len(tasks)} things")

    # Check if list metadata is present
    has_list_metadata = False
    if tasks and isinstance(tasks[0], dict):
        has_list_metadata = "item_type" in tasks[0] or "list_id" in tasks[0]

    if not has_list_metadata:
        print("⚠️  WARNING: List metadata not found in JSON!")
        print("   Things will be loaded without item_type and list_id.")
        print(
            "   💡 Tip: Run scripts/build_list_metadata.py first to add list semantics."
        )
        print()

    # Load separate lists metadata file if it exists
    lists_metadata = []
    lists_file = args.json_file.parent / "lists_metadata.json"
    if lists_file.exists():
        print(f"Loading lists metadata from {lists_file}...")
        with open(lists_file, "r", encoding="utf-8") as f:
            lists_metadata = json.load(f)
        print(f"  Loaded {len(lists_metadata)} lists")
    elif has_list_metadata:
        # Extract from enhanced tasks metadata
        if isinstance(data, dict) and "lists_metadata" in data:
            lists_metadata = data["lists_metadata"]
            print(f"  Found {len(lists_metadata)} lists in JSON metadata")

    # Choose loading strategy based on mode
    if args.mode == "rebuild" or not db_exists:
        # Full rebuild (delete and recreate)
        if db_exists:
            args.output.unlink()

        conn = create_database(args.output, args.schema)

        try:
            # Load data and build normalized tables.
            load_tasks(conn, tasks)
            load_people(conn, tasks)
            load_tags(conn, tasks)
            load_links(conn, tasks)

            # Load list metadata and relationships
            if lists_metadata:
                load_lists(conn, lists_metadata)
                load_thing_lists(conn, tasks)
            elif has_list_metadata:
                # List metadata embedded in tasks but no separate lists file
                # Generate lists metadata from tasks
                print("ℹ️  Generating lists metadata from embedded task data...")
                from collections import defaultdict

                lists_by_id = defaultdict(
                    lambda: {
                        "items": [],
                        "item_count": 0,
                        "substantive_count": 0,
                    }
                )

                for task in tasks:
                    list_id = task.get("list_id")
                    if list_id:
                        lists_by_id[list_id]["items"].append(task)
                        lists_by_id[list_id]["item_count"] += 1
                        if task.get("is_substantive", True):
                            lists_by_id[list_id]["substantive_count"] += 1

                # Create lists entries (minimal metadata)
                generated_lists = []
                for list_id, info in lists_by_id.items():
                    items = info["items"]
                    # Extract metadata from first item
                    first_item = items[0]
                    section_header = first_item.get("section_header", "Unknown")

                    generated_lists.append(
                        {
                            "list_id": list_id,
                            "list_type": "unknown",  # Can't infer without build_list_metadata
                            "list_name": list_id,
                            "list_name_raw": section_header,
                            "list_date": None,
                            "start_line": min(t.get("line_number", 0) for t in items),
                            "end_line": max(t.get("line_number", 0) for t in items),
                            "item_count": info["item_count"],
                            "substantive_count": info["substantive_count"],
                            "created_at": datetime.now().isoformat(),
                        }
                    )

                load_lists(conn, generated_lists)
                load_thing_lists(conn, tasks)

            # Generate embeddings for semantic search
            generate_embeddings(conn, tasks)

            update_metadata(
                conn,
                metadata.get("source_file"),
                len(tasks),
                json_file=str(args.json_file),
            )

            # Validate with summary queries.
            validate_database(conn)

            print(f"\n✅ Successfully loaded data into {args.output}")

        except Exception as e:
            # Re-raise after reporting to keep the traceback for debugging.
            print(f"\n❌ Error during loading: {e}")
            raise
        finally:
            conn.close()
    else:
        # Incremental mode (sync or append)
        conn = sqlite3.connect(args.output)

        # Enable foreign key constraints (required for proper CASCADE behavior)
        # Critical in sync mode: when things are deleted, CASCADE ensures
        # junction table rows (thing_tags, thing_people, etc) are also deleted.
        # Without this, orphaned rows accumulate silently.
        conn.execute("PRAGMA foreign_keys = ON")

        try:
            stats = incremental_load(
                conn,
                tasks,
                source_file=metadata.get("source_file", ""),
                json_file=str(args.json_file),
                mode=args.mode,
            )

            # Update metadata
            update_metadata(
                conn,
                metadata.get("source_file"),
                len(tasks),
                json_file=str(args.json_file),
            )

            print("\n📊 Import Statistics:")
            print(f"  New: {stats['new_count']}")
            print(f"  Updated: {stats['updated_count']}")
            print(f"  Deleted: {stats['deleted_count']}")
            print(f"  Duration: {stats['duration_seconds']:.2f}s")

            print(f"\n✅ Successfully updated database at {args.output}")

        except Exception as e:
            print(f"\n❌ Error during incremental load: {e}")
            raise
        finally:
            conn.close()

    return 0


if __name__ == "__main__":
    exit(main())
