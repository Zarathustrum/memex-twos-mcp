#!/usr/bin/env python3
"""
Migrate existing Twos database to add list semantics (Phase 6).

This script adds list metadata to an existing database by:
1. Adding new columns to things table (item_type, list_id, is_substantive)
2. Creating lists and thing_lists tables
3. Deriving list metadata from existing section_header + line_number data
4. Backfilling item_type, list_id, is_substantive for all things

Safe to run on existing databases (idempotent).

Usage:
    python3 scripts/migrate_add_list_semantics.py data/processed/twos.db
"""

import sqlite3
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from dateutil import parser as dateparser


def slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def parse_date_header(text: str) -> Optional[datetime]:
    """Parse date from section header."""
    if not text or len(text) < 3:
        return None
    try:
        return dateparser.parse(text, fuzzy=True)
    except (ValueError, TypeError):
        return None


def classify_section_header(header: str) -> Tuple[str, str, Optional[str]]:
    """Classify section header as date/topic/metadata."""
    header_clean = " ".join(header.split()).strip()

    if not header_clean:
        return ("metadata", "unknown", None)

    # Dividers
    if re.match(r"^[\-=_\*•]{3,}$", header_clean):
        return ("metadata", "divider", None)

    # Try parsing as date
    date_parsed = parse_date_header(header_clean)
    if date_parsed:
        list_date = date_parsed.date().isoformat()
        return ("date", list_date, list_date)

    # Topic list
    return ("topic", header_clean, None)


def is_divider(content: str) -> bool:
    """Check if content is a divider line."""
    if not content or len(content) < 3:
        return False
    return bool(re.match(r"^[\-=_\*•]{3,}$", content.strip()))


def classify_item_type(content: str) -> str:
    """Classify a thing as content/divider/header/metadata."""
    if not content:
        return "metadata"

    content = content.strip()

    # Dividers
    if is_divider(content):
        return "divider"

    # Headers
    if len(content) < 50 and (content.isupper() or content.endswith(":")):
        return "header"

    # Metadata
    if content.startswith("[") and content.endswith("]"):
        return "metadata"

    return "content"


def generate_list_id(list_type: str, list_name: str, start_line: int) -> str:
    """Generate deterministic list_id."""
    if list_type == "date":
        return f"date_{list_name}"

    slug = slugify(list_name)
    return f"{list_type}_{slug}_{start_line}"


def check_migration_needed(conn: sqlite3.Connection) -> Tuple[bool, str]:
    """
    Check if migration is needed.

    Returns:
        (needs_migration, reason)
    """
    cursor = conn.cursor()

    # Check if lists table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lists'")
    if not cursor.fetchone():
        return (True, "lists table missing")

    # Check if things table has new columns
    cursor.execute("PRAGMA table_info(things)")
    columns = {row[1] for row in cursor.fetchall()}

    if "item_type" not in columns:
        return (True, "item_type column missing from things table")

    if "list_id" not in columns:
        return (True, "list_id column missing from things table")

    if "is_substantive" not in columns:
        return (True, "is_substantive column missing from things table")

    # Check if thing_lists table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='thing_lists'"
    )
    if not cursor.fetchone():
        return (True, "thing_lists table missing")

    return (False, "already migrated")


def add_new_columns(conn: sqlite3.Connection):
    """Add new columns to things table."""
    print("  Adding new columns to things table...")

    cursor = conn.cursor()

    # Check which columns need to be added
    cursor.execute("PRAGMA table_info(things)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    if "item_type" not in existing_columns:
        cursor.execute("ALTER TABLE things ADD COLUMN item_type TEXT DEFAULT 'content'")
        print("    Added item_type column")

    if "list_id" not in existing_columns:
        cursor.execute("ALTER TABLE things ADD COLUMN list_id TEXT")
        print("    Added list_id column")

    if "is_substantive" not in existing_columns:
        cursor.execute("ALTER TABLE things ADD COLUMN is_substantive BOOLEAN DEFAULT 1")
        print("    Added is_substantive column")

    conn.commit()


def create_new_tables(conn: sqlite3.Connection):
    """Create lists and thing_lists tables."""
    print("  Creating lists and thing_lists tables...")

    cursor = conn.cursor()

    # Create lists table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS lists (
            list_id TEXT PRIMARY KEY,
            list_type TEXT NOT NULL,
            list_name TEXT NOT NULL,
            list_name_raw TEXT NOT NULL,
            list_date TEXT,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            item_count INTEGER DEFAULT 0,
            substantive_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(list_name, start_line)
        )
    """
    )

    # Create thing_lists table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS thing_lists (
            thing_id TEXT NOT NULL,
            list_id TEXT NOT NULL,
            position_in_list INTEGER,
            PRIMARY KEY (thing_id, list_id),
            FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE,
            FOREIGN KEY (list_id) REFERENCES lists(list_id) ON DELETE CASCADE
        )
    """
    )

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lists_type ON lists(list_type)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_lists_date ON lists(list_date) WHERE list_date IS NOT NULL"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lists_name ON lists(list_name)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_lists_lines ON lists(start_line, end_line)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_thing_lists_list ON thing_lists(list_id, position_in_list)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_things_list_id ON things(list_id)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_things_item_type ON things(item_type)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_things_substantive ON things(is_substantive) WHERE is_substantive = 1"
    )

    conn.commit()
    print("    Tables and indexes created")


def derive_and_backfill_metadata(conn: sqlite3.Connection):
    """
    Derive list metadata from existing things and backfill.

    Algorithm:
    1. Fetch all things ordered by line_number
    2. Group by section_header
    3. For each section:
       a. Classify header (date vs topic)
       b. Determine boundaries
       c. Classify each item's type
       d. Generate list_id
       e. Update things with item_type, list_id, is_substantive
       f. Insert list metadata
       g. Insert thing_lists relationships
    """
    print("  Deriving list metadata from existing things...")

    cursor = conn.cursor()

    # Fetch all things
    cursor.execute(
        """
        SELECT id, content, section_header, line_number
        FROM things
        ORDER BY line_number
    """
    )
    things = cursor.fetchall()
    print(f"    Processing {len(things)} things...")

    # Group by section_header
    from collections import defaultdict

    sections = defaultdict(list)

    for row in things:
        thing_id, content, section_header, line_number = row
        sections[section_header or "Unknown"].append(
            {
                "id": thing_id,
                "content": content,
                "section_header": section_header,
                "line_number": line_number,
            }
        )

    print(f"    Found {len(sections)} unique sections")

    # Process each section
    lists_created = 0
    things_updated = 0
    relationships_created = 0

    for header, items in sections.items():
        if not items:
            continue

        # Classify section
        list_type, list_name, list_date = classify_section_header(header)

        # Boundaries
        start_line = min(t["line_number"] for t in items if t["line_number"])
        end_line = max(t["line_number"] for t in items if t["line_number"])

        # Generate list_id
        list_id = generate_list_id(list_type, list_name, start_line)

        # Classify and update each item
        substantive_count = 0
        for idx, item in enumerate(
            sorted(items, key=lambda t: t.get("line_number", 0))
        ):
            # Classify item type
            item_type = classify_item_type(item["content"])
            is_substantive = item_type == "content" and len(item["content"]) > 3

            if is_substantive:
                substantive_count += 1

            # Update thing
            cursor.execute(
                """
                UPDATE things
                SET item_type = ?, list_id = ?, is_substantive = ?
                WHERE id = ?
            """,
                (item_type, list_id, is_substantive, item["id"]),
            )
            things_updated += 1

            # Insert thing_lists relationship
            cursor.execute(
                """
                INSERT OR IGNORE INTO thing_lists (thing_id, list_id, position_in_list)
                VALUES (?, ?, ?)
            """,
                (item["id"], list_id, idx),
            )
            relationships_created += 1

        # Insert list metadata
        cursor.execute(
            """
            INSERT OR REPLACE INTO lists (
                list_id, list_type, list_name, list_name_raw, list_date,
                start_line, end_line, item_count, substantive_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                list_id,
                list_type,
                list_name,
                header,
                list_date,
                start_line,
                end_line,
                len(items),
                substantive_count,
                datetime.now().isoformat(),
            ),
        )
        lists_created += 1

    conn.commit()

    print(f"    Created {lists_created} lists")
    print(f"    Updated {things_updated} things")
    print(f"    Created {relationships_created} thing-list relationships")

    # Print summary
    cursor.execute("SELECT list_type, COUNT(*) FROM lists GROUP BY list_type")
    print("\n    List breakdown:")
    for list_type, count in cursor.fetchall():
        print(f"      - {list_type}: {count}")


def update_schema_version(conn: sqlite3.Connection):
    """Update schema version in metadata table."""
    print("  Updating schema version...")

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO metadata (key, value)
        VALUES ('schema_version', '1.1')
    """
    )

    cursor.execute(
        """
        INSERT OR REPLACE INTO metadata (key, value)
        VALUES ('list_semantics_migration', ?)
    """,
        (datetime.now().isoformat(),),
    )

    conn.commit()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate existing Twos database to add list semantics"
    )
    parser.add_argument("database", type=Path, help="Path to SQLite database file")
    parser.add_argument(
        "--force", action="store_true", help="Force migration even if already migrated"
    )

    args = parser.parse_args()

    if not args.database.exists():
        print(f"❌ Error: Database not found: {args.database}")
        return 1

    print(f"Migrating database: {args.database}")

    # Open database
    conn = sqlite3.connect(args.database)

    try:
        # Check if migration needed
        needs_migration, reason = check_migration_needed(conn)

        if not needs_migration and not args.force:
            print(f"✅ Database already migrated ({reason})")
            return 0

        if args.force:
            print(f"⚠️  Forcing migration (was: {reason})")

        print("\n📊 Starting migration...\n")

        # Step 1: Add new columns
        add_new_columns(conn)

        # Step 2: Create new tables
        create_new_tables(conn)

        # Step 3: Derive and backfill metadata
        derive_and_backfill_metadata(conn)

        # Step 4: Update schema version
        update_schema_version(conn)

        print("\n✅ Migration complete!")
        print(f"   Database: {args.database}")
        print("   Schema version: 1.1 (list semantics enabled)")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    exit(main())
