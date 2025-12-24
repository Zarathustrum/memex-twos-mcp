#!/usr/bin/env python3
"""
Load converted Twos JSON data into SQLite database.

This script:
- Creates the database schema
- Loads all things
- Populates normalized entity tables (people, tags, links)
- Creates relationships via junction tables
- Populates FTS index
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
import argparse


def create_database(db_path: Path, schema_path: Path):
    """
    Create database and apply schema.

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
    for index, task in enumerate(tasks, start=1):
        # Parameterized INSERT protects against SQL injection and quoting issues.
        cursor.execute(
            """
            INSERT INTO things (
                id, timestamp, timestamp_raw, content, content_raw,
                section_header, section_date, line_number, indent_level,
                parent_task_id, bullet_type, is_completed, is_pending,
                is_strikethrough
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task["id"],
                task.get("timestamp"),
                task.get("timestamp_raw"),
                task.get("content"),
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
            ),
        )

        conn.commit()
        if index % 500 == 0:
            print(f"  - Inserted {index} things")

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

    args = parser.parse_args()

    # Defaults
    if not args.output:
        args.output = Path(__file__).parent.parent / "data" / "processed" / "twos.db"

    if not args.schema:
        args.schema = Path(__file__).parent.parent / "schema" / "schema.sql"

    # Check if database exists to avoid overwriting without --force.
    if args.output.exists():
        if not args.force:
            print(
                f"Error: Database {args.output} already exists. Use --force to overwrite."
            )
            return 1
        # Force means delete and recreate to avoid duplicate inserts.
        args.output.unlink()

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
        metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    else:
        tasks = data

    print(f"Found {len(tasks)} things")

    # Create database and schema before inserting data.
    conn = create_database(args.output, args.schema)

    try:
        # Load data and build normalized tables.
        load_tasks(conn, tasks)
        load_people(conn, tasks)
        load_tags(conn, tasks)
        load_links(conn, tasks)
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

    return 0


if __name__ == "__main__":
    exit(main())
