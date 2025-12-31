#!/usr/bin/env python3
"""
ThreadPacks Builder

Builds tag and person thread indices for fast "what's active?" queries.

I/O Boundary: Reads from SQLite database, writes thread indices back to database.

Purpose:
- Precompute activity indices for tags and people (deterministic, no embeddings)
- Answers "What's active with Alice?" or "Show me recent #work threads"
- Token optimization: compact thread pack vs full query results
- Activity detection: 90-day active window (configurable)

Thread types:
- Single-tag threads: thr:tag:work, thr:tag:personal, etc.
- Single-person threads: thr:person:alice, thr:person:bob, etc.
- Multi-tag intersection computed at query time (not precomputed)

Incremental rebuild:
- Uses src_hash (SHA256 of thing IDs + content hashes)
- Skips rebuild if source data unchanged for this thread
- Force flag available to override

TH1 Pack Format:
TH1|id=<thread_id>|st=<active|stale>|last=<last_ts>|n=<total>|a90=<count_90d>|kw=<word,word,...>|hi=<thing_id~label;...>
"""

import argparse
import hashlib
import math
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse functions from build_timepacks
sys.path.insert(0, str(Path(__file__).parent))
try:
    from build_timepacks import make_label
except ImportError:
    # Fallback implementation
    def make_label(content: str, max_len: int = 32) -> str:
        words = re.findall(r"\b[a-zA-Z0-9]+\b", content)[:4]
        label = "_".join(words).lower()
        if len(label) > max_len:
            label = label[:max_len]
        return label if label else "item"


# Common English stopwords for keyword extraction
STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "might",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "s",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def compute_src_hash(things: List[Dict[str, Any]]) -> str:
    """
    Compute deterministic hash of (thing_id, content_hash) tuples.

    Args:
        things: List of thing dictionaries

    Returns:
        SHA256 hex digest
    """
    # Sort by thing_id for deterministic ordering
    sorted_things = sorted(things, key=lambda x: x["id"])

    # Build string from (id, content_hash) pairs
    hash_input = "|".join(
        f"{t['id']}:{t.get('content_hash', '')}" for t in sorted_things
    )

    return hashlib.sha256(hash_input.encode()).hexdigest()


def score_thread_highlight(
    thing: Dict[str, Any], now: datetime, active_days: int = 90
) -> float:
    """
    Deterministic highlight scoring for thread highlights.

    score = (
        recency_score * 0.50 +
        entity_density_score * 0.30 +
        content_length_score * 0.20
    )

    Args:
        thing: Thing dictionary
        now: Current datetime
        active_days: Active window in days

    Returns:
        Composite score (0-1)
    """
    # Parse timestamp
    ts = thing.get("timestamp", "")
    try:
        thing_date = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        thing_date = now

    # Recency score: exponential decay from now
    days_ago = (now - thing_date).days
    recency_score = math.exp(-days_ago / active_days)

    # Entity density: (tags + people) / 10, capped at 1.0
    tag_count = len(thing.get("tags", []))
    people_count = len(thing.get("people_mentioned", []))
    entity_density_score = min((tag_count + people_count) / 10.0, 1.0)

    # Content length: prefer substantive items
    content_len = len(thing.get("content", ""))
    content_length_score = min(content_len / 200.0, 1.0)

    # Weighted combination
    score = (
        recency_score * 0.50 + entity_density_score * 0.30 + content_length_score * 0.20
    )

    return score


def extract_keywords(things: List[Dict[str, Any]], top_n: int = 20) -> List[str]:
    """
    Deterministic keyword extraction using TF.

    Args:
        things: List of thing dictionaries
        top_n: Number of keywords to return

    Returns:
        List of keywords sorted alphabetically
    """
    token_counts = Counter()

    # Tokenize content
    for thing in things:
        content = thing.get("content", "").lower()
        # Extract words (alphanumeric sequences)
        tokens = re.findall(r"\b[a-z]+\b", content)

        for token in tokens:
            # Skip stopwords, single chars, numbers
            if token not in STOPWORDS and len(token) > 1:
                token_counts[token] += 1

    # Get top N by frequency, then sort alphabetically
    top_tokens = [word for word, _ in token_counts.most_common(top_n)]
    top_tokens.sort()

    return top_tokens


def build_th1_pack(
    thread_id: str,
    status: str,
    last_ts: str,
    thing_count: int,
    thing_count_90d: int,
    keywords: List[str],
    highlights: List[Dict[str, Any]],
) -> str:
    """
    Build TH1 format pack string.

    TH1|id=<thread_id>|st=<active|stale>|last=<last_ts>|n=<total>|a90=<count_90d>|kw=<word,word,...>|hi=<thing_id~label;...>

    Args:
        thread_id: Thread identifier
        status: 'active' or 'stale'
        last_ts: Last thing timestamp
        thing_count: Total things
        thing_count_90d: Things in last 90 days
        keywords: Top keywords
        highlights: Top highlights with labels

    Returns:
        TH1 pack string
    """
    # Format last_ts (date only)
    try:
        last_date = datetime.fromisoformat(last_ts).date().isoformat()
    except (ValueError, TypeError):
        last_date = last_ts[:10] if last_ts else "unknown"

    # Keywords
    kw_str = ",".join(keywords) if keywords else ""

    # Highlights
    hi_parts = []
    for hl in highlights[:10]:
        thing_id = hl["id"]
        label = hl.get("label", make_label(hl.get("content", "")))
        hi_parts.append(f"{thing_id}~{label}")

    hi_str = ";".join(hi_parts) if hi_parts else ""

    # Build pack
    pack = (
        f"TH1|id={thread_id}|st={status}|last={last_date}|n={thing_count}|"
        f"a90={thing_count_90d}|kw={kw_str}|hi={hi_str}"
    )

    # Ensure bounded length (~600 chars max)
    if len(pack) > 600:
        # Truncate highlights if needed
        while hi_parts and len(pack) > 600:
            hi_parts.pop()
            hi_str = ";".join(hi_parts)
            pack = (
                f"TH1|id={thread_id}|st={status}|last={last_date}|n={thing_count}|"
                f"a90={thing_count_90d}|kw={kw_str}|hi={hi_str}"
            )

    return pack


def fetch_things_for_thread(
    conn: sqlite3.Connection, kind: str, label: str
) -> List[Dict[str, Any]]:
    """
    Fetch all things associated with a tag or person thread.

    Args:
        conn: Database connection
        kind: 'tag' or 'person'
        label: Tag name or person name

    Returns:
        List of thing dictionaries with tags and people
    """
    cursor = conn.cursor()
    cursor.row_factory = sqlite3.Row

    if kind == "tag":
        # Fetch things with this tag (tags.name is already lowercase)
        cursor.execute(
            """
            SELECT t.* FROM things t
            JOIN thing_tags tt ON t.id = tt.thing_id
            JOIN tags tg ON tt.tag_id = tg.id
            WHERE tg.name = ?
            ORDER BY t.timestamp DESC
            """,
            (label,),
        )
    else:  # person
        # Fetch things with this person
        # IMPORTANT: Use normalized_name (lowercase), not name (original case).
        # build_thread() passes label_norm (lowercase) as the label parameter.
        # people.name stores original case ("Alice"), but normalized_name is lowercase ("alice").
        # Querying by p.name would return zero results due to case mismatch.
        cursor.execute(
            """
            SELECT t.* FROM things t
            JOIN thing_people tp ON t.id = tp.thing_id
            JOIN people p ON tp.person_id = p.id
            WHERE p.normalized_name = ?
            ORDER BY t.timestamp DESC
            """,
            (label,),
        )

    things = []
    for row in cursor.fetchall():
        thing = dict(row)
        thing_id = thing["id"]

        # Fetch tags
        cursor.execute(
            """
            SELECT t.name FROM tags t
            JOIN thing_tags tt ON t.id = tt.tag_id
            WHERE tt.thing_id = ?
            """,
            (thing_id,),
        )
        thing["tags"] = [r[0] for r in cursor.fetchall()]

        # Fetch people
        cursor.execute(
            """
            SELECT p.name FROM people p
            JOIN thing_people tp ON p.id = tp.person_id
            WHERE tp.thing_id = ?
            """,
            (thing_id,),
        )
        thing["people_mentioned"] = [r[0] for r in cursor.fetchall()]

        things.append(thing)

    return things


def build_thread(
    conn: sqlite3.Connection,
    kind: str,
    label: str,
    label_norm: str,
    active_days: int = 90,
    force: bool = False,
    builder_v: str = "1.0",
) -> Tuple[bool, Optional[str]]:
    """
    Build a single thread index.

    Args:
        conn: Database connection
        kind: 'tag' or 'person'
        label: Display label
        label_norm: Normalized label (lowercase)
        active_days: Active window in days
        force: Force rebuild even if src_hash matches
        builder_v: Builder version

    Returns:
        (was_built, error_message)
    """
    cursor = conn.cursor()

    thread_id = f"thr:{kind}:{label_norm}"

    # Fetch things for this thread
    things = fetch_things_for_thread(conn, kind, label_norm)

    if not things:
        # No things in thread - skip or archive
        return False, None

    # Compute src_hash
    src_hash = compute_src_hash(things)

    # Check existing thread
    cursor.execute("SELECT src_hash FROM threads WHERE thread_id = ?", (thread_id,))
    row = cursor.fetchone()

    if row and row[0] == src_hash and not force:
        # Hash matches, skip rebuild
        return False, None

    # Compute metrics
    now = datetime.now()
    cutoff = now - timedelta(days=active_days)
    cutoff_str = cutoff.isoformat()

    recent_things = [t for t in things if t.get("timestamp", "") >= cutoff_str]

    thing_count = len(things)
    thing_count_90d = len(recent_things)

    # Timestamps
    timestamps = [t.get("timestamp", "") for t in things if t.get("timestamp")]
    start_ts = min(timestamps) if timestamps else None
    last_ts = max(timestamps) if timestamps else None

    # Status
    if thing_count_90d > 0:
        status = "active"
    else:
        status = "stale"

    # Score highlights from recent things
    if recent_things:
        scored = []
        for thing in recent_things:
            score = score_thread_highlight(thing, now, active_days)
            scored.append((score, thing))

        scored.sort(reverse=True, key=lambda x: x[0])

        # Top 10 = highlights, next 20 = evidence
        highlights = []
        for _, thing in scored[:10]:
            thing["label"] = make_label(thing.get("content", ""))
            highlights.append(thing)

        evidence = [t for _, t in scored[10:30]]
    else:
        highlights = []
        evidence = []

    # Extract keywords from recent things
    keywords = extract_keywords(recent_things, top_n=20) if recent_things else []
    kw_text = " ".join(keywords)

    # Build TH1 pack
    pack = build_th1_pack(
        thread_id,
        status,
        last_ts or "",
        thing_count,
        thing_count_90d,
        keywords,
        highlights,
    )

    # Delete old thread + evidence
    cursor.execute("DELETE FROM thread_evidence WHERE thread_id = ?", (thread_id,))
    cursor.execute("DELETE FROM threads_fts WHERE thread_id = ?", (thread_id,))
    cursor.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))

    # Insert thread
    cursor.execute(
        """
        INSERT INTO threads (
            thread_id, kind, label, label_norm,
            start_ts, last_ts, thing_count, thing_count_90d,
            status, archived_at,
            pack_v, pack, kw, src_hash, builder_v
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            thread_id,
            kind,
            label,
            label_norm,
            start_ts,
            last_ts,
            thing_count,
            thing_count_90d,
            status,
            None,
            1,
            pack,
            kw_text,
            src_hash,
            builder_v,
        ),
    )

    # Insert highlights
    for rank, thing in enumerate(highlights):
        cursor.execute(
            """
            INSERT INTO thread_evidence (thread_id, thing_id, role, rank)
            VALUES (?, ?, 'hi', ?)
            """,
            (thread_id, thing["id"], rank),
        )

    # Insert evidence
    for rank, thing in enumerate(evidence):
        cursor.execute(
            """
            INSERT INTO thread_evidence (thread_id, thing_id, role, rank)
            VALUES (?, ?, 'ev', ?)
            """,
            (thread_id, thing["id"], rank),
        )

    # Insert into FTS
    cursor.execute(
        """
        INSERT INTO threads_fts (thread_id, label, kw)
        VALUES (?, ?, ?)
        """,
        (thread_id, label, kw_text),
    )

    return True, None


def build(
    db_path: Path, force: bool = False, active_days: int = 90, kinds: str = "tag,person"
) -> Dict[str, Any]:
    """
    Build thread indices for tags and people.

    Args:
        db_path: Path to SQLite database
        force: Force rebuild (ignore src_hash)
        active_days: Active window in days
        kinds: Comma-separated list ('tag', 'person', or 'tag,person')

    Returns:
        {
            "success": bool,
            "stats": {
                "thread_count": 127,
                "tag_threads": 45,
                "person_threads": 82,
                "active_count": 89,
                "stale_count": 38,
                "new_count": 15,
                "updated_count": 8,
                "skipped_count": 104
            },
            "duration_seconds": 1.8,
            "error": Optional[str]
        }
    """
    start_time = time.time()

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Parse kinds
        kind_list = [k.strip() for k in kinds.split(",")]

        stats = {
            "thread_count": 0,
            "tag_threads": 0,
            "person_threads": 0,
            "active_count": 0,
            "stale_count": 0,
            "new_count": 0,
            "updated_count": 0,
            "skipped_count": 0,
        }

        # Build tag threads
        if "tag" in kind_list:
            cursor.execute("SELECT DISTINCT name FROM tags ORDER BY name")
            tags = [row[0] for row in cursor.fetchall()]

            for tag_name in tags:
                was_built, error = build_thread(
                    conn, "tag", tag_name, tag_name.lower(), active_days, force
                )

                if was_built:
                    stats["new_count"] += 1
                else:
                    stats["skipped_count"] += 1

                stats["tag_threads"] += 1

            # Commit after tags
            conn.commit()

        # Build person threads
        if "person" in kind_list:
            cursor.execute("SELECT DISTINCT name FROM people ORDER BY name")
            people = [row[0] for row in cursor.fetchall()]

            for person_name in people:
                was_built, error = build_thread(
                    conn, "person", person_name, person_name.lower(), active_days, force
                )

                if was_built:
                    stats["new_count"] += 1
                else:
                    stats["skipped_count"] += 1

                stats["person_threads"] += 1

            # Commit after people
            conn.commit()

        # Count final stats
        stats["thread_count"] = stats["tag_threads"] + stats["person_threads"]

        # Count by status
        cursor.execute("SELECT COUNT(*) FROM threads WHERE status = 'active'")
        stats["active_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM threads WHERE status = 'stale'")
        stats["stale_count"] = cursor.fetchone()[0]

        conn.close()

        duration = time.time() - start_time

        return {
            "success": True,
            "stats": stats,
            "duration_seconds": round(duration, 2),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stats": {},
            "duration_seconds": time.time() - start_time,
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build ThreadPacks indices for memex-twos-mcp"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/processed/twos.db"),
        help="Path to SQLite database (default: data/processed/twos.db)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rebuild (ignore src_hash)"
    )
    parser.add_argument(
        "--active-days",
        type=int,
        default=90,
        help="Active window in days (default: 90)",
    )
    parser.add_argument(
        "--kinds",
        type=str,
        default="tag,person",
        help="Comma-separated kinds to build (default: tag,person)",
    )

    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    print(f"Building threads for {args.db}...")
    print(f"  Force rebuild: {args.force}")
    print(f"  Active window: {args.active_days} days")
    print(f"  Kinds: {args.kinds}")
    print()

    result = build(
        db_path=args.db,
        force=args.force,
        active_days=args.active_days,
        kinds=args.kinds,
    )

    if result["success"]:
        print("✓ Build completed successfully")
        print()
        print("Statistics:")
        stats = result["stats"]
        print(f"  Total threads: {stats['thread_count']}")
        print(f"    Tags: {stats['tag_threads']}")
        print(f"    People: {stats['person_threads']}")
        print()
        print(f"  Active: {stats['active_count']}")
        print(f"  Stale: {stats['stale_count']}")
        print()
        print(f"  New: {stats['new_count']}")
        print(f"  Updated: {stats['updated_count']}")
        print(f"  Skipped: {stats['skipped_count']}")
        print()
        print(f"  Duration: {result['duration_seconds']}s")
    else:
        print(f"✗ Build failed: {result['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
