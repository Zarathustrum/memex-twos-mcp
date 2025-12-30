#!/usr/bin/env python3
"""
TimePacks Builder

Builds day/week/month rollups stored in the database for fast
"what happened last week/month?" queries.

TP1 Pack Format:
TP1|k=<d|w|m>|s=<start_date>|e=<end_date>|n=<total>|cx=<completed>|pn=<pending>|st=<strikethrough>|tg=<tag:count,...>|pp=<person:count,...>|kw=<word,word,...>|hi=<thing_id~label;...>
"""

import argparse
import hashlib
import math
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Common English stopwords for keyword extraction
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "don",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "might",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "s", "same", "she", "should", "so", "some", "such", "t", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "would", "you", "your", "yours", "yourself", "yourselves"
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


def score_highlight(
    thing: Dict[str, Any],
    end_date: date,
    window_days: int
) -> float:
    """
    Deterministic highlight scoring using weighted formula.

    score = (
        recency_score * 0.40 +
        entity_density_score * 0.30 +
        content_length_score * 0.20 +
        status_score * 0.10
    )

    Args:
        thing: Thing dictionary
        end_date: End of rollup window
        window_days: Size of rollup window in days

    Returns:
        Composite score (0-1)
    """
    # Parse timestamp
    ts = thing.get("timestamp", "")
    try:
        thing_date = datetime.fromisoformat(ts).date()
    except (ValueError, TypeError):
        thing_date = end_date

    # Recency score: exponential decay from end_date
    days_ago = (end_date - thing_date).days
    recency_score = math.exp(-days_ago / max(window_days, 1))

    # Entity density: (tags + people) / 10, capped at 1.0
    tag_count = len(thing.get("tags", []))
    people_count = len(thing.get("people_mentioned", []))
    entity_density_score = min((tag_count + people_count) / 10.0, 1.0)

    # Content length: prefer substantive items
    content_len = len(thing.get("content", ""))
    content_length_score = min(content_len / 200.0, 1.0)

    # Status score: favor incomplete/pending items
    if thing.get("is_completed"):
        status_score = 0.0
    elif thing.get("is_pending"):
        status_score = 0.5
    else:
        status_score = 1.0

    # Weighted combination
    score = (
        recency_score * 0.40 +
        entity_density_score * 0.30 +
        content_length_score * 0.20 +
        status_score * 0.10
    )

    return score


def extract_keywords(things: List[Dict[str, Any]], top_n: int = 20) -> List[str]:
    """
    Deterministic keyword extraction using TF within window.

    1. Collect all tag names
    2. Tokenize content, remove stopwords
    3. Compute term frequency
    4. Select top N by TF
    5. Sort alphabetically for stability

    Args:
        things: List of thing dictionaries
        top_n: Number of keywords to return

    Returns:
        List of keywords sorted alphabetically
    """
    token_counts = Counter()

    # Collect tag names (already normalized)
    for thing in things:
        for tag in thing.get("tags", []):
            token_counts[tag.lower()] += 1

    # Tokenize content
    for thing in things:
        content = thing.get("content", "").lower()
        # Extract words (alphanumeric sequences)
        tokens = re.findall(r'\b[a-z]+\b', content)

        for token in tokens:
            # Skip stopwords, single chars, numbers
            if token not in STOPWORDS and len(token) > 1:
                token_counts[token] += 1

    # Get top N by frequency, then sort alphabetically
    top_tokens = [word for word, _ in token_counts.most_common(top_n)]
    top_tokens.sort()

    return top_tokens


def make_label(content: str, max_len: int = 32) -> str:
    """
    Generate snake_case label from content (extractive).

    Args:
        content: Thing content
        max_len: Maximum label length

    Returns:
        Snake-cased label
    """
    # Extract first few words
    words = re.findall(r'\b[a-zA-Z0-9]+\b', content)[:4]

    # Convert to snake_case
    label = "_".join(words).lower()

    # Truncate to max length
    if len(label) > max_len:
        label = label[:max_len]

    return label if label else "item"


def build_tp1_pack(
    kind: str,
    start_date: str,
    end_date: str,
    things: List[Dict[str, Any]],
    highlights: List[Dict[str, Any]]
) -> str:
    """
    Build TP1 format pack string.

    TP1|k=<d|w|m>|s=<start_date>|e=<end_date>|n=<total>|cx=<completed>|pn=<pending>|st=<strikethrough>|tg=<tag:count,...>|pp=<person:count,...>|kw=<word,word,...>|hi=<thing_id~label;...>

    Args:
        kind: 'd', 'w', or 'm'
        start_date: ISO date string
        end_date: ISO date string
        things: All things in window
        highlights: Top 10 scored highlights

    Returns:
        TP1 pack string
    """
    # Counts
    n = len(things)
    cx = sum(1 for t in things if t.get("is_completed"))
    pn = sum(1 for t in things if t.get("is_pending"))
    st = sum(1 for t in things if t.get("is_strikethrough"))

    # Tag frequencies (top 8)
    tag_counts = Counter()
    for thing in things:
        for tag in thing.get("tags", []):
            tag_counts[tag.lower()] += 1

    top_tags = tag_counts.most_common(8)
    tg_str = ",".join(f"{tag}:{count}" for tag, count in top_tags) if top_tags else ""

    # People frequencies (top 8)
    people_counts = Counter()
    for thing in things:
        for person in thing.get("people_mentioned", []):
            people_counts[person] += 1

    top_people = people_counts.most_common(8)
    pp_str = ",".join(f"{person}:{count}" for person, count in top_people) if top_people else ""

    # Keywords (top 20)
    keywords = extract_keywords(things, top_n=20)
    kw_str = ",".join(keywords) if keywords else ""

    # Highlights (top 10)
    hi_parts = []
    for hl in highlights[:10]:
        thing_id = hl["id"]
        label = make_label(hl.get("content", ""))
        hi_parts.append(f"{thing_id}~{label}")

    hi_str = ";".join(hi_parts) if hi_parts else ""

    # Build pack
    pack = (
        f"TP1|k={kind}|s={start_date}|e={end_date}|n={n}|cx={cx}|pn={pn}|st={st}|"
        f"tg={tg_str}|pp={pp_str}|kw={kw_str}|hi={hi_str}"
    )

    # Ensure pack is bounded (~800 chars max)
    if len(pack) > 800:
        # Truncate highlights if needed
        while hi_parts and len(pack) > 800:
            hi_parts.pop()
            hi_str = ";".join(hi_parts)
            pack = (
                f"TP1|k={kind}|s={start_date}|e={end_date}|n={n}|cx={cx}|pn={pn}|st={st}|"
                f"tg={tg_str}|pp={pp_str}|kw={kw_str}|hi={hi_str}"
            )

    return pack


def fetch_things_in_window(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str
) -> List[Dict[str, Any]]:
    """
    Fetch all things with full details in date range.

    Args:
        conn: Database connection
        start_date: ISO date (inclusive)
        end_date: ISO date (inclusive)

    Returns:
        List of thing dictionaries with tags and people
    """
    cursor = conn.cursor()
    cursor.row_factory = sqlite3.Row

    # Fetch things
    cursor.execute(
        """
        SELECT * FROM things
        WHERE DATE(timestamp) BETWEEN ? AND ?
        ORDER BY timestamp
        """,
        (start_date, end_date)
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
            (thing_id,)
        )
        thing["tags"] = [r[0] for r in cursor.fetchall()]

        # Fetch people
        cursor.execute(
            """
            SELECT p.name FROM people p
            JOIN thing_people tp ON p.id = tp.person_id
            WHERE tp.thing_id = ?
            """,
            (thing_id,)
        )
        thing["people_mentioned"] = [r[0] for r in cursor.fetchall()]

        things.append(thing)

    return things


def generate_day_windows(
    start_date: date,
    end_date: date
) -> List[Tuple[date, date]]:
    """Generate day windows (single day each)."""
    windows = []
    current = start_date

    while current <= end_date:
        windows.append((current, current))
        current += timedelta(days=1)

    return windows


def generate_week_windows(
    start_date: date,
    end_date: date
) -> List[Tuple[date, date]]:
    """
    Generate ISO 8601 week windows (Monday-based).

    Week ID = Monday date.
    """
    windows = []

    # Find first Monday on or before start_date
    current = start_date
    while current.weekday() != 0:  # 0 = Monday
        current -= timedelta(days=1)

    while current <= end_date:
        week_start = current
        week_end = current + timedelta(days=6)  # Sunday

        # Only include weeks that overlap with data range
        if week_end >= start_date and week_start <= end_date:
            windows.append((week_start, week_end))

        current += timedelta(days=7)

    return windows


def generate_month_windows(
    start_date: date,
    end_date: date
) -> List[Tuple[date, date]]:
    """Generate month windows (calendar months)."""
    windows = []

    # Start from first day of start_date's month
    current = date(start_date.year, start_date.month, 1)

    while current <= end_date:
        # Last day of month
        if current.month == 12:
            month_end = date(current.year, 12, 31)
        else:
            month_end = date(current.year, current.month + 1, 1) - timedelta(days=1)

        windows.append((current, month_end))

        # Next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return windows


def build_rollup(
    conn: sqlite3.Connection,
    kind: str,
    start_date: date,
    end_date: date,
    force: bool = False,
    builder_v: str = "1.0"
) -> bool:
    """
    Build a single rollup window.

    Args:
        conn: Database connection
        kind: 'd', 'w', or 'm'
        start_date: Window start
        end_date: Window end
        force: Rebuild even if src_hash matches
        builder_v: Builder version

    Returns:
        True if rollup was built/updated, False if skipped
    """
    cursor = conn.cursor()

    # Generate rollup_id
    if kind == 'd':
        rollup_id = f"d:{start_date.isoformat()}"
    elif kind == 'w':
        rollup_id = f"w:{start_date.isoformat()}"  # Monday date
    else:  # 'm'
        rollup_id = f"m:{start_date.strftime('%Y-%m')}"

    # Fetch things in window
    things = fetch_things_in_window(
        conn,
        start_date.isoformat(),
        end_date.isoformat()
    )

    if not things:
        # No data in window, skip
        return False

    # Compute src_hash
    src_hash = compute_src_hash(things)

    # Check existing rollup
    cursor.execute(
        "SELECT src_hash FROM rollups WHERE rollup_id = ?",
        (rollup_id,)
    )
    row = cursor.fetchone()

    if row and row[0] == src_hash and not force:
        # Hash matches, skip rebuild
        return False

    # Score highlights
    window_days = (end_date - start_date).days + 1
    scored_things = []
    for thing in things:
        score = score_highlight(thing, end_date, window_days)
        scored_things.append((score, thing))

    # Sort by score descending
    scored_things.sort(reverse=True, key=lambda x: x[0])

    # Top 10 = highlights, next 20 = evidence
    highlights = [t for _, t in scored_things[:10]]
    evidence = [t for _, t in scored_things[10:30]]

    # Build TP1 pack
    pack = build_tp1_pack(
        kind,
        start_date.isoformat(),
        end_date.isoformat(),
        things,
        highlights
    )

    # Extract keywords for search column
    keywords = extract_keywords(things, top_n=20)
    kw_text = " ".join(keywords)

    # Counts
    thing_count = len(things)
    completed_count = sum(1 for t in things if t.get("is_completed"))
    pending_count = sum(1 for t in things if t.get("is_pending"))
    strikethrough_count = sum(1 for t in things if t.get("is_strikethrough"))

    # Delete old rollup + evidence
    cursor.execute("DELETE FROM rollup_evidence WHERE rollup_id = ?", (rollup_id,))
    cursor.execute("DELETE FROM rollups WHERE rollup_id = ?", (rollup_id,))

    # Insert rollup
    cursor.execute(
        """
        INSERT INTO rollups (
            rollup_id, kind, start_date, end_date,
            thing_count, completed_count, pending_count, strikethrough_count,
            pack_v, pack, kw, src_hash, builder_v
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rollup_id, kind, start_date.isoformat(), end_date.isoformat(),
            thing_count, completed_count, pending_count, strikethrough_count,
            1, pack, kw_text, src_hash, builder_v
        )
    )

    # Insert highlights
    for rank, thing in enumerate(highlights):
        cursor.execute(
            """
            INSERT INTO rollup_evidence (rollup_id, thing_id, role, rank)
            VALUES (?, ?, 'hi', ?)
            """,
            (rollup_id, thing["id"], rank)
        )

    # Insert evidence
    for rank, thing in enumerate(evidence):
        cursor.execute(
            """
            INSERT INTO rollup_evidence (rollup_id, thing_id, role, rank)
            VALUES (?, ?, 'ev', ?)
            """,
            (rollup_id, thing["id"], rank)
        )

    return True


def build(
    db_path: Path,
    force: bool = False,
    months: int = 12,
    kinds: str = "d,w,m"
) -> Dict[str, Any]:
    """
    Build time-based rollups (days, weeks, months).

    Args:
        db_path: Path to SQLite database
        force: Force rebuild (ignore src_hash)
        months: Number of months of history to build
        kinds: Comma-separated list of kinds to build (d,w,m)

    Returns:
        {
            "success": bool,
            "stats": {
                "rollup_count": 858,
                "day_count": 730,
                "week_count": 104,
                "month_count": 24,
                "new_count": 120,
                "updated_count": 10,
                "skipped_count": 728
            },
            "duration_seconds": 3.2,
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

        # Get date range from things table
        cursor.execute(
            "SELECT MIN(DATE(timestamp)), MAX(DATE(timestamp)) FROM things"
        )
        row = cursor.fetchone()

        if not row[0] or not row[1]:
            return {
                "success": False,
                "error": "No things found in database",
                "stats": {},
                "duration_seconds": time.time() - start_time
            }

        min_date = datetime.fromisoformat(row[0]).date()
        max_date = datetime.fromisoformat(row[1]).date()

        # Limit to last N months
        cutoff_date = max_date - timedelta(days=months * 30)
        if min_date < cutoff_date:
            min_date = cutoff_date

        # Generate windows
        stats = {
            "day_count": 0,
            "week_count": 0,
            "month_count": 0,
            "new_count": 0,
            "updated_count": 0,
            "skipped_count": 0
        }

        # Build rollups by kind
        for kind in kind_list:
            if kind == 'd':
                windows = generate_day_windows(min_date, max_date)
            elif kind == 'w':
                windows = generate_week_windows(min_date, max_date)
            elif kind == 'm':
                windows = generate_month_windows(min_date, max_date)
            else:
                continue

            # Build each window
            for start_date, end_date in windows:
                was_built = build_rollup(
                    conn, kind, start_date, end_date, force
                )

                if was_built:
                    stats["new_count"] += 1
                else:
                    stats["skipped_count"] += 1

                # Update kind count
                if kind == 'd':
                    stats["day_count"] += 1
                elif kind == 'w':
                    stats["week_count"] += 1
                else:
                    stats["month_count"] += 1

            # Commit after each kind (batch commit)
            conn.commit()

        stats["rollup_count"] = stats["day_count"] + stats["week_count"] + stats["month_count"]

        conn.close()

        duration = time.time() - start_time

        return {
            "success": True,
            "stats": stats,
            "duration_seconds": round(duration, 2),
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stats": {},
            "duration_seconds": time.time() - start_time
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build TimePacks rollups for memex-twos-mcp"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/processed/twos.db"),
        help="Path to SQLite database (default: data/processed/twos.db)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild (ignore src_hash)"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Number of months of history to build (default: 12)"
    )
    parser.add_argument(
        "--kinds",
        type=str,
        default="d,w,m",
        help="Comma-separated rollup kinds to build (default: d,w,m)"
    )

    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    print(f"Building timepacks for {args.db}...")
    print(f"  Force rebuild: {args.force}")
    print(f"  History: {args.months} months")
    print(f"  Kinds: {args.kinds}")
    print()

    result = build(
        db_path=args.db,
        force=args.force,
        months=args.months,
        kinds=args.kinds
    )

    if result["success"]:
        print("✓ Build completed successfully")
        print()
        print("Statistics:")
        stats = result["stats"]
        print(f"  Total rollups: {stats['rollup_count']}")
        print(f"    Days: {stats['day_count']}")
        print(f"    Weeks: {stats['week_count']}")
        print(f"    Months: {stats['month_count']}")
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
