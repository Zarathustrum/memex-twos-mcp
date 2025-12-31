#!/usr/bin/env python3
"""
MonthlySummaries Builder

Generates LLM-powered semantic summaries for monthly data exploration.
Uses Claude Code CLI for semantic analysis.

I/O Boundary: Reads from SQLite database, writes summaries back to database.
External API: Invokes Claude Code CLI via subprocess (uses user's API quota).

Purpose:
- TimePacks provide mechanical facts ("what happened")
- MonthlySummaries provide semantic framing ("so what")
- Enables "system prompt lite" context for monthly exploration
- All insights anchored to specific thing IDs (no hallucinations)

Billing/Quota:
- Each month summary consumes ~2K-5K tokens (input + output)
- Building 12 months = ~24K-60K tokens (~$0.06-$0.15 at Sonnet rates)
- Uses user's Claude API quota via Claude Code CLI
- Rate limits may apply (handled by Claude Code CLI)

MS1 Pack Format:
MS1|m=<YYYY-MM>|n=<total>|tg=<tag:count,...>|pp=<person:count,...>|th=<theme@thing_id,...;...>|hi=<thing_id~label;...>|nq=<question_count>
"""

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse scoring from build_timepacks
sys.path.insert(0, str(Path(__file__).parent))
try:
    from build_timepacks import compute_src_hash, score_highlight, make_label
except ImportError:
    print("ERROR: Could not import from build_timepacks.py", file=sys.stderr)
    sys.exit(1)


def invoke_llm_via_claude_code(prompt: str, timeout: int = 120) -> Dict[str, Any]:
    """
    Invoke Claude Code CLI for semantic analysis.

    External dependency: Requires `claude-code` CLI installed and authenticated.
    Billing: Consumes user's Claude API quota (~2K-5K tokens per call).

    Security considerations:
    - subprocess.run() with shell=True (via bash -c) for pipe compatibility
    - Temp file used to avoid command-line injection (prompt could contain shell metacharacters)
    - Timeout enforced to prevent indefinite hangs (default 120s)
    - Temp file always cleaned up even on error (finally block)

    Failure modes:
    - Claude Code CLI not installed → RuntimeError
    - API rate limit hit → RuntimeError (retry manually)
    - Timeout exceeded → subprocess.TimeoutExpired → RuntimeError
    - Invalid JSON in response → json.JSONDecodeError → RuntimeError
    - Network failure → RuntimeError

    Response parsing:
    - Claude Code returns markdown with JSON code blocks
    - Extracts JSON from ```json...``` blocks or raw JSON
    - Lenient parsing (tries multiple strategies)

    Args:
        prompt: Analysis prompt (can be large, ~1K-2K chars)
        timeout: Timeout in seconds (default 120s for API calls)

    Returns:
        Parsed JSON response from LLM

    Raises:
        RuntimeError: If LLM invocation fails (any reason)
        subprocess.TimeoutExpired: If timeout exceeded
    """
    # Write prompt to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        prompt_file = Path(f.name)
        f.write(prompt)

    try:
        # Invoke claude-code with prompt file
        # Use echo to pipe the prompt as the user would in interactive mode
        result = subprocess.run(
            ["bash", "-c", f"cat {prompt_file} | claude-code --model sonnet"],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"Claude Code invocation failed: {result.stderr}")

        # Parse JSON from response (should be in stdout)
        # Claude Code typically returns markdown with JSON code blocks
        output = result.stdout

        # Extract JSON from markdown code block if present
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise RuntimeError(f"No JSON found in Claude Code response: {output[:500]}")

        response = json.loads(json_str)
        return response

    finally:
        # Cleanup temp file
        if prompt_file.exists():
            prompt_file.unlink()


def fetch_things_in_month(
    conn: sqlite3.Connection,
    month_start: str,
    month_end: str
) -> List[Dict[str, Any]]:
    """
    Fetch all things in month with full details.

    Args:
        conn: Database connection
        month_start: ISO date (YYYY-MM-01)
        month_end: ISO date (last day of month)

    Returns:
        List of thing dictionaries with tags and people
    """
    cursor = conn.cursor()
    cursor.row_factory = sqlite3.Row

    cursor.execute(
        """
        SELECT * FROM things
        WHERE DATE(timestamp) BETWEEN ? AND ?
        ORDER BY timestamp DESC
        """,
        (month_start, month_end)
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


def build_llm_prompt(
    month_id: str,
    start_date: str,
    end_date: str,
    thing_count: int,
    tags_summary: str,
    people_summary: str,
    candidates: List[Dict[str, Any]]
) -> str:
    """
    Build prompt for LLM semantic analysis.

    Args:
        month_id: YYYY-MM
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        thing_count: Total things in month
        tags_summary: Top tags string
        people_summary: Top people string
        candidates: Top 30 candidate things

    Returns:
        Formatted prompt string
    """
    # Format candidates
    candidate_lines = []
    for i, thing in enumerate(candidates, 1):
        content_preview = thing.get("content", "")[:100]
        tags = ", ".join(thing.get("tags", [])[:3])
        people = ", ".join(thing.get("people_mentioned", [])[:2])

        candidate_lines.append(
            f"{i}. {thing['id']} | {thing.get('timestamp', '')[:10]} | {content_preview} "
            f"| tags:[{tags}] | people:[{people}]"
        )

    candidates_text = "\n".join(candidate_lines)

    prompt = f"""You are analyzing a month of personal data for contextual framing.

Month: {month_id}
Date range: {start_date} to {end_date}
Total things: {thing_count}

Top tags: {tags_summary}
Top people: {people_summary}

Candidate highlights (top 30 by relevance):
{candidates_text}

Your task:
1. Identify 3-8 semantic themes (clusters of related activity)
   - Each theme must be grounded to at least 2 thing IDs from candidates above
   - Theme names: snake_case, <=32 chars, descriptive (e.g., "work_planning", "home_maintenance")

2. Select 10-12 best highlights that represent the month
   - Must be from candidate list only (use exact thing IDs shown above)
   - Prefer: recency, diversity of themes, substantive content
   - Generate a short label for each highlight (snake_case, <=32 chars)

3. Suggest 3-5 follow-up questions for exploration
   - Focus on: progress, changes, patterns, unresolved items
   - Anchor each question to specific thing IDs from the candidates
   - Keep questions <100 chars
   - Provide a thread_id suggestion (format: "thr:tag:TAG" or "thr:person:NAME")

CRITICAL: All thing_ids in your response MUST be from the candidate list above. Do not hallucinate IDs.

Return ONLY valid JSON (no markdown, no explanation):
{{
  "themes": [
    {{"name": "work_planning", "thing_ids": ["task_08190", "task_08155"]}},
    {{"name": "health_care", "thing_ids": ["task_08123", "task_08001"]}}
  ],
  "highlights": [
    {{"thing_id": "task_08190", "label": "q4_review"}},
    {{"thing_id": "task_08123", "label": "dentist_appt"}}
  ],
  "questions": [
    {{
      "text": "What progress on Q4 planning?",
      "anchors": ["task_08190", "task_08155"],
      "thread_id": "thr:tag:work",
      "rationale": "High activity in work thread"
    }}
  ]
}}"""

    return prompt


def validate_llm_response(
    response: Dict[str, Any],
    candidate_ids: set
) -> Tuple[bool, Optional[str]]:
    """
    Validate LLM response structure and content.

    Args:
        response: Parsed JSON from LLM
        candidate_ids: Set of valid thing IDs

    Returns:
        (is_valid, error_message)
    """
    # Check required keys
    if not all(k in response for k in ["themes", "highlights", "questions"]):
        return False, "Missing required keys (themes, highlights, questions)"

    # Validate themes
    themes = response.get("themes", [])
    if not (3 <= len(themes) <= 8):
        return False, f"Expected 3-8 themes, got {len(themes)}"

    for theme in themes:
        if not isinstance(theme, dict) or "name" not in theme or "thing_ids" not in theme:
            return False, "Invalid theme structure"

        name = theme["name"]
        if not re.match(r'^[a-z_]{1,32}$', name):
            return False, f"Invalid theme name: {name}"

        thing_ids = theme["thing_ids"]
        if len(thing_ids) < 2:
            return False, f"Theme {name} needs at least 2 thing IDs"

        for tid in thing_ids:
            if tid not in candidate_ids:
                return False, f"Theme {name} references invalid thing_id: {tid}"

    # Validate highlights
    highlights = response.get("highlights", [])
    if not (10 <= len(highlights) <= 12):
        return False, f"Expected 10-12 highlights, got {len(highlights)}"

    for hl in highlights:
        if not isinstance(hl, dict) or "thing_id" not in hl or "label" not in hl:
            return False, "Invalid highlight structure"

        if hl["thing_id"] not in candidate_ids:
            return False, f"Highlight references invalid thing_id: {hl['thing_id']}"

        label = hl["label"]
        if not re.match(r'^[a-z0-9_]{1,32}$', label):
            return False, f"Invalid highlight label: {label}"

    # Validate questions
    questions = response.get("questions", [])
    if not (3 <= len(questions) <= 5):
        return False, f"Expected 3-5 questions, got {len(questions)}"

    for q in questions:
        if not isinstance(q, dict) or "text" not in q or "anchors" not in q:
            return False, "Invalid question structure"

        if len(q["text"]) > 100:
            return False, f"Question too long (>100 chars): {q['text']}"

        for anchor in q["anchors"]:
            if anchor not in candidate_ids:
                return False, f"Question anchor references invalid thing_id: {anchor}"

    return True, None


def build_ms1_pack(
    month_id: str,
    thing_count: int,
    tags_summary: str,
    people_summary: str,
    themes: List[Dict[str, Any]],
    highlights: List[Dict[str, Any]],
    question_count: int
) -> str:
    """
    Build MS1 format pack string.

    MS1|m=<YYYY-MM>|n=<total>|tg=<tag:count,...>|pp=<person:count,...>|th=<theme@thing_id,...;...>|hi=<thing_id~label;...>|nq=<question_count>

    Args:
        month_id: YYYY-MM
        thing_count: Total things
        tags_summary: Tag frequencies
        people_summary: People frequencies
        themes: Theme list with thing_ids
        highlights: Highlight list with labels
        question_count: Number of questions

    Returns:
        MS1 pack string
    """
    # Build theme string
    theme_parts = []
    for theme in themes:
        name = theme["name"]
        ids = ",".join(theme["thing_ids"])
        theme_parts.append(f"{name}@{ids}")
    th_str = ";".join(theme_parts) if theme_parts else ""

    # Build highlight string
    hi_parts = []
    for hl in highlights:
        hi_parts.append(f"{hl['thing_id']}~{hl['label']}")
    hi_str = ";".join(hi_parts) if hi_parts else ""

    # Build pack
    pack = (
        f"MS1|m={month_id}|n={thing_count}|tg={tags_summary}|pp={people_summary}|"
        f"th={th_str}|hi={hi_str}|nq={question_count}"
    )

    # Ensure bounded length (~1200 chars max)
    if len(pack) > 1200:
        # Truncate themes if needed
        while theme_parts and len(pack) > 1200:
            theme_parts.pop()
            th_str = ";".join(theme_parts)
            pack = (
                f"MS1|m={month_id}|n={thing_count}|tg={tags_summary}|pp={people_summary}|"
                f"th={th_str}|hi={hi_str}|nq={question_count}"
            )

    return pack


def build_month_summary(
    conn: sqlite3.Connection,
    month_start: date,
    month_end: date,
    force: bool = False,
    dry_run: bool = False,
    builder_v: str = "1.0"
) -> Tuple[bool, Optional[str]]:
    """
    Build a single month summary.

    Args:
        conn: Database connection
        month_start: First day of month
        month_end: Last day of month
        force: Force rebuild even if src_hash matches
        dry_run: Don't invoke LLM, just show what would be built
        builder_v: Builder version

    Returns:
        (was_built, error_message)
    """
    cursor = conn.cursor()

    month_id = month_start.strftime("%Y-%m")
    print(f"  Processing {month_id}...", end=" ", flush=True)

    # Fetch things in month
    things = fetch_things_in_month(
        conn,
        month_start.isoformat(),
        month_end.isoformat()
    )

    if not things:
        print("SKIP (no data)")
        return False, None

    # Compute src_hash
    src_hash = compute_src_hash(things)

    # Check existing summary
    cursor.execute(
        "SELECT src_hash FROM month_summaries WHERE month_id = ?",
        (month_id,)
    )
    row = cursor.fetchone()

    if row and row[0] == src_hash and not force:
        print("SKIP (hash match)")
        return False, None

    if dry_run:
        print(f"DRY-RUN ({len(things)} things)")
        return False, None

    # Score candidates (top 30)
    window_days = (month_end - month_start).days + 1
    scored = []
    for thing in things:
        score = score_highlight(thing, month_end, window_days)
        scored.append((score, thing))

    scored.sort(reverse=True, key=lambda x: x[0])
    candidates = [t for _, t in scored[:30]]
    candidate_ids = {t["id"] for t in candidates}

    # Build aggregates
    tag_counts = Counter()
    people_counts = Counter()
    for thing in things:
        for tag in thing.get("tags", []):
            tag_counts[tag] += 1
        for person in thing.get("people_mentioned", []):
            people_counts[person] += 1

    top_tags = tag_counts.most_common(8)
    tags_summary = ",".join(f"{t}:{c}" for t, c in top_tags) if top_tags else ""

    top_people = people_counts.most_common(8)
    people_summary = ",".join(f"{p}:{c}" for p, c in top_people) if top_people else ""

    # Build LLM prompt
    prompt = build_llm_prompt(
        month_id,
        month_start.isoformat(),
        month_end.isoformat(),
        len(things),
        tags_summary,
        people_summary,
        candidates
    )

    # Invoke LLM
    try:
        print("LLM...", end=" ", flush=True)
        llm_response = invoke_llm_via_claude_code(prompt)
    except Exception as e:
        error_msg = f"LLM invocation failed: {str(e)}"
        print(f"FAIL ({error_msg})")
        return False, error_msg

    # Validate response
    is_valid, validation_error = validate_llm_response(llm_response, candidate_ids)

    if not is_valid:
        error_msg = f"Validation failed: {validation_error}"
        print(f"FAIL ({error_msg})")
        return False, error_msg

    # Build MS1 pack
    pack = build_ms1_pack(
        month_id,
        len(things),
        tags_summary,
        people_summary,
        llm_response["themes"],
        llm_response["highlights"],
        len(llm_response["questions"])
    )

    # Build questions JSON
    questions_data = {"questions": []}
    for rank, q in enumerate(llm_response["questions"], 1):
        questions_data["questions"].append({
            "rank": rank,
            "text": q["text"],
            "anchors": q["anchors"],
            "thread_id": q.get("thread_id", ""),
            "rationale": q.get("rationale", "")
        })

    questions_json = json.dumps(questions_data)

    # Delete old summary + evidence
    cursor.execute("DELETE FROM month_summary_evidence WHERE month_id = ?", (month_id,))
    cursor.execute("DELETE FROM month_summaries WHERE month_id = ?", (month_id,))

    # Insert summary
    cursor.execute(
        """
        INSERT INTO month_summaries (
            month_id, start_date, end_date, thing_count,
            pack_v, pack, suggested_questions,
            src_hash, builder_v, llm_model, llm_conf
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            month_id,
            month_start.isoformat(),
            month_end.isoformat(),
            len(things),
            1,
            pack,
            questions_json,
            src_hash,
            builder_v,
            "claude-sonnet-4-5",
            0.95  # High confidence for validated responses
        )
    )

    # Insert highlights
    for rank, hl in enumerate(llm_response["highlights"]):
        cursor.execute(
            """
            INSERT INTO month_summary_evidence (month_id, thing_id, role, rank)
            VALUES (?, ?, 'hi', ?)
            """,
            (month_id, hl["thing_id"], rank)
        )

    # Insert evidence (remaining candidates)
    evidence_ids = [t["id"] for t in candidates if t["id"] not in {h["thing_id"] for h in llm_response["highlights"]}]
    for rank, thing_id in enumerate(evidence_ids[:20]):
        cursor.execute(
            """
            INSERT INTO month_summary_evidence (month_id, thing_id, role, rank)
            VALUES (?, ?, 'ev', ?)
            """,
            (month_id, thing_id, rank)
        )

    print("OK")
    return True, None


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


def build(
    db_path: Path,
    force: bool = False,
    months: int = 12,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Build LLM-powered monthly summaries.

    Args:
        db_path: Path to SQLite database
        force: Force rebuild (ignore src_hash)
        months: Number of months of history to build
        dry_run: Show what would be built, don't invoke LLM

    Returns:
        {
            "success": bool,
            "stats": {
                "summary_count": 12,
                "new_count": 3,
                "updated_count": 2,
                "skipped_count": 7,
                "failed_count": 0
            },
            "duration_seconds": 45.3,
            "error": Optional[str]
        }
    """
    start_time = time.time()

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

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

        # Generate month windows
        windows = generate_month_windows(min_date, max_date)

        stats = {
            "summary_count": 0,
            "new_count": 0,
            "updated_count": 0,
            "skipped_count": 0,
            "failed_count": 0
        }

        # Build each month
        for month_start, month_end in windows:
            was_built, error = build_month_summary(
                conn, month_start, month_end, force, dry_run
            )

            if error:
                stats["failed_count"] += 1
            elif was_built:
                stats["new_count"] += 1
            else:
                stats["skipped_count"] += 1

            stats["summary_count"] += 1

            # Commit after each month
            conn.commit()

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
        description="Build LLM-powered monthly summaries for memex-twos-mcp"
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
        "--dry-run",
        action="store_true",
        help="Show what would be built, don't invoke LLM"
    )

    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    print(f"Building monthly summaries for {args.db}...")
    print(f"  Force rebuild: {args.force}")
    print(f"  History: {args.months} months")
    print(f"  Dry run: {args.dry_run}")
    print()

    result = build(
        db_path=args.db,
        force=args.force,
        months=args.months,
        dry_run=args.dry_run
    )

    if result["success"]:
        print()
        print("✓ Build completed successfully")
        print()
        print("Statistics:")
        stats = result["stats"]
        print(f"  Total months: {stats['summary_count']}")
        print(f"  New: {stats['new_count']}")
        print(f"  Updated: {stats['updated_count']}")
        print(f"  Skipped: {stats['skipped_count']}")
        print(f"  Failed: {stats['failed_count']}")
        print()
        print(f"  Duration: {result['duration_seconds']}s")
    else:
        print(f"✗ Build failed: {result['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
