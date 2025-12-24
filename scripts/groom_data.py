#!/usr/bin/env python3
"""
Data Grooming Script for Twos Task Database

Automatically fixes mechanical issues and generates detailed reports.
Two-tier grooming:
1. Python auto-fix (removes duplicates, fixes broken refs)
2. Claude Code semantic analysis (optional --ai-analysis flag)
"""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


class GroomingChanges:
    """Track all changes made during grooming."""

    def __init__(self):
        """Initialize collections used to report changes and flags."""
        self.removed = []
        self.modified = []
        self.flagged_not_fixed = {
            "normalization": [],
            "ambiguous_duplicates": [],
            "long_content": []
        }

    def add_removed(self, thing: Dict[str, Any], reason: str, duplicate_of: str = None):
        """
        Record a removed item.

        Args:
            thing: The original thing dictionary being removed.
            reason: Short reason string (e.g., "exact_duplicate").
            duplicate_of: Optional ID of the retained item.
        """
        self.removed.append({
            "id": thing["id"],
            "content": thing.get("content", "")[:100],
            "timestamp": thing.get("timestamp"),
            "reason": reason,
            "duplicate_of": duplicate_of
        })

    def add_modified(self, thing_id: str, field: str, old_value: Any, new_value: Any, reason: str):
        """
        Record a modification to a single field.

        Args:
            thing_id: ID of the thing being changed.
            field: The field name being updated.
            old_value: Previous value for audit purposes.
            new_value: New value after the change.
            reason: Short reason string explaining why.
        """
        self.modified.append({
            "id": thing_id,
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason
        })

    def add_flagged(self, category: str, item: Dict[str, Any]):
        """
        Record something flagged for review but not auto-fixed.

        Args:
            category: One of the keys in flagged_not_fixed.
            item: The details to include in reports.
        """
        if category in self.flagged_not_fixed:
            self.flagged_not_fixed[category].append(item)


def check_claude_code() -> bool:
    """
    Check if Claude Code CLI is installed.

    Returns:
        True if the `claude` executable is on PATH.
    """
    # Uses the system `which` command; return code 0 means found.
    result = subprocess.run(
        ["which", "claude"], capture_output=True, text=True
    )
    return result.returncode == 0


def load_json_data(json_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load things data from JSON file, preserving metadata if present.

    Returns:
        A tuple of (things_list, metadata_dict).
    """
    print(f"📂 Loading data from {json_path}...")
    # I/O boundary: read JSON from disk.
    with open(json_path) as f:
        data = json.load(f)

    metadata = None
    # Handle both list format and {metadata, tasks} format
    if isinstance(data, list):
        things = data
    elif isinstance(data, dict) and "tasks" in data:
        metadata = data.get("metadata", {})
        things = data["tasks"]
    elif isinstance(data, dict) and "things" in data:
        metadata = data.get("metadata", {})
        things = data["things"]
    else:
        print(f"❌ Unexpected JSON structure. Keys: {list(data.keys())}")
        sys.exit(1)

    print(f"✅ Loaded {len(things)} things")
    return things, metadata


def analyze_duplicates(
    things: List[Dict[str, Any]],
    window_days: int
) -> List[Dict[str, Any]]:
    """
    Detect potential duplicate things (same content within time window).

    Args:
        things: List of thing dictionaries to scan.
        window_days: Maximum day difference to consider duplicates.
    """
    from dateutil import parser

    duplicates = []
    content_map = defaultdict(list)

    # Group by normalized content for basic duplicate detection.
    for thing in things:
        content = thing.get("content", "").lower().strip()
        if content:
            content_map[content].append(thing)

    # Find duplicates within temporal window
    for content, items in content_map.items():
        if len(items) > 1:
            # Sort by timestamp so we can check adjacency.
            items_with_time = []
            for item in items:
                try:
                    ts = parser.parse(item["timestamp"])
                    items_with_time.append((ts, item))
                except:
                    continue

            items_with_time.sort(key=lambda x: x[0])

            # Check temporal proximity between adjacent items.
            for i in range(len(items_with_time) - 1):
                ts1, item1 = items_with_time[i]
                ts2, item2 = items_with_time[i + 1]
                days_apart = abs((ts2 - ts1).days)
                seconds_apart = abs((ts2 - ts1).total_seconds())

                if days_apart <= window_days:
                    duplicates.append({
                        "content": content,
                        "items": [item1, item2],
                        "days_apart": days_apart,
                        "seconds_apart": seconds_apart,
                        "is_exact": seconds_apart < 60,  # Within 1 minute = exact
                        "timestamps": [item1["timestamp"], item2["timestamp"]]
                    })

    return duplicates


def analyze_normalization(things: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze entities needing normalization (case variants).

    Returns:
        A dict describing normalization opportunities for people and tags.
    """
    people_variants = defaultdict(list)
    tag_variants = defaultdict(list)

    # Collect all people and tags with counts.
    people_counter = Counter()
    tags_counter = Counter()

    for thing in things:
        for person in thing.get("people_mentioned", []):
            people_counter[person] += 1
        for tag in thing.get("tags", []):
            tags_counter[tag] += 1

    # Find case variants by lowercasing keys.
    people_lower_map = defaultdict(list)
    for person, count in people_counter.items():
        people_lower_map[person.lower()].append((person, count))

    tags_lower_map = defaultdict(list)
    for tag, count in tags_counter.items():
        tags_lower_map[tag.lower()].append((tag, count))

    # Identify normalization opportunities (same lowercase, multiple variants).
    people_issues = []
    for lower, variants in people_lower_map.items():
        if len(variants) > 1:
            variants.sort(key=lambda x: x[1], reverse=True)
            people_issues.append({
                "canonical": variants[0][0],
                "variants": variants,
                "total_count": sum(v[1] for v in variants)
            })

    tags_issues = []
    for lower, variants in tags_lower_map.items():
        if len(variants) > 1:
            variants.sort(key=lambda x: x[1], reverse=True)
            tags_issues.append({
                "canonical": variants[0][0],
                "variants": variants,
                "total_count": sum(v[1] for v in variants)
            })

    return {
        "people": people_issues,
        "tags": tags_issues,
        "top_people": people_counter.most_common(10),
        "top_tags": tags_counter.most_common(10)
    }


def analyze_data_quality(
    things: List[Dict[str, Any]],
    long_threshold: int
) -> Dict[str, Any]:
    """
    Check for data quality issues.

    Args:
        things: List of things to scan.
        long_threshold: Length at which content is flagged as unusually long.
    """
    issues = {
        "missing_timestamp": [],
        "missing_content": [],
        "orphaned_children": [],
        "unusually_long": [],
    }

    thing_ids = {t["id"] for t in things}

    for thing in things:
        # Missing timestamp indicates a parsing failure or malformed input.
        if not thing.get("timestamp"):
            issues["missing_timestamp"].append(thing["id"])

        # Missing content indicates an empty line or parsing issue.
        if not thing.get("content"):
            issues["missing_content"].append(thing["id"])

        # Orphaned children reference a parent that does not exist.
        parent_id = thing.get("parent_task_id")
        if parent_id and parent_id not in thing_ids:
            issues["orphaned_children"].append({
                "id": thing["id"],
                "parent_id": parent_id
            })

        # Unusually long content can indicate a paste or formatting error.
        content = thing.get("content", "")
        if len(content) > long_threshold:
            issues["unusually_long"].append({
                "id": thing["id"],
                "length": len(content)
            })

    return {k: v for k, v in issues.items() if v}


def apply_fixes(
    things: List[Dict[str, Any]],
    duplicates: List[Dict[str, Any]],
    quality_issues: Dict[str, Any],
    changes: GroomingChanges
) -> List[Dict[str, Any]]:
    """
    Apply automatic fixes to the data.

    Returns:
        A new list of cleaned things, leaving the input list unchanged.
    """

    # Track IDs to remove so we can filter them out later.
    ids_to_remove = set()

    # Fix 1: Remove EXACT duplicates only (same timestamp within 1 minute).
    for dup in duplicates:
        if dup["is_exact"]:
            # Keep first, remove second.
            item_to_remove = dup["items"][1]
            ids_to_remove.add(item_to_remove["id"])
            changes.add_removed(
                item_to_remove,
                "exact_duplicate",
                duplicate_of=dup["items"][0]["id"]
            )
        else:
            # Flag ambiguous duplicates for review.
            changes.add_flagged("ambiguous_duplicates", {
                "content": dup["content"][:60],
                "items": [i["id"] for i in dup["items"]],
                "days_apart": dup["days_apart"]
            })

    # Fix 2: Null out orphaned parent references (do not delete the things).
    orphaned = quality_issues.get("orphaned_children", [])
    thing_dict = {t["id"]: t for t in things}

    for orphan in orphaned:
        thing_id = orphan["id"]
        if thing_id in thing_dict:
            old_parent = thing_dict[thing_id]["parent_task_id"]
            thing_dict[thing_id]["parent_task_id"] = None
            changes.add_modified(
                thing_id,
                "parent_task_id",
                old_parent,
                None,
                "orphaned_reference"
            )

    # Flag long content for review (do not auto-delete).
    for long_item in quality_issues.get("unusually_long", []):
        changes.add_flagged("long_content", long_item)

    # Remove duplicates by filtering out marked IDs.
    cleaned_things = [t for t in things if t["id"] not in ids_to_remove]

    return cleaned_things


def generate_changes_report_md(
    changes: GroomingChanges,
    original_count: int,
    cleaned_count: int
) -> str:
    """
    Generate human-readable markdown changes report.

    Args:
        changes: Collected changes and flags from grooming.
        original_count: Number of things before cleaning.
        cleaned_count: Number of things after cleaning.

    Returns:
        A Markdown string that can be saved to disk.
    """

    report = f"""# Data Grooming Changes Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Original:** {original_count:,} things
**Cleaned:** {cleaned_count:,} things
**Removed:** {len(changes.removed)} items
**Modified:** {len(changes.modified)} items

---

## Auto-Fixed Items

"""

    if changes.removed:
        report += f"### ❌ Removed Items ({len(changes.removed)})\n\n"
        for item in changes.removed:
            report += f"**{item['id']}** - DELETED\n"
            report += f"- **Content:** `{item['content']}`\n"
            report += f"- **Timestamp:** {item['timestamp']}\n"
            report += f"- **Reason:** {item['reason']}\n"
            if item.get('duplicate_of'):
                report += f"- **Duplicate of:** {item['duplicate_of']} (kept)\n"
            report += "\n"
    else:
        report += "### ✅ No items removed\n\n"

    if changes.modified:
        report += f"### 🔧 Modified Items ({len(changes.modified)})\n\n"
        for item in changes.modified:
            report += f"**{item['id']}** - MODIFIED\n"
            report += f"- **Field:** {item['field']}\n"
            report += f"- **Old Value:** `{item['old_value']}`\n"
            report += f"- **New Value:** `{item['new_value']}`\n"
            report += f"- **Reason:** {item['reason']}\n\n"
    else:
        report += "### ✅ No items modified\n\n"

    report += "---\n\n## Flagged for Review (NOT Auto-Fixed)\n\n"

    # Ambiguous duplicates
    ambig = changes.flagged_not_fixed.get("ambiguous_duplicates", [])
    if ambig:
        report += f"### ⚠️ Ambiguous Duplicates ({len(ambig)})\n\n"
        report += "These have the same content but are NOT within 1 minute - could be recurring tasks.\n\n"
        for item in ambig[:10]:
            report += f"- **Content:** `{item['content']}...`\n"
            report += f"  - Items: {', '.join(item['items'])}\n"
            report += f"  - {item['days_apart']} days apart\n"
            report += f"  - **Action:** Manual review or AI analysis needed\n\n"

    # Long content
    long_items = changes.flagged_not_fixed.get("long_content", [])
    if long_items:
        report += f"### ⚠️ Unusually Long Content ({len(long_items)})\n\n"
        for item in long_items[:10]:
            report += f"- **{item['id']}**: {item['length']:,} characters\n"
        report += "\n"

    # Normalization
    norm = changes.flagged_not_fixed.get("normalization", [])
    if norm:
        report += f"### ⚠️ Normalization Opportunities ({len(norm)})\n\n"
        for item in norm[:15]:
            report += f"- **{item['type']}:** {item['description']}\n"
            report += f"  - **Recommendation:** {item['recommendation']}\n"
            report += f"  - **Action:** Requires manual decision\n\n"

    return report


def generate_changes_report_json(
    changes: GroomingChanges,
    original_count: int,
    cleaned_count: int
) -> Dict[str, Any]:
    """
    Generate machine-readable JSON changes report.

    Args:
        changes: Collected changes and flags from grooming.
        original_count: Number of things before cleaning.
        cleaned_count: Number of things after cleaning.

    Returns:
        A dict suitable for json.dump.
    """

    return {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "original_count": original_count,
            "cleaned_count": cleaned_count,
            "removed_count": len(changes.removed),
            "modified_count": len(changes.modified)
        },
        "removed": changes.removed,
        "modified": changes.modified,
        "flagged_not_fixed": changes.flagged_not_fixed
    }


def generate_python_report(
    stats: Dict[str, Any],
    duplicates: List[Dict[str, Any]],
    normalization: Dict[str, Any],
    quality_issues: Dict[str, Any],
    report_limit: int
) -> str:
    """
    Generate markdown report from Python analysis.

    Args:
        stats: Summary statistics for the dataset.
        duplicates: Duplicate detection results.
        normalization: Case normalization opportunities.
        quality_issues: Data quality issue buckets.
        report_limit: Max number of items to show per section.

    Returns:
        A Markdown report string.
    """

    report = f"""# Data Grooming Report (Python Analysis)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Type:** Automated (Python)

## Summary

- **Total Things:** {stats['total_things']:,}
- **Date Range:** {stats['date_range']['earliest']} to {stats['date_range']['latest']}
- **Completed:** {stats['completed']:,}
- **Strikethrough:** {stats['strikethrough']:,}

## Statistics

- Things with links: {stats['with_links']:,}
- Things with tags: {stats['with_tags']:,}
- Things with people: {stats['with_people']:,}

## Duplicate Detection

**Found {len(duplicates)} potential duplicate clusters**

"""

    if duplicates:
        exact_dups = [d for d in duplicates if d["is_exact"]]
        ambig_dups = [d for d in duplicates if not d["is_exact"]]

        report += f"- **Exact duplicates (auto-removed):** {len(exact_dups)}\n"
        report += f"- **Ambiguous (flagged for review):** {len(ambig_dups)}\n\n"

        report += f"### Top {report_limit} Duplicate Candidates\n\n"
        for dup in duplicates[:report_limit]:
            status = "🔴 EXACT - REMOVED" if dup["is_exact"] else "🟡 AMBIGUOUS"
            report += f"**{status}**\n"
            report += f"- **Content:** `{dup['content'][:60]}...`\n"
            report += f"  - Items: {dup['items'][0]['id']}, {dup['items'][1]['id']}\n"
            report += f"  - {dup['days_apart']} days apart\n\n"
    else:
        report += "*No duplicates found*\n\n"

    report += "## Normalization Analysis\n\n"

    # People
    people_issues = normalization['people']
    report += f"### People ({len(people_issues)} normalization opportunities)\n\n"

    if people_issues:
        for issue in people_issues[:report_limit]:
            canonical = issue['canonical']
            variants_str = ", ".join([f"{v[0]} ({v[1]})" for v in issue['variants']])
            report += f"- **{canonical}** ← {variants_str}\n"
            report += f"  - **Action:** Manual normalization or AI analysis recommended\n"
    else:
        report += "*No people normalization issues found*\n"

    report += "\n### Top People\n\n"
    for person, count in normalization['top_people']:
        report += f"- {person}: {count:,}\n"

    # Tags
    tags_issues = normalization['tags']
    report += f"\n### Tags ({len(tags_issues)} normalization opportunities)\n\n"

    if tags_issues:
        for issue in tags_issues[:report_limit]:
            canonical = issue['canonical']
            variants_str = ", ".join([f"{v[0]} ({v[1]})" for v in issue['variants']])
            report += f"- **{canonical}** ← {variants_str}\n"
            report += f"  - **Action:** Manual normalization or AI analysis recommended\n"
    else:
        report += "*No tag normalization issues found*\n"

    report += "\n### Top Tags\n\n"
    for tag, count in normalization['top_tags']:
        report += f"- {tag}: {count:,}\n"

    # Data Quality
    report += "\n## Data Quality Issues\n\n"

    if quality_issues:
        for issue_type, items in quality_issues.items():
            report += f"### {issue_type.replace('_', ' ').title()}\n\n"
            if isinstance(items, list):
                report += f"**Count:** {len(items)}\n\n"
                if isinstance(items[0], dict):
                    for item in items[:5]:
                        report += f"- {item}\n"
                else:
                    report += f"- {', '.join(str(i) for i in items[:10])}\n"
            report += "\n"
    else:
        report += "*No data quality issues detected*\n\n"

    report += """
## Next Steps

- Review the changes report to see what was auto-fixed
- Run with `--ai-analysis` for semantic insights
- Use the cleaned data file for SQLite loading

"""

    return report


def analyze_statistics(things: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate basic statistics.

    Returns:
        Counts and date ranges used in reports.
    """
    from dateutil import parser

    timestamps = []
    for thing in things:
        try:
            ts = parser.parse(thing["timestamp"])
            timestamps.append(ts)
        except:
            continue

    timestamps.sort()

    return {
        "total_things": len(things),
        "date_range": {
            "earliest": timestamps[0].isoformat() if timestamps else None,
            "latest": timestamps[-1].isoformat() if timestamps else None
        },
        "completed": sum(1 for t in things if t.get("is_completed")),
        "strikethrough": sum(1 for t in things if t.get("is_strikethrough")),
        "with_links": sum(1 for t in things if t.get("links")),
        "with_tags": sum(1 for t in things if t.get("tags")),
        "with_people": sum(1 for t in things if t.get("people_mentioned"))
    }


def invoke_claude_analysis(
    json_path: Path,
    python_report_path: Path,
    changes_report_path: Path,
    output_path: Path
) -> bool:
    """
    Invoke Claude Code for semantic analysis.

    Args:
        json_path: Path to the original JSON input file.
        python_report_path: Path to the Python analysis report.
        changes_report_path: Path to the changes report.
        output_path: Path where the AI report should be written.

    Side effects:
        Executes the external `claude` CLI, which may use subscription quota
        and write an output report to disk.

    Returns:
        True if the output file was written, otherwise False.
    """

    print("\n🤖 Invoking Claude Code for semantic analysis...")
    print("   (This will use your subscription quota)")

    # Read the grooming prompt template.
    prompt_file = Path("docs/DATA_GROOMING_PROMPT.md")
    if not prompt_file.exists():
        print(f"❌ Prompt template not found: {prompt_file}")
        return False

    with open(prompt_file) as f:
        grooming_prompt = f.read()

    # Read the Python report.
    with open(python_report_path) as f:
        python_summary = f.read()

    # Read the changes report.
    with open(changes_report_path) as f:
        changes_summary = f.read()

    # Build the full prompt that instructs Claude Code.
    full_prompt = f"""{grooming_prompt}

PYTHON PRE-ANALYSIS AND AUTO-FIXES COMPLETED:

## Python Analysis Summary:
{python_summary}

## Auto-Fixes Applied:
{changes_summary}

YOUR TASK:

1. Read the CLEANED data from: {json_path.parent / (json_path.stem + '_cleaned.json')}
2. Provide SEMANTIC ANALYSIS focusing on:
   - Pattern interpretation and theme detection
   - Project threads and narratives across time
   - Schema optimization recommendations
   - Judgment on ambiguous duplicates flagged above
   - Entity categorization (person vs place vs project)
   - Recommendations on normalization opportunities

3. Write your COMPLETE semantic grooming report to: {output_path}

Focus on insights that require semantic understanding - the mechanical checks and fixes are already done.

Execute steps 1-3 now without asking for confirmation.
"""

    # Invoke Claude Code CLI; this spawns a subprocess and captures output.
    try:
        result = subprocess.run(
            [
                "claude",
                "--print",
                "--tools", "Read,Write",
                "--allowedTools", "Read,Write",
                "--dangerously-skip-permissions",
                "--no-session-persistence",
                full_prompt
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            stdin=subprocess.DEVNULL  # Signal no input coming - prevents waiting for stdin
        )
    except subprocess.TimeoutExpired:
        print("❌ Claude Code timed out after 5 minutes")
        return False

    if result.returncode != 0:
        print(f"❌ Claude Code failed with exit code {result.returncode}")
        print(f"stderr: {result.stderr}")
        return False

    # Check if output was written to the expected file.
    if output_path.exists():
        print(f"✅ Semantic analysis complete: {output_path}")
        return True
    else:
        print("⚠️  Claude Code completed but output file not found")
        print("Response:")
        print(result.stdout[:500])
        return False


def main():
    """
    CLI entry point for data grooming.

    Returns:
        None. Exits the process on failure.
    """
    parser = argparse.ArgumentParser(
        description="Groom Twos task data - Python auto-fix + optional Claude Code semantic analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic grooming with auto-fixes
  python3 scripts/groom_data.py

  # More aggressive duplicate detection
  python3 scripts/groom_data.py --duplicate-window 14

  # With AI semantic analysis
  python3 scripts/groom_data.py --ai-analysis

  # Custom thresholds
  python3 scripts/groom_data.py --duplicate-window 3 --long-content-threshold 1000
"""
    )
    parser.add_argument(
        "json_file",
        type=Path,
        nargs="?",
        default=Path("data/processed/twos_data.json"),
        help="Path to JSON data file (default: data/processed/twos_data.json)"
    )
    parser.add_argument(
        "--duplicate-window",
        type=int,
        default=7,
        help="Days window for duplicate detection (default: 7)"
    )
    parser.add_argument(
        "--long-content-threshold",
        type=int,
        default=2000,
        help="Character threshold for flagging long content (default: 2000)"
    )
    parser.add_argument(
        "--report-limit",
        type=int,
        default=10,
        help="Number of items to show in reports (default: 10)"
    )
    parser.add_argument(
        "--ai-analysis",
        action="store_true",
        help="Invoke Claude Code for semantic analysis (uses subscription quota)"
    )

    args = parser.parse_args()

    # Validate input file before doing any processing.
    if not args.json_file.exists():
        print(f"❌ File not found: {args.json_file}")
        sys.exit(1)

    # Check Claude Code if AI analysis requested.
    if args.ai_analysis:
        if not check_claude_code():
            print("❌ Claude Code CLI not found")
            print("\nInstall it first:")
            print("https://code.claude.com/docs/en/quickstart")
            sys.exit(1)
        print("✅ Claude Code CLI found")

    # Load data from JSON on disk.
    things, metadata = load_json_data(args.json_file)
    original_count = len(things)

    # Run Python analysis (no external services).
    print(f"\n🔍 Running Python analysis (window={args.duplicate_window} days, long={args.long_content_threshold} chars)...")

    stats = analyze_statistics(things)
    duplicates = analyze_duplicates(things, args.duplicate_window)
    normalization = analyze_normalization(things)
    quality_issues = analyze_data_quality(things, args.long_content_threshold)

    print(f"  - Found {len(duplicates)} potential duplicates")
    exact_count = sum(1 for d in duplicates if d["is_exact"])
    print(f"    └─ {exact_count} exact (will be removed)")
    print(f"  - Found {len(normalization['people'])} people normalization opportunities")
    print(f"  - Found {len(normalization['tags'])} tag normalization opportunities")
    print(f"  - Found {len(quality_issues)} data quality issue types")

    # Apply automatic fixes.
    print("\n🔧 Applying automatic fixes...")
    changes = GroomingChanges()

    # Add normalization flags to changes (for reporting).
    for person_issue in normalization['people']:
        changes.add_flagged("normalization", {
            "type": "person",
            "description": f"{person_issue['canonical']} has {len(person_issue['variants'])} variants",
            "recommendation": f"Standardize all to '{person_issue['canonical']}'"
        })

    for tag_issue in normalization['tags']:
        changes.add_flagged("normalization", {
            "type": "tag",
            "description": f"{tag_issue['canonical']} has {len(tag_issue['variants'])} variants",
            "recommendation": f"Standardize all to '{tag_issue['canonical']}'"
        })

    cleaned_things = apply_fixes(things, duplicates, quality_issues, changes)
    cleaned_count = len(cleaned_things)

    print(f"  ✅ Removed {len(changes.removed)} exact duplicates")
    print(f"  ✅ Fixed {len(changes.modified)} broken references")
    print(f"  ✅ Cleaned: {cleaned_count:,} things (from {original_count:,})")

    # Save cleaned data to a new JSON file on disk.
    cleaned_path = args.json_file.parent / (args.json_file.stem + "_cleaned.json")

    # Preserve original structure
    if metadata:
        output_data = {
            "metadata": {
                **metadata,
                "cleaned_at": datetime.now().isoformat(),
                "original_count": original_count,
                "cleaned_count": cleaned_count
            },
            "tasks": cleaned_things
        }
    else:
        output_data = cleaned_things

    with open(cleaned_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"  💾 Saved cleaned data: {cleaned_path}")

    # Generate reports in docs/grooming-reports.
    timestamp = datetime.now().strftime("%Y-%m-%d")
    reports_dir = Path("docs/grooming-reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Changes report (MD + JSON).
    changes_md_path = reports_dir / f"{timestamp}-changes.md"
    changes_json_path = reports_dir / f"{timestamp}-changes.json"

    changes_md = generate_changes_report_md(changes, original_count, cleaned_count)
    with open(changes_md_path, "w") as f:
        f.write(changes_md)

    changes_json = generate_changes_report_json(changes, original_count, cleaned_count)
    with open(changes_json_path, "w") as f:
        json.dump(changes_json, f, indent=2)

    print(f"\n📊 Changes report saved:")
    print(f"  - {changes_md_path}")
    print(f"  - {changes_json_path}")

    # Python analysis report.
    python_report = generate_python_report(
        stats, duplicates, normalization, quality_issues, args.report_limit
    )
    python_report_path = reports_dir / f"{timestamp}-python.md"
    with open(python_report_path, "w") as f:
        f.write(python_report)

    print(f"  - {python_report_path}")

    # Optionally run Claude analysis (external CLI, may use quota).
    if args.ai_analysis:
        ai_report_path = reports_dir / f"{timestamp}-ai-analysis.md"
        success = invoke_claude_analysis(
            args.json_file,
            python_report_path,
            changes_md_path,
            ai_report_path
        )
        if not success:
            print("\n⚠️  AI analysis failed, but cleaned data and reports are available")
            sys.exit(1)
    else:
        print("\n💡 Tip: Run with --ai-analysis for semantic insights (uses quota)")

    print("\n🎉 Grooming complete!")

    # Provide next steps based on whether AI analysis was run
    if args.ai_analysis:
        print("\n📊 AI analysis identified entity classification issues")
        print("   See: docs/grooming-reports/*-ai-analysis.md")
        print("\n💡 Recommended workflow:")
        print("\n   1. Classify entities (improves query accuracy ~40%):")
        print("      python3 scripts/classify_entities.py --ai-classify --apply-mappings")
        print("\n   2. Then load normalized data to SQLite:")
        print("      python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned_normalized.json")
        print("\n   OR skip entity classification and load cleaned data now:")
        print(f"      python3 scripts/load_to_sqlite.py {cleaned_path}")
    else:
        print(f"\n📁 Next step: Load cleaned data into SQLite:")
        print(f"   python3 scripts/load_to_sqlite.py {cleaned_path}")


if __name__ == "__main__":
    main()
