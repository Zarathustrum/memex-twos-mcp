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
from typing import Any, Dict, List, Optional, Tuple

# Import LLM utilities
try:
    from llm_utils import invoke_llm, check_llm_available
except ImportError:
    invoke_llm = None
    check_llm_available = None


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate using character count.

    Rule of thumb: ~3 characters per token for English prose.
    Conservative for mixed Markdown/URLs and safer for prompt budgeting.

    Args:
        text: String to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 3


def filter_things_by_months(
    things: List[Dict[str, Any]], months_back: int
) -> List[Dict[str, Any]]:
    """
    Filter things to keep only items within the last N months.

    Notes:
        - Uses calendar-aware month subtraction for accurate boundaries.
        - Items without parseable timestamps are kept to avoid accidental loss.
    """
    if months_back <= 0:
        return things

    from dateutil import parser
    from dateutil.relativedelta import relativedelta

    cutoff_date = datetime.now() - relativedelta(months=months_back)
    filtered = []

    for thing in things:
        timestamp = thing.get("timestamp")
        if not timestamp:
            filtered.append(thing)
            continue

        try:
            ts = parser.parse(timestamp)
        except Exception:
            filtered.append(thing)
            continue

        if ts >= cutoff_date:
            filtered.append(thing)

    return filtered


def filter_changes_for_llm(
    changes: "GroomingChanges",
    things_by_id: Dict[str, Dict[str, Any]],
    months_back: int,
) -> "GroomingChanges":
    """
    Filter change summaries to only include items in the LLM time window.

    Notes:
        - Items without parseable timestamps are kept.
        - Normalization flags are excluded and should be re-derived separately.
    """
    if months_back <= 0:
        return changes

    from dateutil import parser
    from dateutil.relativedelta import relativedelta

    cutoff_date = datetime.now() - relativedelta(months=months_back)

    def in_window(timestamp: Optional[str]) -> bool:
        if not timestamp:
            return True
        try:
            ts = parser.parse(timestamp)
        except Exception:
            return True
        return ts >= cutoff_date

    filtered = GroomingChanges()

    for removed in changes.removed:
        if in_window(removed.get("timestamp")):
            filtered.removed.append(removed)

    for modified in changes.modified:
        ts = things_by_id.get(modified["id"], {}).get("timestamp")
        if in_window(ts):
            filtered.modified.append(modified)

    for item in changes.flagged_not_fixed.get("ambiguous_duplicates", []):
        ids = item.get("items", [])
        if not ids:
            filtered.flagged_not_fixed["ambiguous_duplicates"].append(item)
            continue
        if any(in_window(things_by_id.get(i, {}).get("timestamp")) for i in ids):
            filtered.flagged_not_fixed["ambiguous_duplicates"].append(item)

    for item in changes.flagged_not_fixed.get("long_content", []):
        thing_id = item.get("id")
        ts = things_by_id.get(thing_id, {}).get("timestamp")
        if in_window(ts):
            filtered.flagged_not_fixed["long_content"].append(item)

    return filtered


class GroomingChanges:
    """Track all changes made during grooming."""

    def __init__(self):
        """Initialize collections used to report changes and flags."""
        self.removed = []
        self.modified = []
        self.flagged_not_fixed = {
            "normalization": [],
            "ambiguous_duplicates": [],
            "long_content": [],
        }

    def add_removed(self, thing: Dict[str, Any], reason: str, duplicate_of: str = None):
        """
        Record a removed item.

        Args:
            thing: The original thing dictionary being removed.
            reason: Short reason string (e.g., "exact_duplicate").
            duplicate_of: Optional ID of the retained item.
        """
        self.removed.append(
            {
                "id": thing["id"],
                "content": thing.get("content", "")[:100],
                "timestamp": thing.get("timestamp"),
                "reason": reason,
                "duplicate_of": duplicate_of,
            }
        )

    def add_modified(
        self, thing_id: str, field: str, old_value: Any, new_value: Any, reason: str
    ):
        """
        Record a modification to a single field.

        Args:
            thing_id: ID of the thing being changed.
            field: The field name being updated.
            old_value: Previous value for audit purposes.
            new_value: New value after the change.
            reason: Short reason string explaining why.
        """
        self.modified.append(
            {
                "id": thing_id,
                "field": field,
                "old_value": old_value,
                "new_value": new_value,
                "reason": reason,
            }
        )

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
    Check if any LLM provider is available.

    Returns:
        True if an LLM provider (LM Studio, Claude CLI, Anthropic API, etc.) is available.
    """
    if check_llm_available:
        return check_llm_available()
    else:
        # Fallback: check if claude CLI exists
        result = subprocess.run(["which", "claude"], capture_output=True, text=True)
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
    things: List[Dict[str, Any]], window_days: int
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
                except Exception:
                    continue

            items_with_time.sort(key=lambda x: x[0])

            # Check temporal proximity between adjacent items.
            for i in range(len(items_with_time) - 1):
                ts1, item1 = items_with_time[i]
                ts2, item2 = items_with_time[i + 1]
                days_apart = abs((ts2 - ts1).days)
                seconds_apart = abs((ts2 - ts1).total_seconds())

                if days_apart <= window_days:
                    duplicates.append(
                        {
                            "content": content,
                            "items": [item1, item2],
                            "days_apart": days_apart,
                            "seconds_apart": seconds_apart,
                            "is_exact": seconds_apart < 60,  # Within 1 minute = exact
                            "timestamps": [item1["timestamp"], item2["timestamp"]],
                        }
                    )

    return duplicates


def analyze_normalization(things: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze entities needing normalization (case variants).

    Returns:
        A dict describing normalization opportunities for people and tags.
    """
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
            people_issues.append(
                {
                    "canonical": variants[0][0],
                    "variants": variants,
                    "total_count": sum(v[1] for v in variants),
                }
            )

    tags_issues = []
    for lower, variants in tags_lower_map.items():
        if len(variants) > 1:
            variants.sort(key=lambda x: x[1], reverse=True)
            tags_issues.append(
                {
                    "canonical": variants[0][0],
                    "variants": variants,
                    "total_count": sum(v[1] for v in variants),
                }
            )

    return {
        "people": people_issues,
        "tags": tags_issues,
        "top_people": people_counter.most_common(10),
        "top_tags": tags_counter.most_common(10),
    }


def analyze_data_quality(
    things: List[Dict[str, Any]], long_threshold: int
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
            issues["orphaned_children"].append(
                {"id": thing["id"], "parent_id": parent_id}
            )

        # Unusually long content can indicate a paste or formatting error.
        content = thing.get("content", "")
        if len(content) > long_threshold:
            issues["unusually_long"].append({"id": thing["id"], "length": len(content)})

    return {k: v for k, v in issues.items() if v}


def apply_fixes(
    things: List[Dict[str, Any]],
    duplicates: List[Dict[str, Any]],
    quality_issues: Dict[str, Any],
    changes: GroomingChanges,
) -> List[Dict[str, Any]]:
    """
    Apply automatic fixes to the data in three passes.

    Fix strategy:
    1. Remove exact duplicates (same content + timestamp within 1 min)
    2. Null orphaned parent references (parent_task_id points to non-existent item)
    3. Re-check for NEW orphans created by duplicate removal

    Why Fix 3 is necessary:
    - Fix 1 removes duplicate items from the dataset
    - If a removed item was someone's parent, that child becomes orphaned
    - Fix 2 only catches orphans in the ORIGINAL dataset
    - Fix 3 catches orphans CREATED by the grooming process itself
    - Without Fix 3, we'd create new data integrity issues while fixing old ones

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
                item_to_remove, "exact_duplicate", duplicate_of=dup["items"][0]["id"]
            )
        else:
            # Flag ambiguous duplicates for review.
            changes.add_flagged(
                "ambiguous_duplicates",
                {
                    "content": dup["content"][:60],
                    "items": [i["id"] for i in dup["items"]],
                    "days_apart": dup["days_apart"],
                },
            )

    # Fix 2: Null out orphaned parent references (do not delete the things).
    orphaned = quality_issues.get("orphaned_children", [])
    thing_dict = {t["id"]: t for t in things}

    for orphan in orphaned:
        thing_id = orphan["id"]
        if thing_id in thing_dict:
            old_parent = thing_dict[thing_id]["parent_task_id"]
            thing_dict[thing_id]["parent_task_id"] = None
            changes.add_modified(
                thing_id, "parent_task_id", old_parent, None, "orphaned_reference"
            )

    # Flag long content for review (do not auto-delete).
    for long_item in quality_issues.get("unusually_long", []):
        changes.add_flagged("long_content", long_item)

    # Remove duplicates by filtering out marked IDs.
    cleaned_things = [t for t in things if t["id"] not in ids_to_remove]

    # Fix 3: Re-check for newly orphaned children after duplicate removal.
    # If a removed thing was a parent, its children are now orphaned.
    existing_ids = {t["id"] for t in cleaned_things}
    for thing in cleaned_things:
        parent_id = thing.get("parent_task_id")
        if parent_id and parent_id not in existing_ids:
            # Parent was removed, null out the reference
            old_parent = thing["parent_task_id"]
            thing["parent_task_id"] = None
            changes.add_modified(
                thing["id"],
                "parent_task_id",
                old_parent,
                None,
                "parent_removed_during_grooming",
            )

    return cleaned_things


def generate_changes_report_md(
    changes: GroomingChanges, original_count: int, cleaned_count: int
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
            if item.get("duplicate_of"):
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
            report += "  - **Action:** Manual review or AI analysis needed\n\n"

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
            report += "  - **Action:** Requires manual decision\n\n"

    return report


def generate_changes_report_json(
    changes: GroomingChanges, original_count: int, cleaned_count: int
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
            "modified_count": len(changes.modified),
        },
        "removed": changes.removed,
        "modified": changes.modified,
        "flagged_not_fixed": changes.flagged_not_fixed,
    }


def generate_compressed_prose_report(
    stats: Dict[str, Any],
    duplicates: List[Dict[str, Any]],
    normalization: Dict[str, Any],
    quality_issues: Dict[str, Any],
    token_budget: int = 12000,
) -> str:
    """
    Generate compressed prose-style report optimized for LLM context efficiency.

    Uses narrative format with clustering to maximize information density while
    staying within token budget. Designed for LM Studio with limited context.

    Args:
        stats: Summary statistics for the dataset.
        duplicates: Duplicate detection results.
        normalization: Case normalization opportunities.
        quality_issues: Data quality issue buckets.
        token_budget: Maximum tokens for this report (default 12K)

    Returns:
        Compressed prose report string, guaranteed under token_budget
    """
    sections = []

    # Executive summary (always included, ~200-300 tokens)
    total_issues = len(duplicates) + sum(len(normalization.get(k, [])) for k in ['people', 'tags'])
    exec_summary = f"""## Executive Summary

Analysis of {stats['total_things']:,} tasks ({stats['date_range']['earliest']} to {stats['date_range']['latest']}) identified {total_issues} optimization opportunities. Primary areas: duplicate consolidation ({len(duplicates)} clusters), entity normalization ({len(normalization.get('people', []))} people, {len(normalization.get('tags', []))} tags), and data quality validation."""
    sections.append(exec_summary)

    # Duplicate analysis with clustering (~2000-4000 tokens depending on patterns)
    dup_section = _compress_duplicates_prose(duplicates, stats, token_budget=4000)
    sections.append(dup_section)

    # Normalization with clustering (~2000-3000 tokens)
    norm_section = _compress_normalization_prose(normalization, token_budget=3000)
    sections.append(norm_section)

    # Quality issues (brief, ~500-1000 tokens)
    quality_section = _compress_quality_prose(quality_issues, token_budget=1000)
    sections.append(quality_section)

    # Join and check budget
    full_report = "\n\n".join(sections)
    estimated_tokens = estimate_tokens(full_report)

    # If over budget, apply aggressive truncation
    if estimated_tokens > token_budget:
        # Re-generate with tighter budgets
        dup_section = _compress_duplicates_prose(duplicates, stats, token_budget=2500)
        norm_section = _compress_normalization_prose(normalization, token_budget=2000)
        quality_section = _compress_quality_prose(quality_issues, token_budget=500)
        full_report = "\n\n".join([exec_summary, dup_section, norm_section, quality_section])

    return full_report


def _compress_duplicates_prose(
    duplicates: List[Dict[str, Any]],
    stats: Dict[str, Any],
    token_budget: int
) -> str:
    """Generate compressed prose for duplicate analysis."""
    if not duplicates:
        return "## Duplicate Analysis\n\nNo duplicates detected."

    # Cluster by content pattern
    clusters = {}
    for dup in duplicates:
        # Extract pattern (first 30 chars as key)
        pattern = dup['content'][:30]
        if pattern not in clusters:
            clusters[pattern] = {
                'content': dup['content'],
                'count': 0,
                'exact': 0,
                'ambiguous': 0,
                'examples': []
            }
        clusters[pattern]['count'] += 1
        if dup['is_exact']:
            clusters[pattern]['exact'] += 1
        else:
            clusters[pattern]['ambiguous'] += 1
        if len(clusters[pattern]['examples']) < 2:
            clusters[pattern]['examples'].append(dup)

    # Sort by frequency
    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]['count'], reverse=True)

    # Build prose narrative
    exact_total = sum(c['exact'] for _, c in sorted_clusters)
    ambig_total = sum(c['ambiguous'] for _, c in sorted_clusters)

    prose = f"""## Duplicate Analysis ({len(duplicates)} items, {len(sorted_clusters)} patterns)

Duplicate detection identified {len(duplicates)} potential duplicates across {len(sorted_clusters)} content patterns. Analysis shows {exact_total} exact duplicates (auto-removed) and {ambig_total} ambiguous cases requiring manual review."""

    # Show top patterns in detail
    detail_count = min(5, len(sorted_clusters))
    if sorted_clusters:
        prose += "\n\n**High-frequency patterns:**\n"
        for i, (pattern, cluster) in enumerate(sorted_clusters[:detail_count], 1):
            content_preview = cluster['content'][:50]
            if cluster['examples']:
                ex = cluster['examples'][0]
                day_info = f"{ex['days_apart']} days apart" if ex['days_apart'] > 0 else "same day"
                ids = f"{ex['items'][0]['id']}/{ex['items'][1]['id']}"
            else:
                day_info = "timing varies"
                ids = "multiple"

            status = "exact duplicates" if cluster['exact'] > cluster['ambiguous'] else "ambiguous timing"
            prose += f"\n{i}. \"{content_preview}...\" - {cluster['count']} instances, {status} ({ids}, {day_info})"

    # Summarize remaining clusters
    if len(sorted_clusters) > detail_count:
        remaining = len(sorted_clusters) - detail_count
        remaining_count = sum(c['count'] for _, c in sorted_clusters[detail_count:])
        prose += f"\n\n**Remaining {remaining} patterns** ({remaining_count} items): Lower-frequency duplicates following similar exact/ambiguous distribution."

    # Check token budget and truncate if needed
    if estimate_tokens(prose) > token_budget:
        # Keep only top 3 patterns
        prose_lines = prose.split('\n')
        truncated = '\n'.join(prose_lines[:10])  # Keep header + top 3 patterns
        truncated += f"\n\n*(Truncated to fit budget - {len(sorted_clusters) - 3} additional patterns summarized)*"
        return truncated

    return prose


def _compress_normalization_prose(
    normalization: Dict[str, Any],
    token_budget: int
) -> str:
    """Generate compressed prose for normalization analysis."""
    people_issues = normalization.get("people", [])
    tags_issues = normalization.get("tags", [])
    top_people = normalization.get("top_people", [])[:5]
    top_tags = normalization.get("top_tags", [])[:5]

    if not people_issues and not tags_issues:
        return "## Normalization Analysis\n\nNo normalization opportunities identified."

    prose = f"""## Normalization Analysis ({len(people_issues)} people, {len(tags_issues)} tags)

Entity normalization scan identified {len(people_issues) + len(tags_issues)} opportunities for case/variant consolidation."""

    # People
    if people_issues:
        prose += "\n\n**People normalization:**"
        for i, issue in enumerate(people_issues[:3], 1):
            canonical = issue["canonical"]
            variant_summary = ", ".join([f'"{v[0]}" ({v[1]})' for v in issue["variants"][:2]])
            total_variants = len(issue["variants"])
            if total_variants > 2:
                variant_summary += f" +{total_variants - 2} more"
            prose += f"\n{i}. {canonical} ← {variant_summary}"

        if len(people_issues) > 3:
            prose += f"\n\n*({len(people_issues) - 3} additional people patterns similar)*"

        # Top people context
        if top_people:
            top_names = ", ".join([f"{p[0]} ({p[1]})" for p in top_people[:3]])
            prose += f"\n\nMost frequent: {top_names}"

    # Tags
    if tags_issues:
        prose += "\n\n**Tag normalization:**"
        for i, issue in enumerate(tags_issues[:3], 1):
            canonical = issue["canonical"]
            variant_summary = ", ".join([f'#{v[0]}# ({v[1]})' for v in issue["variants"][:2]])
            total_variants = len(issue["variants"])
            if total_variants > 2:
                variant_summary += f" +{total_variants - 2} more"
            prose += f"\n{i}. #{canonical}# ← {variant_summary}"

        if len(tags_issues) > 3:
            prose += f"\n\n*({len(tags_issues) - 3} additional tag patterns similar)*"

        # Top tags context
        if top_tags:
            top_tag_names = ", ".join([f"#{t[0]}# ({t[1]})" for t in top_tags[:3]])
            prose += f"\n\nMost frequent: {top_tag_names}"

    # Check budget
    if estimate_tokens(prose) > token_budget:
        # Reduce to top 2 of each
        prose_lines = prose.split('\n')[:15]
        return '\n'.join(prose_lines) + "\n\n*(Truncated)*"

    return prose


def _compress_quality_prose(
    quality_issues: Dict[str, Any],
    token_budget: int
) -> str:
    """Generate compressed prose for quality issues."""
    if not quality_issues:
        return "## Data Quality\n\nNo critical quality issues detected."

    total_issues = sum(len(items) if isinstance(items, list) else 1 for items in quality_issues.values())

    prose = f"""## Data Quality ({total_issues} issues)

Quality validation identified {total_issues} items requiring attention across {len(quality_issues)} categories."""

    # Summarize each issue type
    for issue_type, items in list(quality_issues.items())[:3]:
        issue_name = issue_type.replace('_', ' ').title()
        if isinstance(items, list):
            count = len(items)
            prose += f"\n\n**{issue_name}:** {count} instances"
            # Show first example if available
            if items and isinstance(items[0], dict):
                example = str(items[0])[:60]
                prose += f" (e.g., {example}...)"
        else:
            prose += f"\n\n**{issue_name}:** {items}"

    if len(quality_issues) > 3:
        prose += f"\n\n*({len(quality_issues) - 3} additional quality categories identified)*"

    return prose


def _compress_changes_summary(
    changes: GroomingChanges,
    token_budget: int = 3000
) -> str:
    """Generate compressed summary of auto-fixes applied."""
    total_removed = len(changes.removed)
    total_modified = len(changes.modified)
    total_flagged = sum(len(items) for items in changes.flagged_not_fixed.values())

    if total_removed == 0 and total_modified == 0 and total_flagged == 0:
        return "No auto-fixes required - data is clean."

    prose = f"""Auto-fix summary: {total_removed} items removed, {total_modified} items modified, {total_flagged} items flagged for manual review."""

    # Removed items (cluster by reason)
    if total_removed > 0:
        removal_reasons = {}
        for item in changes.removed:
            reason = item.get('reason', 'unknown')
            if reason not in removal_reasons:
                removal_reasons[reason] = 0
            removal_reasons[reason] += 1

        prose += "\n\n**Removals:**"
        for reason, count in sorted(removal_reasons.items(), key=lambda x: x[1], reverse=True):
            reason_label = reason.replace('_', ' ').title()
            prose += f" {count} {reason_label},"
        prose = prose.rstrip(',') + "."

    # Modified items (cluster by type)
    if total_modified > 0 and len(changes.modified) <= 5:
        prose += f"\n\n**Modifications:** {total_modified} items (broken references fixed, normalization applied)."
    elif total_modified > 0:
        prose += f"\n\n**Modifications:** {total_modified} items updated."

    # Flagged items
    if total_flagged > 0:
        prose += f"\n\n**Flagged for review:** {total_flagged} items across {len(changes.flagged_not_fixed)} categories require manual judgment."

    # Check budget
    if estimate_tokens(prose) > token_budget:
        # Ultra-compressed version
        return f"Auto-fixes: {total_removed} removed, {total_modified} modified, {total_flagged} flagged for review."

    return prose


def generate_python_report(
    stats: Dict[str, Any],
    duplicates: List[Dict[str, Any]],
    normalization: Dict[str, Any],
    quality_issues: Dict[str, Any],
    report_limit: int,
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
    people_issues = normalization["people"]
    report += f"### People ({len(people_issues)} normalization opportunities)\n\n"

    if people_issues:
        for issue in people_issues[:report_limit]:
            canonical = issue["canonical"]
            variants_str = ", ".join([f"{v[0]} ({v[1]})" for v in issue["variants"]])
            report += f"- **{canonical}** ← {variants_str}\n"
            report += (
                "  - **Action:** Manual normalization or AI analysis recommended\n"
            )
    else:
        report += "*No people normalization issues found*\n"

    report += "\n### Top People\n\n"
    for person, count in normalization["top_people"]:
        report += f"- {person}: {count:,}\n"

    # Tags
    tags_issues = normalization["tags"]
    report += f"\n### Tags ({len(tags_issues)} normalization opportunities)\n\n"

    if tags_issues:
        for issue in tags_issues[:report_limit]:
            canonical = issue["canonical"]
            variants_str = ", ".join([f"{v[0]} ({v[1]})" for v in issue["variants"]])
            report += f"- **{canonical}** ← {variants_str}\n"
            report += (
                "  - **Action:** Manual normalization or AI analysis recommended\n"
            )
    else:
        report += "*No tag normalization issues found*\n"

    report += "\n### Top Tags\n\n"
    for tag, count in normalization["top_tags"]:
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
        except Exception:
            continue

    timestamps.sort()

    return {
        "total_things": len(things),
        "date_range": {
            "earliest": timestamps[0].isoformat() if timestamps else None,
            "latest": timestamps[-1].isoformat() if timestamps else None,
        },
        "completed": sum(1 for t in things if t.get("is_completed")),
        "strikethrough": sum(1 for t in things if t.get("is_strikethrough")),
        "with_links": sum(1 for t in things if t.get("links")),
        "with_tags": sum(1 for t in things if t.get("tags")),
        "with_people": sum(1 for t in things if t.get("people_mentioned")),
    }


def invoke_claude_analysis(
    json_path: Path,
    stats: Dict[str, Any],
    duplicates: List[Dict[str, Any]],
    normalization: Dict[str, Any],
    quality_issues: Dict[str, Any],
    changes: GroomingChanges,
    output_path: Path,
    timeout_seconds: int = 900,
    llm_context_note: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> bool:
    """
    Invoke LLM for semantic analysis with token-budget-aware compression.

    Args:
        json_path: Path to the original JSON input file.
        stats: Analysis statistics
        duplicates: Duplicate detection results
        normalization: Normalization opportunities
        quality_issues: Data quality issues
        changes: GroomingChanges object with auto-fix history
        output_path: Path where the AI report should be written.
        timeout_seconds: Timeout in seconds (default: 900 = 15 minutes).
        llm_context_note: Optional note about LLM time window or scope.

    Returns:
        True if the output file was written, otherwise False.
    """

    print("\n🤖 Invoking LLM for semantic analysis...")
    print(f"   ⏱️  Timeout: {timeout_seconds} seconds ({timeout_seconds//60} minutes)")
    print("   💭 Compressing reports to fit context window...")

    # Read the grooming prompt template.
    prompt_file = Path("docs/DATA_GROOMING_PROMPT.md")
    if not prompt_file.exists():
        print(f"❌ Prompt template not found: {prompt_file}")
        return False

    with open(prompt_file) as f:
        grooming_prompt = f.read()

    # Generate compressed prose summaries (token-budget aware)
    python_summary = generate_compressed_prose_report(
        stats, duplicates, normalization, quality_issues, token_budget=12000
    )

    # Compressed changes summary
    changes_summary = _compress_changes_summary(changes, token_budget=3000)

    # Calculate token estimates
    template_tokens = estimate_tokens(grooming_prompt)
    analysis_tokens = estimate_tokens(python_summary)
    changes_tokens = estimate_tokens(changes_summary)
    instructions_tokens = 150  # Rough estimate for instructions below

    total_prompt_tokens = template_tokens + analysis_tokens + changes_tokens + instructions_tokens

    print(f"   📊 Token budget (estimated):")
    print(f"      Template: ~{template_tokens:,} tokens")
    print(f"      Analysis: ~{analysis_tokens:,} tokens")
    print(f"      Changes: ~{changes_tokens:,} tokens")
    print(f"      Total prompt: ~{total_prompt_tokens:,} tokens")
    print(f"      Generation budget: ~{32000 - total_prompt_tokens:,} tokens")

    if total_prompt_tokens > 20000:
        print(f"   ⚠️  WARNING: Prompt exceeds target (20K), may hit context limit")

    # Build the full prompt
    full_prompt = f"""{grooming_prompt}

# PYTHON PRE-ANALYSIS (Compressed Summary)

{llm_context_note or ""}

{python_summary}

# AUTO-FIXES APPLIED

{changes_summary}

# YOUR TASK

Provide semantic analysis focusing on pattern interpretation, themes, and entity categorization. The mechanical analysis above is complete - focus on insights requiring semantic understanding.

Write your complete semantic grooming report directly (no file operations needed).
"""

    # Invoke LLM via unified provider abstraction
    try:
        llm_response = invoke_llm(
            prompt=full_prompt,
            response_format="text",
            timeout=timeout_seconds,
            config_path=config_path,
        )

        # Write response to output file
        with open(output_path, "w") as f:
            f.write(llm_response)

        print(f"✅ Semantic analysis complete: {output_path}")
        return True
    except RuntimeError as e:
        print(f"❌ LLM invocation failed: {e}")
        print(f"   💡 Tip: Increase timeout with --ai-timeout {timeout_seconds * 2}")
        return False
    except Exception as e:
        print(f"❌ LLM invocation error: {e}")
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

  # With AI semantic analysis (15 min timeout)
  python3 scripts/groom_data.py --ai-analysis

  # AI analysis with longer timeout for large datasets
  python3 scripts/groom_data.py --ai-analysis --ai-timeout 1800

  # Custom thresholds
  python3 scripts/groom_data.py --duplicate-window 3 --long-content-threshold 1000
""",
    )
    parser.add_argument(
        "json_file",
        type=Path,
        nargs="?",
        default=Path("data/processed/twos_data.json"),
        help="Path to JSON data file (default: data/processed/twos_data.json)",
    )
    parser.add_argument(
        "--duplicate-window",
        type=int,
        default=7,
        help="Days window for duplicate detection (default: 7)",
    )
    parser.add_argument(
        "--long-content-threshold",
        type=int,
        default=2000,
        help="Character threshold for flagging long content (default: 2000)",
    )
    parser.add_argument(
        "--report-limit",
        type=int,
        default=10,
        help="Number of items to show in reports (default: 10)",
    )
    parser.add_argument(
        "--ai-analysis",
        action="store_true",
        help="Invoke Claude Code for semantic analysis (uses subscription quota)",
    )
    parser.add_argument(
        "--ai-timeout",
        type=int,
        default=900,
        help="Timeout in seconds for AI analysis (default: 900 = 15 minutes)",
    )
    parser.add_argument(
        "--llm-months-back",
        type=int,
        default=0,
        help="Limit LLM analysis inputs to the last N months (0 = all data)",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        help="Path to LLM config YAML file (default: .llm_config.yaml if exists)",
    )

    args = parser.parse_args()

    # Validate input file before doing any processing.
    if not args.json_file.exists():
        print(f"❌ File not found: {args.json_file}")
        sys.exit(1)

    # Check Claude Code if AI analysis requested.
    if args.ai_analysis:
        if not check_claude_code():
            print("❌ No LLM provider available")
            print("\nInstall one of:")
            print("  - LM Studio: https://lmstudio.ai/ (local/free)")
            print("  - Claude Code CLI: https://code.claude.com/docs/en/quickstart")
            print("  - Anthropic API: pip install -e '.[llm-anthropic]' + set ANTHROPIC_API_KEY")
            sys.exit(1)
        print("✅ LLM provider available")

    # Load data from JSON on disk.
    things, metadata = load_json_data(args.json_file)
    original_count = len(things)

    # Run Python analysis (no external services).
    print(
        f"\n🔍 Running Python analysis "
        f"(window={args.duplicate_window} days, long={args.long_content_threshold} chars)..."
    )

    stats = analyze_statistics(things)
    duplicates = analyze_duplicates(things, args.duplicate_window)
    normalization = analyze_normalization(things)
    quality_issues = analyze_data_quality(things, args.long_content_threshold)

    print(f"  - Found {len(duplicates)} potential duplicates")
    exact_count = sum(1 for d in duplicates if d["is_exact"])
    print(f"    └─ {exact_count} exact (will be removed)")
    print(
        f"  - Found {len(normalization['people'])} people normalization opportunities"
    )
    print(f"  - Found {len(normalization['tags'])} tag normalization opportunities")
    print(f"  - Found {len(quality_issues)} data quality issue types")

    # Apply automatic fixes.
    print("\n🔧 Applying automatic fixes...")
    changes = GroomingChanges()

    # Add normalization flags to changes (for reporting).
    for person_issue in normalization["people"]:
        changes.add_flagged(
            "normalization",
            {
                "type": "person",
                "description": f"{person_issue['canonical']} has {len(person_issue['variants'])} variants",
                "recommendation": f"Standardize all to '{person_issue['canonical']}'",
            },
        )

    for tag_issue in normalization["tags"]:
        changes.add_flagged(
            "normalization",
            {
                "type": "tag",
                "description": f"{tag_issue['canonical']} has {len(tag_issue['variants'])} variants",
                "recommendation": f"Standardize all to '{tag_issue['canonical']}'",
            },
        )

    cleaned_things = apply_fixes(things, duplicates, quality_issues, changes)
    cleaned_count = len(cleaned_things)

    print(f"  ✅ Removed {len(changes.removed)} exact duplicates")
    print(f"  ✅ Fixed {len(changes.modified)} broken references")
    print(f"  ✅ Cleaned: {cleaned_count:,} things (from {original_count:,})")

    llm_stats = stats
    llm_duplicates = duplicates
    llm_normalization = normalization
    llm_quality_issues = quality_issues
    llm_changes = changes
    llm_context_note = None

    if args.ai_analysis and args.llm_months_back > 0:
        llm_things = filter_things_by_months(things, args.llm_months_back)
        llm_total = len(llm_things)
        print(
            f"\n🧠 LLM time filter: last {args.llm_months_back} months "
            f"({llm_total:,} of {original_count:,} things)"
        )

        llm_stats = analyze_statistics(llm_things)
        llm_duplicates = analyze_duplicates(llm_things, args.duplicate_window)
        llm_normalization = analyze_normalization(llm_things)
        llm_quality_issues = analyze_data_quality(
            llm_things, args.long_content_threshold
        )

        things_by_id = {t["id"]: t for t in things}
        llm_changes = filter_changes_for_llm(
            changes, things_by_id, args.llm_months_back
        )

        for person_issue in llm_normalization["people"]:
            llm_changes.add_flagged(
                "normalization",
                {
                    "type": "person",
                    "description": f"{person_issue['canonical']} has {len(person_issue['variants'])} variants",
                    "recommendation": f"Standardize all to '{person_issue['canonical']}'",
                },
            )

        for tag_issue in llm_normalization["tags"]:
            llm_changes.add_flagged(
                "normalization",
                {
                    "type": "tag",
                    "description": f"{tag_issue['canonical']} has {len(tag_issue['variants'])} variants",
                    "recommendation": f"Standardize all to '{tag_issue['canonical']}'",
                },
            )

        llm_context_note = (
            f"LLM time window: last {args.llm_months_back} months. "
            f"Items summarized: {llm_total:,} of {original_count:,}."
        )

    # Save cleaned data to a new JSON file on disk.
    cleaned_path = args.json_file.parent / (args.json_file.stem + "_cleaned.json")

    # Preserve original structure
    if metadata:
        output_data = {
            "metadata": {
                **metadata,
                "cleaned_at": datetime.now().isoformat(),
                "original_count": original_count,
                "cleaned_count": cleaned_count,
            },
            "tasks": cleaned_things,
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

    print("\n📊 Changes report saved:")
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

    # Optionally run AI analysis (uses configured LLM provider).
    if args.ai_analysis:
        ai_report_path = reports_dir / f"{timestamp}-ai-analysis.md"
        success = invoke_claude_analysis(
            args.json_file,
            llm_stats,
            llm_duplicates,
            llm_normalization,
            llm_quality_issues,
            llm_changes,
            ai_report_path,
            args.ai_timeout,
            llm_context_note,
            config_path=args.llm_config,
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
        print(
            "      python3 scripts/classify_entities.py --ai-classify --apply-mappings"
        )
        print("\n   2. Then load normalized data to SQLite:")
        print(
            "      python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned_normalized.json"
        )
        print("\n   OR skip entity classification and load cleaned data now:")
        print(f"      python3 scripts/load_to_sqlite.py {cleaned_path}")
    else:
        print("\n📁 Next step: Load cleaned data into SQLite:")
        print(f"   python3 scripts/load_to_sqlite.py {cleaned_path}")


if __name__ == "__main__":
    main()
