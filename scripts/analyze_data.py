#!/usr/bin/env python3
"""
Data quality analysis script (from Gemini grooming analysis).
Analyzes converted JSON for duplicates, normalization needs, quality issues, and patterns.
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# Helper to parse ISO format timestamps used in the JSON export.
def parse_ts(ts_str: str | None) -> datetime | None:
    """
    Parse an ISO timestamp into a datetime.

    Returns:
        A datetime object or None if parsing fails.
    """
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None


def analyze_data(file_path: Path) -> None:
    """
    Run complete data quality analysis.

    Args:
        file_path: Path to the JSON data file to analyze.

    Side effects:
        Reads a JSON file from disk and prints results to stdout.

    Returns:
        None.
    """

    print(f"Loading {file_path}...")
    # I/O boundary: read JSON data from disk.
    with open(file_path, "r", encoding="utf-8") as f:
        data_raw = json.load(f)

    # Handle both dict wrapper or list.
    if isinstance(data_raw, dict) and "tasks" in data_raw:
        tasks = data_raw["tasks"]
        metadata = data_raw.get("metadata", {})
    else:
        tasks = data_raw
        metadata = {}

    print(f"Loaded {len(tasks)} things.")
    if metadata:
        date_range = metadata.get("date_range", {})
        earliest = date_range.get("earliest")
        latest = date_range.get("latest")
        print(f"Date range: {earliest} to {latest}")

    # --- 1. Duplicate Detection ---
    print("\n=== 1. DUPLICATE DETECTION ===")

    content_groups = defaultdict(list)
    for t in tasks:
        c = t.get("content", "")
        if not c:
            continue
        c_norm = c.strip().lower()
        content_groups[c_norm].append(t)

    duplicates = []

    for c_norm, group in content_groups.items():
        if len(group) < 2:
            continue

        # Sort by timestamp so nearby entries can be compared.
        group_sorted = sorted(
            group, key=lambda x: parse_ts(x.get("timestamp")) or datetime.min
        )

        for i in range(len(group_sorted)):
            t1 = group_sorted[i]
            ts1 = parse_ts(t1.get("timestamp"))
            if not ts1:
                continue

            for j in range(i + 1, len(group_sorted)):
                t2 = group_sorted[j]
                ts2 = parse_ts(t2.get("timestamp"))
                if not ts2:
                    continue

                diff = abs((ts2 - ts1).days)

                # Within 7 days check (conservative for recurring things).
                if diff <= 7:
                    p1 = t1.get("parent_task_id")
                    p2 = t2.get("parent_task_id")

                    if p1 == p2:
                        duplicates.append(
                            {
                                "id1": t1["id"],
                                "ts1": t1["timestamp"],
                                "id2": t2["id"],
                                "ts2": t2["timestamp"],
                                "content": t1["content"],
                                "diff": diff,
                            }
                        )

    print(f"Duplicate Pairs Found (within 7 days): {len(duplicates)}")
    if duplicates:
        print("\nSample duplicates:")
        for dup in duplicates[:5]:
            print(
                f"  - {dup['id1']}, {dup['id2']}: \"{dup['content'][:50]}...\" (gap: {dup['diff']} days)"
            )

    # --- 2. Entity Normalization ---
    print("\n=== 2. ENTITY NORMALIZATION ===")

    people_counts = Counter()
    tag_counts = Counter()

    for t in tasks:
        for p in t.get("people_mentioned", []):
            people_counts[p] += 1
        for tag in t.get("tags", []):
            tag_counts[tag] += 1

    # Find case variants
    def find_case_variants(counter):
        """
        Group terms by lowercase to find case-only variants.

        Returns:
            A list of lists, where each inner list contains (term, count) pairs.
        """
        variants = defaultdict(set)
        for term in counter:
            variants[term.lower()].add(term)

        issues = []
        for lower, originals in variants.items():
            if len(originals) > 1:
                details = [(term, counter[term]) for term in originals]
                details.sort(key=lambda x: x[1], reverse=True)
                issues.append(details)
        return issues

    people_variants = find_case_variants(people_counts)
    tag_variants = find_case_variants(tag_counts)

    print(f"People Variants: {len(people_variants)}")
    if people_variants:
        print("\nSample people case variants:")
        for variants in people_variants[:5]:
            print(f"  - {variants}")

    print(f"\nTag Variants: {len(tag_variants)}")
    if tag_variants:
        print("\nSample tag case variants:")
        for variants in tag_variants[:5]:
            print(f"  - {variants}")

    print("\nTop 10 People:")
    for person, count in people_counts.most_common(10):
        print(f"  - {person}: {count}")

    print("\nTop 10 Tags:")
    for tag, count in tag_counts.most_common(10):
        print(f"  - {tag}: {count}")

    # --- 3. Data Quality Validation ---
    print("\n=== 3. DATA QUALITY VALIDATION ===")

    parsing_errors = []
    broken_refs = []
    anomalies = []

    id_set = set(t["id"] for t in tasks)
    ts_counts = Counter()

    for t in tasks:
        tid = t.get("id")
        ts_str = t.get("timestamp")
        ts_raw = t.get("timestamp_raw")

        # Malformed timestamps: raw exists but parsed timestamp is missing.
        if not ts_str and ts_raw:
            parsing_errors.append(f"{tid}: timestamp null but raw exists '{ts_raw}'")

        # Broken references: parent_task_id does not exist in dataset.
        pid = t.get("parent_task_id")
        if pid and pid not in id_set:
            broken_refs.append(f"{tid}: parent {pid} not found")

        # Long content can signal pasted blocks or parsing issues.
        if len(t.get("content", "")) > 2000:
            anomalies.append(f"{tid}: Content length {len(t.get('content'))}")

        # Timestamp clusters: many items share the exact same timestamp.
        if ts_str:
            ts_counts[ts_str] += 1

    suspicious_timestamps = [
        (ts, count) for ts, count in ts_counts.items() if count > 10
    ]

    print(f"Parsing Errors: {len(parsing_errors)}")
    if parsing_errors:
        for err in parsing_errors[:5]:
            print(f"  - {err}")

    print(f"\nBroken References: {len(broken_refs)}")
    if broken_refs:
        for ref in broken_refs[:5]:
            print(f"  - {ref}")

    print(f"\nAnomalies (long content): {len(anomalies)}")
    if anomalies:
        for anom in anomalies[:5]:
            print(f"  - {anom}")

    print(f"\nSuspicious Timestamps (>10 occurrences): {len(suspicious_timestamps)}")
    if suspicious_timestamps:
        for ts, count in suspicious_timestamps[:5]:
            print(f"  - {ts}: {count} occurrences")

    # --- 5. Pattern Detection ---
    print("\n=== 5. PATTERN DETECTION ===")

    # Theme detection via keywords using simple token counts.
    all_words = []
    stop_words = set(
        [
            "the",
            "and",
            "to",
            "a",
            "in",
            "for",
            "of",
            "on",
            "with",
            "at",
            "is",
            "it",
            "my",
            "I",
            "i",
            "-",
            "&",
        ]
    )
    for t in tasks:
        c = t.get("content", "").lower()
        words = re.findall(r"\b[a-z]{3,}\b", c)
        all_words.extend([w for w in words if w not in stop_words])

    theme_counts = Counter(all_words).most_common(20)

    print("\nTop 20 Keywords:")
    for word, count in theme_counts:
        print(f"  - {word}: {count}")

    # Temporal patterns by hour and day of week.
    hour_counts = Counter()
    day_counts = Counter()
    for t in tasks:
        ts = parse_ts(t.get("timestamp"))
        if ts:
            hour_counts[ts.hour] += 1
            day_counts[ts.strftime("%A")] += 1

    print("\nThing Creation by Hour (top 5):")
    for hour, count in hour_counts.most_common(5):
        print(f"  - {hour:02d}:00: {count} things")

    print("\nThing Creation by Day of Week:")
    for day, count in day_counts.most_common():
        print(f"  - {day}: {count} things")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Total Things: {len(tasks)}")
    print(f"Things with People: {sum(1 for t in tasks if t.get('people_mentioned'))}")
    print(f"Things with Tags: {sum(1 for t in tasks if t.get('tags'))}")
    print(f"Things with Links: {sum(1 for t in tasks if t.get('links'))}")
    print(f"Completed Things: {sum(1 for t in tasks if t.get('is_completed'))}")
    print(f"Strikethrough Things: {sum(1 for t in tasks if t.get('is_strikethrough'))}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = (
            Path(__file__).parent.parent / "data" / "processed" / "twos_data.json"
        )

    # CLI entry: analyze the provided file path.
    analyze_data(file_path)
