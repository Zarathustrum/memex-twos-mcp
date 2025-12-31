#!/usr/bin/env python3
"""
Build list metadata from Twos JSON export.

This script analyzes the structured JSON data and:
1. Classifies section headers as date-based lists vs topic-based lists
2. Assigns item_type to each thing (content/divider/header/metadata)
3. Generates deterministic list_id for each section
4. Computes list boundaries (start_line, end_line)
5. Counts substantive items per list

Outputs:
- Enhanced JSON with item_type and list_id fields added to each thing
- Separate lists_metadata.json with list registry

Usage:
    python3 scripts/build_list_metadata.py data/processed/twos_data.json

Options:
    -o, --output         Output JSON file (default: input_with_lists.json)
    --lists-output       Lists metadata file (default: lists_metadata.json)
    --pretty             Pretty-print JSON output
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dateutil import parser as dateparser
from collections import defaultdict


def slugify(text: str) -> str:
    """
    Convert text to URL-safe slug.

    Args:
        text: Input text

    Returns:
        Slugified text (lowercase, hyphens, no special chars)

    Example:
        >>> slugify("Tech Projects")
        'tech-projects'
    """
    # Convert to lowercase
    text = text.lower()
    # Replace non-alphanumeric with hyphens
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    # Strip leading/trailing hyphens
    return text.strip('-')


def parse_date_header(text: str) -> Optional[datetime]:
    """
    Parse various date formats from section headers.

    Handles:
    - 'Mon, Dec 30, 2025'
    - 'December 30, 2025'
    - '12/30/2025'
    - 'Tue, Oct 8, 2023'

    Args:
        text: Header text that might contain a date

    Returns:
        Parsed datetime object or None if parsing fails
    """
    if not text or len(text) < 3:
        return None

    try:
        # Use dateutil.parser with fuzzy matching to extract dates
        # fuzzy=True allows surrounding text
        dt = dateparser.parse(text, fuzzy=True)
        return dt
    except (ValueError, TypeError):
        return None


def classify_section_header(header: str) -> Tuple[str, str, Optional[str]]:
    """
    Classify section header as date-based or topic-based.

    Returns: (list_type, list_name, list_date)

    Args:
        header: Section header text (e.g., "Mon, Dec 30, 2025" or "Tech Projects")

    Returns:
        Tuple of (list_type, list_name, list_date)
        - list_type: 'date' | 'topic' | 'metadata'
        - list_name: Normalized name for indexing
        - list_date: ISO date string (for date lists) or None

    Examples:
        >>> classify_section_header("Mon, Dec 30, 2025")
        ('date', '2025-12-30', '2025-12-30')
        >>> classify_section_header("Tech Projects")
        ('topic', 'Tech Projects', None)
        >>> classify_section_header("---- divider ----")
        ('metadata', 'divider', None)
    """
    # Normalize whitespace
    header_clean = ' '.join(header.split()).strip()

    if not header_clean:
        return ('metadata', 'unknown', None)

    # Check for divider patterns (3+ repeated chars)
    if re.match(r'^[\-=_\*•]{3,}$', header_clean):
        return ('metadata', 'divider', None)

    # Try parsing as date
    date_parsed = parse_date_header(header_clean)
    if date_parsed:
        # It's a date-based list
        list_date = date_parsed.date().isoformat()
        return ('date', list_date, list_date)

    # Default: topic list
    return ('topic', header_clean, None)


def is_divider(content: str) -> bool:
    """
    Check if content is a divider line.

    Args:
        content: Thing content

    Returns:
        True if content matches divider patterns
    """
    if not content or len(content) < 3:
        return False

    # Dividers: lines with only dashes, equals, underscores, asterisks
    return bool(re.match(r'^[\-=_\*•]{3,}$', content.strip()))


def classify_item_type(thing: Dict[str, Any]) -> str:
    """
    Classify a thing as content/divider/header/metadata.

    Args:
        thing: Thing dictionary with content, content_raw fields

    Returns:
        'content' | 'divider' | 'header' | 'metadata'

    Logic:
    - Dividers: lines with only symbols (---, ===, ___, ***)
    - Headers: short (< 50 chars), all-caps, or ends with ':'
    - Metadata: system-generated (starts/ends with brackets)
    - Content: everything else (substantive items)
    """
    content = thing.get('content', '').strip()
    content_raw = thing.get('content_raw', '').strip()

    if not content:
        return 'metadata'

    # Dividers: lines with only dashes, equals, or symbols
    if is_divider(content):
        return 'divider'

    # Headers: short, all-caps, or ends with ':'
    if len(content) < 50 and (content.isupper() or content.endswith(':')):
        return 'header'

    # Metadata: system-generated (e.g., section timestamps, markers)
    if content.startswith('[') and content.endswith(']'):
        return 'metadata'

    # Everything else is substantive content
    return 'content'


def generate_list_id(list_type: str, list_name: str, start_line: int) -> str:
    """
    Generate deterministic list_id.

    Args:
        list_type: 'date' | 'topic' | 'metadata'
        list_name: Normalized list name
        start_line: Start line number (for disambiguation)

    Returns:
        Unique, stable list_id

    Examples:
        >>> generate_list_id('date', '2025-12-30', 1234)
        'date_2025-12-30'
        >>> generate_list_id('topic', 'Tech Projects', 5678)
        'topic_tech-projects_5678'
    """
    if list_type == 'date':
        # Date lists: use date as identifier (unique per export)
        # Assumption: one section per date per export
        return f"date_{list_name}"

    # Topic lists: slugify name + line number for disambiguation
    # (same topic might appear multiple times)
    slug = slugify(list_name)
    return f"{list_type}_{slug}_{start_line}"


def build_list_metadata(tasks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Build list metadata from things.

    Args:
        tasks: List of thing dictionaries from JSON

    Returns:
        Tuple of (enriched_tasks, lists_metadata)
        - enriched_tasks: tasks with item_type, list_id, is_substantive added
        - lists_metadata: list of list dictionaries

    Algorithm:
    1. Sort by line_number
    2. Group by section_header
    3. For each section:
       a. Classify header (date vs topic)
       b. Determine boundaries (start_line, end_line)
       c. Classify each item (content/divider/header/metadata)
       d. Generate deterministic list_id
       e. Assign list_id to all items in section
       f. Count substantive items
    4. Return enriched tasks + list metadata
    """
    print(f"  Building list metadata for {len(tasks)} things...")

    # Sort by line_number for sequential processing
    tasks_sorted = sorted(tasks, key=lambda t: t.get('line_number', 0))

    # Group by section_header
    sections = defaultdict(list)
    for task in tasks_sorted:
        header = task.get('section_header', 'Unknown')
        sections[header].append(task)

    print(f"  Found {len(sections)} unique sections")

    lists_metadata = []
    list_id_counter = {}  # Track disambiguators for repeated headers

    for header, items in sections.items():
        if not items:
            continue

        # Step 1: Classify section header
        list_type, list_name, list_date = classify_section_header(header)

        # Step 2: Determine boundaries
        start_line = min(t.get('line_number', 0) for t in items)
        end_line = max(t.get('line_number', 0) for t in items)

        # Step 3: Classify items and count substantive
        substantive_count = 0
        for item in items:
            # Classify item type
            item_type = classify_item_type(item)
            item['item_type'] = item_type

            # Determine if substantive
            is_substantive = (item_type == 'content' and len(item.get('content', '')) > 3)
            item['is_substantive'] = is_substantive

            if is_substantive:
                substantive_count += 1

        # Step 4: Generate deterministic list_id
        list_id = generate_list_id(list_type, list_name, start_line)

        # Disambiguate if needed (same normalized name, different sections)
        # This handles repeated headers like "Tech Projects" appearing multiple times
        key = f"{list_type}_{list_name}"
        if key in list_id_counter:
            list_id_counter[key] += 1
            # For date lists, this shouldn't happen (one date per day)
            # For topic lists, append counter
            if list_type != 'date':
                list_id = f"{list_id}_{list_id_counter[key]}"
        else:
            list_id_counter[key] = 0

        # Step 5: Assign list_id to all items
        for item in items:
            item['list_id'] = list_id

        # Step 6: Build list metadata entry
        lists_metadata.append({
            'list_id': list_id,
            'list_type': list_type,
            'list_name': list_name,
            'list_name_raw': header,
            'list_date': list_date,
            'start_line': start_line,
            'end_line': end_line,
            'item_count': len(items),
            'substantive_count': substantive_count,
            'created_at': datetime.now().isoformat(),
        })

    print(f"  Created {len(lists_metadata)} list entries")
    print(f"  - Date lists: {sum(1 for l in lists_metadata if l['list_type'] == 'date')}")
    print(f"  - Topic lists: {sum(1 for l in lists_metadata if l['list_type'] == 'topic')}")
    print(f"  - Metadata sections: {sum(1 for l in lists_metadata if l['list_type'] == 'metadata')}")

    return tasks_sorted, lists_metadata


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build list metadata from Twos JSON export"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to Twos JSON file (output from convert_to_json.py)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output JSON file with enhanced metadata (default: input_with_lists.json)"
    )
    parser.add_argument(
        "--lists-output",
        type=Path,
        help="Lists metadata output file (default: lists_metadata.json)"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    # Default output paths
    if not args.output:
        args.output = args.input_file.parent / f"{args.input_file.stem}_with_lists.json"

    if not args.lists_output:
        args.lists_output = args.input_file.parent / "lists_metadata.json"

    print(f"Loading {args.input_file}...")

    # Load input JSON
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both raw list and {metadata, tasks} structure
    metadata = {}
    if isinstance(data, dict) and 'tasks' in data:
        tasks = data['tasks']
        metadata = data.get('metadata', {})
    else:
        tasks = data

    print(f"  Loaded {len(tasks)} things")

    # Build list metadata
    enriched_tasks, lists_metadata = build_list_metadata(tasks)

    # Update metadata
    if isinstance(data, dict) and 'metadata' in data:
        data['metadata']['list_metadata_built_at'] = datetime.now().isoformat()
        data['metadata']['total_lists'] = len(lists_metadata)
        data['tasks'] = enriched_tasks
        output_data = data
    else:
        output_data = {
            'metadata': {
                'list_metadata_built_at': datetime.now().isoformat(),
                'total_lists': len(lists_metadata),
                'total_tasks': len(enriched_tasks),
            },
            'tasks': enriched_tasks,
        }

    # Write enhanced JSON
    print(f"Writing enhanced JSON to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        if args.pretty:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(output_data, f, ensure_ascii=False)

    # Write lists metadata
    print(f"Writing lists metadata to {args.lists_output}...")
    with open(args.lists_output, 'w', encoding='utf-8') as f:
        if args.pretty:
            json.dump(lists_metadata, f, indent=2, ensure_ascii=False)
        else:
            json.dump(lists_metadata, f, ensure_ascii=False)

    print("\n✅ List metadata build complete!")
    print(f"  Enhanced JSON: {args.output}")
    print(f"  Lists metadata: {args.lists_output}")

    # Print sample list
    if lists_metadata:
        print("\nSample list:")
        print(json.dumps(lists_metadata[0], indent=2))

    return 0


if __name__ == "__main__":
    exit(main())
