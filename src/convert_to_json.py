#!/usr/bin/env python3
"""
Convert Twos markdown export (Markdown with timestamps format) to structured JSON.

I/O Boundary: Reads markdown file from disk, writes JSON to disk.

This script parses the Twos "Markdown with timestamps" export format and extracts:
- Task content and timestamps
- Hierarchical relationships (parent/child tasks)
- Metadata (completion status, links, tags, people)
- Original date section groupings

ID Strategy:
- Uses content-hash based stable IDs (not sequential counters)
- IDs remain stable across exports even when file lines shift
- Critical for incremental database updates (append/sync modes)
- Matches the hash computation in load_to_sqlite.py for consistency

Assumptions:
- US date format (MM/DD/YY or MM/DD/YYYY)
- Twos export format with # ⌛️ section headers
- Tab or 8-space indentation for hierarchy
"""

import re
import json
import sys
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any, TypedDict
from pathlib import Path
import argparse


def parse_timestamp(timestamp_str: str) -> Optional[str]:
    """
    Parse timestamp string to ISO format.

    Assumption: US date format (MM/DD/YY or MM/DD/YYYY). Non-US users with
    DD/MM/YY exports will get incorrect dates or parsing failures.

    Args:
        timestamp_str: Timestamp like "10/27/23 9:14 pm" or "10/27/2023 9:14 pm"

    Returns:
        ISO format timestamp or None if parsing fails

    Edge cases:
        - Returns None if the string does not match expected formats.
        - Assumes month/day/year with a 12-hour clock and am/pm.
        - Supports both 2-digit (23) and 4-digit (2023) years.
        - No timezone handling (naive datetime, assumes local time).
    """
    formats = [
        "%m/%d/%y %I:%M %p",
        "%m/%d/%y %I:%M%p",
        "%m/%d/%Y %I:%M %p",  # 4-digit year support
        "%m/%d/%Y %I:%M%p",  # 4-digit year without space before am/pm
    ]

    timestamp_str = timestamp_str.strip()
    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp_str, fmt)
            return dt.isoformat()
        except ValueError:
            continue
    return None


def compute_stable_id(timestamp_iso: str, content: str, section_header: str) -> str:
    """
    Generate stable ID from content hash.

    Why hash-based IDs:
    - Sequential IDs (task_00001) break when file lines shift (add/remove items).
    - Incremental database updates rely on matching incoming IDs to existing rows.
    - Without stable IDs, incremental mode would treat all items as new/deleted.

    Collision risk:
    - Uses first 12 chars of SHA256 (2^48 combinations).
    - Collision probability negligible for personal datasets (<100K items).
    - If collision occurs, newer item will overwrite older in incremental mode.

    Canonical fields:
    - Must match load_to_sqlite.py:compute_content_hash() for consistency.
    - Excludes line_number, content_raw (formatting/position don't affect identity).

    Args:
        timestamp_iso: ISO format timestamp
        content: Cleaned content text
        section_header: Section header text

    Returns:
        12-character hex ID (e.g., "task_a3f9b2c1e5d4")
    """
    canonical = {
        "timestamp": timestamp_iso,
        "content": content.strip(),
        "section_header": section_header.strip(),
    }

    # Sort keys for deterministic JSON
    hash_input = json.dumps(canonical, sort_keys=True)
    full_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    # Use first 12 chars for reasonable uniqueness (2^48 combinations)
    # Prefix with "task_" for readability
    return f"task_{full_hash[:12]}"


def extract_links(text: str) -> List[Dict[str, str]]:
    """
    Extract markdown links from text.

    Args:
        text: The raw task content that may contain Markdown links.

    Returns:
        A list of {"text": ..., "url": ...} dictionaries for each link found.
    """
    # Pattern: [text](url)
    pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
    links = []
    for match in re.finditer(pattern, text):
        links.append({"text": match.group(1), "url": match.group(2)})
    return links


def extract_tags(text: str) -> List[str]:
    """
    Extract #tags# from text and normalize to lowercase.

    Args:
        text: The raw task content.

    Returns:
        A list of tag strings without the surrounding # characters.
    """
    # Pattern: #word#
    pattern = r"#(\w+)#"
    tags = re.findall(pattern, text)
    # Normalize to lowercase for consistency
    return [tag.lower() for tag in tags]


# Global spaCy model cache (loaded once, reused for all extractions)
_SPACY_MODEL_CACHE = None


def extract_people_ner(text: str) -> List[str]:
    """
    Extract people mentions using spaCy Named Entity Recognition.

    Uses cached model to avoid reloading for every call.

    Args:
        text: Content to extract names from

    Returns:
        List of unique person names (PERSON entities)

    Raises:
        ImportError: If spaCy not available or model not downloaded
    """
    global _SPACY_MODEL_CACHE

    import spacy  # type: ignore

    # Load model once and cache
    if _SPACY_MODEL_CACHE is None:
        try:
            _SPACY_MODEL_CACHE = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )

    nlp = _SPACY_MODEL_CACHE

    if not text:
        return []

    # Strip tags before NER for better accuracy
    # Tags like #personal# interfere with spaCy's context analysis
    text_clean = re.sub(r"#\w+#", "", text)
    text_clean = " ".join(text_clean.split())

    if not text_clean:
        return []

    # Run NER pipeline
    doc = nlp(text_clean)

    # Extract PERSON entities
    people = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            # Skip single-character names (likely errors)
            if len(name) > 1:
                people.append(name)

    return list(set(people))


def extract_people_batch_ner(texts: List[str]) -> List[List[str]]:
    """
    Extract people from multiple texts in batch (faster than individual calls).

    Args:
        texts: List of content strings

    Returns:
        List of lists, each containing extracted names for corresponding text

    Raises:
        ImportError: If spaCy not available or model not downloaded
    """
    global _SPACY_MODEL_CACHE

    import spacy  # type: ignore

    if _SPACY_MODEL_CACHE is None:
        try:
            _SPACY_MODEL_CACHE = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )

    nlp = _SPACY_MODEL_CACHE

    # Process texts in batch using nlp.pipe()
    results = []

    # nlp.pipe() is more efficient than individual nlp() calls
    for doc in nlp.pipe(texts, batch_size=50):
        people = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if len(name) > 1:
                    people.append(name)
        results.append(list(set(people)))

    return results


def extract_people(text: str, use_ner: bool = True) -> List[str]:
    """
    Extract people mentions from text.

    Tries NER-based extraction first (if spaCy available), falls back to regex.

    Strategy (NER mode):
    - Use spaCy Named Entity Recognition to identify PERSON entities
    - High accuracy, low false positive rate

    Strategy (regex fallback mode):
    - Find capitalized words (proper nouns)
    - Filter out common non-name words
    - Look for possessive forms (Alex's, Mom's)
    - Normalize common variations (mom -> Mom)

    Args:
        text: Content to extract names from
        use_ner: Whether to attempt NER extraction (default: True)

    Returns:
        A deduplicated list of likely person names. False positives are possible
        in regex mode, but rare in NER mode.
    """
    # Try NER extraction first if enabled
    if use_ner:
        try:
            return extract_people_ner(text)
        except ImportError as e:
            # spaCy not installed or model not downloaded
            print(
                f"⚠️  spaCy NER not available ({e}), falling back to regex extraction",
                file=sys.stderr,
            )
            print(
                "💡 Tip: Install spaCy with: pip install spacy && python -m spacy download en_core_web_sm",
                file=sys.stderr,
            )
            # Fall through to regex

    # Regex-based extraction (fallback mode)
    people = []

    # Common stop words that are capitalized but not names.
    stop_words = {
        # Pronouns and question words
        "I",
        "You",
        "He",
        "She",
        "It",
        "We",
        "They",
        "Them",
        "Their",
        "Our",
        "My",
        "Your",
        "This",
        "That",
        "These",
        "Those",
        "What",
        "When",
        "Where",
        "Who",
        "Why",
        "How",
        "Which",
        "Whose",
        # Articles, conjunctions, prepositions
        "A",
        "An",
        "The",
        "And",
        "Or",
        "But",
        "For",
        "To",
        "From",
        "In",
        "On",
        "At",
        "Of",
        "With",
        "By",
        "About",
        "As",
        "Into",
        "Through",
        "During",
        "Before",
        "After",
        # Days and months
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        # Time zones
        "Am",
        "Pm",
        "Est",
        "Pst",
        "Utc",
        "Pdt",
        "Edt",
        # Common action verbs (often sentence starts)
        "Call",
        "Email",
        "Text",
        "Message",
        "Meeting",
        "Send",
        "Talk",
        "Ask",
        "Tell",
        "Buy",
        "Get",
        "Make",
        "Do",
        "Go",
        "See",
        "Find",
        "Check",
        "Fix",
        "Clean",
        "Read",
        "Write",
        "Start",
        "Finish",
        "Complete",
        "Schedule",
        "Cancel",
        "Confirm",
        "Update",
        "Review",
        "Follow",
        "Watch",
        "Listen",
        "Plan",
        "Organize",
        "Sort",
        "Pick",
        "Drop",
        "Take",
        "Bring",
        "Pack",
        "Unpack",
        "Move",
        "Transfer",
        "Reply",
        "Set",
        "Install",
        "Look",
        "Try",
        "Add",
        "Remove",
        "Delete",
        "Open",
        "Close",
        "Save",
        "Print",
        "Copy",
        "Paste",
        "Cut",
        "Search",
        "Replace",
        "Change",
        # Common nouns that aren't people
        "Flight",
        "Trip",
        "Travel",
        "Vacation",
        "Holiday",
        "Visit",
        "Tour",
        "Meeting",
        "Conference",
        "Event",
        "Party",
        "Dinner",
        "Lunch",
        "Breakfast",
        "House",
        "Home",
        "Office",
        "Work",
        "School",
        "College",
        "University",
        "Car",
        "Bike",
        "Bus",
        "Train",
        "Plane",
        "Uber",
        "Lyft",
        "Phone",
        "Computer",
        "Laptop",
        "Tablet",
        "Iphone",
        "Android",
        "Email",
        "Zoom",
        "Skype",
        "Teams",
        "Slack",
        "Google",
        "Amazon",
        "Apple",
        "Tesla",
        "Netflix",
        "Spotify",
        "Youtube",
        "Facebook",
        "Twitter",
        "Instagram",
        # Common placeholders
        "Thing",
        "Item",
        "Note",
        "Task",
        "Todo",
        "Reminder",
        "List",
    }

    # Common name patterns and normalizations.
    # Lowercase -> proper case mapping to reduce duplicates.
    name_normalizations = {
        "mom": "Mom",
        "mother": "Mom",
        "dad": "Dad",
        "father": "Dad",
        "dr": "Dr",
        "mr": "Mr",
        "mrs": "Mrs",
        "ms": "Ms",
    }

    # Remove links first (they often have capitalized text).
    text_no_links = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Find all capitalized words (2+ chars).
    # Pattern: word that starts with capital, may have apostrophe (Alex's).
    pattern = r"\b([A-Z][a-z]+(?:\'s)?)\b"
    matches = re.findall(pattern, text_no_links)

    for match in matches:
        # Remove possessive for normalization ("Alex's" -> "Alex").
        name = match.replace("'s", "")

        # Skip if in stop words to avoid common false positives.
        if name in stop_words:
            continue

        # Normalize common variations (e.g., "mom" -> "Mom").
        name_lower = name.lower()
        if name_lower in name_normalizations:
            name = name_normalizations[name_lower]

        # Skip single letters (often initials, not full names).
        if len(name) <= 1:
            continue

        people.append(name)

    # Also look for all-lowercase common family terms and normalize.
    text_lower = text_no_links.lower()
    for lowercase, proper in name_normalizations.items():
        # Use word boundaries to avoid false matches.
        if re.search(r"\b" + lowercase + r"\b", text_lower):
            people.append(proper)

    # Deduplicate because the same name may be found via multiple paths.
    return list(set(people))


def strip_tags_for_ner(text: str) -> str:
    """
    Strip #tags# from text before NER processing.

    Tags like #personal# can interfere with spaCy's context analysis,
    reducing NER accuracy. This function removes them while preserving
    the rest of the text.

    Args:
        text: Content with potential #tags#

    Returns:
        Text with tags removed

    Example:
        >>> strip_tags_for_ner("Send card to Jennifer #personal#")
        'Send card to Jennifer'
    """
    # Remove #tag# patterns
    text_no_tags = re.sub(r"#\w+#", "", text)
    # Clean up extra whitespace
    return " ".join(text_no_tags.split())


def is_completed(content: str) -> bool:
    """
    Check if task is marked as completed via a checked Markdown checkbox.

    Args:
        content: The raw line content from the Twos export.

    Returns:
        True if the line starts with "- [x]".
    """
    return content.lstrip().startswith("- [x]")


def is_pending_checkbox(content: str) -> bool:
    """
    Check if task has an unchecked Markdown checkbox.

    Args:
        content: The raw line content from the Twos export.

    Returns:
        True if the line starts with "- [ ]".
    """
    return content.lstrip().startswith("- [ ]")


def is_strikethrough(content: str) -> bool:
    """
    Check if content has strikethrough-style formatting.

    Args:
        content: The cleaned task content.

    Returns:
        True if a dash-wrapped pattern is detected.

    Note:
        This is a very simple heuristic that looks for dash-wrapped text.
    """
    return bool(re.search(r"-[^-]+-", content))


def get_bullet_type(content: str) -> str:
    """
    Determine the bullet/marker type used for the task line.

    Args:
        content: The raw line content from the Twos export.

    Returns:
        A string label such as "checkbox_done", "dash", or "bullet".
    """
    stripped = content.lstrip()
    if stripped.startswith("- [x]"):
        return "checkbox_done"
    elif stripped.startswith("- [ ]"):
        return "checkbox_pending"
    elif stripped.startswith("- "):
        return "dash"
    elif stripped.startswith("• "):
        return "bullet"
    return "unknown"


def clean_content(content: str) -> str:
    """
    Remove bullet markers and trailing timestamps from content.

    Args:
        content: The raw line content from the Twos export.

    Returns:
        A cleaned string suitable for storage and indexing.
    """
    # Remove leading bullet/checkbox formatting from the line.
    content = re.sub(r"^\s*[-•]\s*(\[[ x]\]\s*)?", "", content)
    # Remove trailing timestamp to keep content text-only.
    content = re.sub(
        r"\s+\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*[ap]m\s*$",
        "",
        content,
        flags=re.IGNORECASE,
    )
    return content.strip()


def parse_twos_file(file_path: Path, use_ner: bool = True) -> Dict[str, Any]:
    """
    Parse Twos markdown file into structured data.

    Args:
        file_path: Path to the Twos export file (Markdown with timestamps format).
        use_ner: Whether to use spaCy NER for people extraction (default: True)
    Returns:
        Dictionary with metadata and list of tasks

    Side effects:
        Reads from disk. Does not modify the input file.
    """

    class ParentEntry(TypedDict):
        id: str
        indent: int

    # Read entire file content first
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    tasks = []
    current_section = ""
    current_section_date = ""
    task_id_counter = 1
    # Track parent tasks for hierarchy based on indentation.
    parent_stack: list[ParentEntry] = []

    # Parse the filtered content line-by-line
    lines = file_content.split('\n')
    for line_num, line in enumerate(lines, 1):
        line_stripped = line.rstrip("\n")

        # Skip empty lines and reset hierarchy.
        if not line_stripped:
            parent_stack = []
            continue

        # Check for section header that groups tasks by date.
        if line_stripped.startswith("# ⌛️"):
            # Extract date from header using a simple regex.
            header_match = re.match(r"# ⌛️\s+(.+?)\s+\(([^)]+)\)", line_stripped)
            if header_match:
                current_section = header_match.group(1)
                current_section_date = header_match.group(2)
            parent_stack = []
            continue

        # Check for task item lines (bullets, dashes, or checkboxes).
        if line_stripped and (
            line_stripped.lstrip().startswith("•")
            or line_stripped.lstrip().startswith("- [")
            or line_stripped.lstrip().startswith("-")
        ):

            # Calculate indent level to infer parent-child relationships.
            indent_level = len(line) - len(line.lstrip())
            indent_tabs = (
                indent_level // 8 if "\t" in line else 0
            )  # Assuming 8-space tabs

            # Extract timestamp that appears at the end of the line.
            timestamp_match = re.search(
                r"(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*[ap]m)\s*$",
                line_stripped,
                re.IGNORECASE,
            )

            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                timestamp_iso = parse_timestamp(timestamp_str)

                if timestamp_iso:
                    # Clean content for storage and indexing.
                    content_clean = clean_content(line_stripped)

                    # Generate stable ID from content hash (not sequential counter)
                    stable_id = compute_stable_id(
                        timestamp_iso, content_clean, current_section
                    )

                    # Determine parent task based on indent depth.
                    parent_id = None
                    if indent_tabs > 0 and parent_stack:
                        # Find parent at previous indent level
                        for i in range(len(parent_stack) - 1, -1, -1):
                            if parent_stack[i]["indent"] < indent_tabs:
                                parent_id = parent_stack[i]["id"]
                                break

                    # Build task object with extracted metadata.
                    task = {
                        "id": stable_id,
                        "line_number": line_num,
                        "timestamp": timestamp_iso,
                        "timestamp_raw": timestamp_str,
                        "section_header": current_section,
                        "section_date": current_section_date,
                        "content": content_clean,
                        "content_raw": line_stripped,
                        "indent_level": indent_tabs,
                        "parent_task_id": parent_id,
                        "bullet_type": get_bullet_type(line_stripped),
                        "is_completed": is_completed(line_stripped),
                        "is_pending": is_pending_checkbox(line_stripped),
                        "is_strikethrough": is_strikethrough(content_clean),
                        "links": extract_links(content_clean),
                        "tags": extract_tags(content_clean),
                        "people_mentioned": [],  # Will be filled by batch processing
                    }

                    tasks.append(task)

                    # Update parent stack so children can link to this task.
                    # Remove items at same or higher indent.
                    parent_stack = [
                        entry
                        for entry in parent_stack
                        if entry["indent"] < indent_tabs
                    ]
                    task_id = str(task["id"])
                    parent_stack.append({"id": task_id, "indent": indent_tabs})

                    task_id_counter += 1

    # Batch extract people after all tasks parsed (if using NER)
    if use_ner:
        try:
            # Extract all content texts and strip tags for better NER accuracy
            # Tags like #personal# interfere with spaCy's context analysis
            all_content = [
                strip_tags_for_ner(str(task.get("content", ""))) for task in tasks
            ]

            # Batch process with NER
            people_results = extract_people_batch_ner(all_content)

            # Assign back to tasks
            for task, people in zip(tasks, people_results):
                task["people_mentioned"] = people

        except ImportError:
            # Fall back to regex for each task
            print(
                "⚠️  NER not available, using regex extraction for people",
                file=sys.stderr,
            )
            for task in tasks:
                task["people_mentioned"] = extract_people(
                    str(task.get("content", "")), use_ner=False
                )
    else:
        # Regex extraction (use_ner=False)
        for task in tasks:
            task["people_mentioned"] = extract_people(
                str(task.get("content", "")), use_ner=False
            )

    # Compute a date range summary for quick metadata inspection.
    timestamps: list[str] = []
    for task in tasks:
        timestamp = task.get("timestamp")
        if isinstance(timestamp, str):
            timestamps.append(timestamp)
    earliest = min(timestamps) if timestamps else None
    latest = max(timestamps) if timestamps else None

    return {
        "metadata": {
            "source_file": str(file_path),
            "parsed_at": datetime.now().isoformat(),
            "total_tasks": len(tasks),
            "date_range": {"earliest": earliest, "latest": latest},
        },
        "tasks": tasks,
    }


def main():
    """
    CLI entry point for converting Twos Markdown to JSON.

    Returns:
        Exit code integer (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Convert Twos markdown export (with timestamps) to JSON"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to Twos markdown file (Markdown with timestamps format)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file path")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    parser.add_argument(
        "--no-ner",
        action="store_true",
        help="Disable NER extraction, use regex fallback (faster but less accurate)",
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    # Determine output path (defaults to input filename with .json extension).
    if args.output:
        output_path = args.output
    else:
        output_path = args.input_file.with_suffix(".json")

    print(f"Parsing {args.input_file}...")
    data = parse_twos_file(args.input_file, use_ner=not args.no_ner)

    print(f"Found {data['metadata']['total_tasks']} tasks")
    print(
        f"Date range: {data['metadata']['date_range']['earliest']} to {data['metadata']['date_range']['latest']}"
    )

    print(f"Writing to {output_path}...")
    # I/O boundary: write parsed data to JSON on disk.
    with open(output_path, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)

    print("Done!")

    # Print sample task
    if data["tasks"]:
        print("\nSample task:")
        print(json.dumps(data["tasks"][0], indent=2))

    return 0


if __name__ == "__main__":
    exit(main())
