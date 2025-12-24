#!/usr/bin/env python3
"""
Entity Classification Script for Twos Data

Extracts entities (people, tags) from cleaned data, classifies them using AI,
and generates normalization mappings. Follows the same two-tier pattern as
groom_data.py: Python extraction + optional AI classification.

Usage:
    # Extract entities and show summary
    python3 scripts/classify_entities.py

    # Extract and classify using AI
    python3 scripts/classify_entities.py --ai-classify

    # Apply existing mappings to create normalized data
    python3 scripts/classify_entities.py --apply-mappings

    # Full workflow: classify then apply
    python3 scripts/classify_entities.py --ai-classify --apply-mappings
"""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def check_claude_code() -> bool:
    """
    Check if Claude Code CLI is installed.

    Returns:
        True if the `claude` executable is on PATH.
    """
    result = subprocess.run(
        ["which", "claude"], capture_output=True, text=True
    )
    return result.returncode == 0


def load_json_data(json_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load things data from JSON file, preserving metadata if present.

    Args:
        json_path: Path to the JSON file to load.

    Returns:
        A tuple of (things_list, metadata_dict).
    """
    print(f"📂 Loading data from {json_path}...")
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


def extract_entities(things: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract all unique entities (people, tags) with frequency counts.

    Args:
        things: List of thing dictionaries.

    Returns:
        Dictionary containing entity analysis:
        {
            "people": Counter of people_mentioned,
            "tags": Counter of tags,
            "people_lower_map": Dict mapping lowercase -> list of variants,
            "tags_lower_map": Dict mapping lowercase -> list of variants
        }
    """
    print("\n🔍 Extracting entities...")

    people_counter = Counter()
    tags_counter = Counter()

    # Count all entity occurrences
    for thing in things:
        for person in thing.get("people_mentioned", []):
            people_counter[person] += 1
        for tag in thing.get("tags", []):
            tags_counter[tag] += 1

    # Build case-variant maps for normalization detection
    people_lower_map = defaultdict(list)
    for person, count in people_counter.items():
        people_lower_map[person.lower()].append((person, count))

    tags_lower_map = defaultdict(list)
    for tag, count in tags_counter.items():
        tags_lower_map[tag.lower()].append((tag, count))

    print(f"  - Found {len(people_counter)} unique people (mentions: {sum(people_counter.values())})")
    print(f"  - Found {len(tags_counter)} unique tags (uses: {sum(tags_counter.values())})")
    print(f"  - Case variants: {sum(1 for v in people_lower_map.values() if len(v) > 1)} people, "
          f"{sum(1 for v in tags_lower_map.values() if len(v) > 1)} tags")

    return {
        "people": people_counter,
        "tags": tags_counter,
        "people_lower_map": people_lower_map,
        "tags_lower_map": tags_lower_map
    }


def generate_entity_summary(entities: Dict[str, Any], limit: int = 50) -> str:
    """
    Generate a markdown summary of extracted entities.

    Args:
        entities: Entity extraction results from extract_entities().
        limit: Number of top entities to show per category.

    Returns:
        Markdown formatted summary string.
    """
    people = entities["people"]
    tags = entities["tags"]
    people_lower_map = entities["people_lower_map"]
    tags_lower_map = entities["tags_lower_map"]

    report = f"""# Entity Extraction Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

- **Unique People:** {len(people):,} ({sum(people.values()):,} total mentions)
- **Unique Tags:** {len(tags):,} ({sum(tags.values()):,} total uses)
- **People with case variants:** {sum(1 for v in people_lower_map.values() if len(v) > 1)}
- **Tags with case variants:** {sum(1 for v in tags_lower_map.values() if len(v) > 1)}

---

## Top {limit} People (by mention count)

"""
    for person, count in people.most_common(limit):
        report += f"- **{person}**: {count:,} mentions\n"

    report += f"\n## Top {limit} Tags (by use count)\n\n"
    for tag, count in tags.most_common(limit):
        report += f"- **{tag}**: {count:,} uses\n"

    # Show case variants
    people_variants = [(lower, variants) for lower, variants in people_lower_map.items() if len(variants) > 1]
    if people_variants:
        report += f"\n## People Case Variants ({len(people_variants)} groups)\n\n"
        for lower, variants in sorted(people_variants, key=lambda x: sum(v[1] for v in x[1]), reverse=True)[:20]:
            variants_str = ", ".join([f"{v[0]} ({v[1]})" for v in sorted(variants, key=lambda x: x[1], reverse=True)])
            report += f"- {variants_str}\n"

    tags_variants = [(lower, variants) for lower, variants in tags_lower_map.items() if len(variants) > 1]
    if tags_variants:
        report += f"\n## Tag Case Variants ({len(tags_variants)} groups)\n\n"
        for lower, variants in sorted(tags_variants, key=lambda x: sum(v[1] for v in x[1]), reverse=True)[:20]:
            variants_str = ", ".join([f"{v[0]} ({v[1]})" for v in sorted(variants, key=lambda x: x[1], reverse=True)])
            report += f"- {variants_str}\n"

    report += """
---

## Next Steps

1. Review the extracted entities above
2. Run with `--ai-classify` to classify people (person/place/project/verb)
3. Review and edit `data/processed/entity_mappings.json` as needed
4. Run with `--apply-mappings` to create normalized data
5. Load normalized data to SQLite

"""
    return report


def invoke_claude_classification(
    entities: Dict[str, Any],
    output_path: Path
) -> bool:
    """
    Invoke Claude Code to classify entities and generate mapping file.

    Args:
        entities: Entity extraction results from extract_entities().
        output_path: Path where the mapping JSON should be written.

    Returns:
        True if classification succeeded and mapping file was created.
    """
    print("\n🤖 Invoking Claude Code for entity classification...")
    print("   (This will use your subscription quota)")

    # Prepare entity lists for Claude
    people_list = [{"name": person, "count": count}
                   for person, count in entities["people"].most_common(200)]
    tags_list = [{"tag": tag, "count": count}
                 for tag, count in entities["tags"].most_common(100)]

    # Build the classification prompt
    prompt = f"""I need you to classify entities extracted from personal task data.

YOUR TASK:

1. Classify the people listed below into categories:
   - "person" - Actual human names (Alice, Bob, Dr. Smith)
   - "place" - Locations (Seattle, Home, Office)
   - "project" - Project names or initiatives
   - "verb" - Action words misclassified as people (Put, Wake, New)
   - "other" - Everything else (typos, unclear)

2. For people with case variants, suggest a canonical form

3. Write a JSON mapping file to: {output_path}

## People to Classify (top 200 by frequency)

{json.dumps(people_list, indent=2)}

## Tags to Review (top 100)

{json.dumps(tags_list, indent=2)}

## Output Format

Write a JSON file with this structure:

```json
{{
  "metadata": {{
    "generated_at": "ISO timestamp",
    "total_people_classified": 200,
    "total_tags_reviewed": 100
  }},
  "people_classification": {{
    "Alice": {{"type": "person", "canonical": "Alice"}},
    "alice": {{"type": "person", "canonical": "Alice"}},
    "ALICE": {{"type": "person", "canonical": "Alice"}},
    "Rory": {{"type": "person", "canonical": "Rory"}},
    "New": {{"type": "verb", "canonical": null, "note": "Common word"}},
    "Put": {{"type": "verb", "canonical": null}},
    "Seattle": {{"type": "place", "canonical": "Seattle"}},
    "Gloucester": {{"type": "place", "canonical": "Gloucester"}}
  }},
  "tag_normalization": {{
    "siri": "siri",
    "Siri": "siri",
    "journal": "journal"
  }},
  "notes": "Any important observations about entity patterns"
}}
```

Focus on accuracy. When in doubt, mark as "other" rather than guessing.

Execute this task now without asking for confirmation. Write the complete mapping file.
"""

    # Invoke Claude Code CLI
    try:
        result = subprocess.run(
            [
                "claude",
                "--print",
                "--tools", "Write",
                "--allowedTools", "Write",
                "--dangerously-skip-permissions",
                "--no-session-persistence",
                prompt
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            stdin=subprocess.DEVNULL  # Signal no input coming
        )
    except subprocess.TimeoutExpired:
        print("❌ Claude Code timed out after 5 minutes")
        return False

    if result.returncode != 0:
        print(f"❌ Claude Code failed with exit code {result.returncode}")
        print(f"stderr: {result.stderr}")
        return False

    # Check if output was written
    if output_path.exists():
        print(f"✅ Entity classification complete: {output_path}")
        return True
    else:
        print("⚠️  Claude Code completed but output file not found")
        print("Response:")
        print(result.stdout[:500])
        return False


def apply_entity_mappings(
    things: List[Dict[str, Any]],
    mappings_path: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply entity mappings to create normalized data.

    Args:
        things: Original things list.
        mappings_path: Path to entity_mappings.json.

    Returns:
        Tuple of (normalized_things, stats_dict).
    """
    print(f"\n🔧 Applying entity mappings from {mappings_path}...")

    with open(mappings_path) as f:
        mappings = json.load(f)

    people_classification = mappings.get("people_classification", {})
    tag_normalization = mappings.get("tag_normalization", {})

    stats = {
        "things_processed": 0,
        "people_normalized": 0,
        "people_filtered": 0,  # Removed non-person entities
        "tags_normalized": 0
    }

    normalized_things = []

    for thing in things:
        normalized_thing = thing.copy()

        # Normalize people_mentioned - filter out non-person entities
        if thing.get("people_mentioned"):
            original_people = thing["people_mentioned"]
            normalized_people = []

            for person in original_people:
                classification = people_classification.get(person, {})
                entity_type = classification.get("type", "other")
                canonical = classification.get("canonical")

                # Only keep actual people
                if entity_type == "person" and canonical:
                    if canonical != person:
                        stats["people_normalized"] += 1
                    normalized_people.append(canonical)
                else:
                    # Filter out verbs, places, etc.
                    stats["people_filtered"] += 1

            # Deduplicate and sort
            normalized_thing["people_mentioned"] = sorted(list(set(normalized_people)))

        # Normalize tags
        if thing.get("tags"):
            original_tags = thing["tags"]
            normalized_tags = []

            for tag in original_tags:
                normalized_tag = tag_normalization.get(tag, tag)
                if normalized_tag != tag:
                    stats["tags_normalized"] += 1
                normalized_tags.append(normalized_tag)

            # Deduplicate and sort
            normalized_thing["tags"] = sorted(list(set(normalized_tags)))

        normalized_things.append(normalized_thing)
        stats["things_processed"] += 1

    print(f"  ✅ Processed {stats['things_processed']:,} things")
    print(f"  ✅ Normalized {stats['people_normalized']:,} people names")
    print(f"  ✅ Filtered {stats['people_filtered']:,} non-person entities")
    print(f"  ✅ Normalized {stats['tags_normalized']:,} tags")

    return normalized_things, stats


def main():
    """CLI entry point for entity classification."""
    parser = argparse.ArgumentParser(
        description="Extract and classify entities from Twos data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract entities and show summary
  python3 scripts/classify_entities.py

  # Extract and classify using AI
  python3 scripts/classify_entities.py --ai-classify

  # Apply existing mappings
  python3 scripts/classify_entities.py --apply-mappings

  # Full workflow
  python3 scripts/classify_entities.py --ai-classify --apply-mappings
"""
    )
    parser.add_argument(
        "json_file",
        type=Path,
        nargs="?",
        default=Path("data/processed/twos_data_cleaned.json"),
        help="Path to cleaned JSON data file (default: data/processed/twos_data_cleaned.json)"
    )
    parser.add_argument(
        "--ai-classify",
        action="store_true",
        help="Use Claude Code to classify entities (uses subscription quota)"
    )
    parser.add_argument(
        "--apply-mappings",
        action="store_true",
        help="Apply entity mappings to create normalized data"
    )
    parser.add_argument(
        "--entity-limit",
        type=int,
        default=50,
        help="Number of entities to show in summary report (default: 50)"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.json_file.exists():
        print(f"❌ File not found: {args.json_file}")
        sys.exit(1)

    # Check Claude Code if AI classification requested
    if args.ai_classify:
        if not check_claude_code():
            print("❌ Claude Code CLI not found")
            print("\nInstall it first:")
            print("https://code.claude.com/docs/en/quickstart")
            sys.exit(1)
        print("✅ Claude Code CLI found")

    # Load data
    things, metadata = load_json_data(args.json_file)

    # Extract entities
    entities = extract_entities(things)

    # Generate and save entity summary
    reports_dir = Path("docs/entity-reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")
    summary_path = reports_dir / f"{timestamp}-entity-summary.md"

    summary = generate_entity_summary(entities, args.entity_limit)
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"\n📊 Entity summary saved: {summary_path}")

    # AI classification (optional)
    mappings_path = Path("data/processed/entity_mappings.json")

    if args.ai_classify:
        success = invoke_claude_classification(entities, mappings_path)
        if not success:
            print("\n⚠️  AI classification failed")
            sys.exit(1)
        print(f"\n✅ Entity mappings created: {mappings_path}")
        print("📝 Review and edit the mappings file before applying")

    # Apply mappings (optional)
    if args.apply_mappings:
        if not mappings_path.exists():
            print(f"❌ Mappings file not found: {mappings_path}")
            print("Run with --ai-classify first, or create the file manually")
            sys.exit(1)

        normalized_things, stats = apply_entity_mappings(things, mappings_path)

        # Save normalized data
        normalized_path = args.json_file.parent / (args.json_file.stem + "_normalized.json")

        # Preserve original structure
        if metadata:
            output_data = {
                "metadata": {
                    **metadata,
                    "normalized_at": datetime.now().isoformat(),
                    "normalization_stats": stats
                },
                "tasks": normalized_things
            }
        else:
            output_data = normalized_things

        with open(normalized_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Normalized data saved: {normalized_path}")
        print(f"\n📁 Next step: Load normalized data into SQLite:")
        print(f"   python3 scripts/load_to_sqlite.py {normalized_path}")

    elif not args.ai_classify:
        # Just extracted entities, no AI or mapping
        print("\n💡 Next steps:")
        print("  1. Review the entity summary above")
        print("  2. Run with --ai-classify to classify entities")
        print("  3. Review and edit entity_mappings.json")
        print("  4. Run with --apply-mappings to create normalized data")

    print("\n🎉 Entity classification workflow complete!")


if __name__ == "__main__":
    main()
