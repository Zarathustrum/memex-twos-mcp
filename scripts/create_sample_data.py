#!/usr/bin/env python3
"""
Generate a sanitized sample Twos export for testing.

This creates a fake Markdown export file that mimics Twos formatting.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path


def generate_sample_export(
    output_path: Path, sections: int, items_per_section: int
) -> None:
    """
    Generate a sample Twos markdown export.

    Args:
        output_path: Where the sample Markdown file will be written.
        sections: Number of date sections to generate.
        items_per_section: Number of items per section.
    """
    start_date = datetime(2024, 1, 6, 9, 0)

    people = ["Alice", "Bob", "Carol", "Diana"]
    projects = ["Home Improvement", "Work Project", "Health Goals", "Family Plans"]
    tags = ["#work#", "#home#", "#health#", "#family#", "#travel#", "#finance#"]

    lines: list[str] = []

    for section_index in range(sections):
        section_date = start_date + timedelta(days=section_index * 7)
        # Header line matches the Twos export section format.
        header = section_date.strftime("# ⌛️ %a, %b %d, %Y (%m/%d/%y %I:%M %p)")
        lines.append(header)

        for item_index in range(items_per_section):
            task_time = section_date + timedelta(minutes=15 * item_index)
            # Rotate through example values to keep data varied.
            person = people[(section_index + item_index) % len(people)]
            project = projects[(section_index + item_index) % len(projects)]
            tag = tags[(section_index + item_index) % len(tags)]

            if item_index % 5 == 0:
                prefix = "- [ ]"
            elif item_index % 7 == 0:
                prefix = "- [x]"
            elif item_index % 3 == 0:
                prefix = "-"
            else:
                prefix = "•"

            # Build a task line with content and a trailing timestamp.
            task_text = (
                f"{prefix} {project}: check in with {person} {tag}"
                f" {task_time.strftime('%m/%d/%y %I:%M %p').lower()}"
            )
            lines.append(task_text)
        lines.append("")

    # I/O boundary: write the sample export file to disk.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """
    CLI entry point for generating sample data.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Generate sample Twos export data")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/sample/sample_export.md"),
        help="Output path for the sample export",
    )
    parser.add_argument(
        "--sections",
        type=int,
        default=12,
        help="Number of date sections to generate",
    )
    parser.add_argument(
        "--items-per-section",
        type=int,
        default=10,
        help="Number of tasks per section",
    )
    args = parser.parse_args()

    generate_sample_export(args.output, args.sections, args.items_per_section)
    print(f"Sample export written to {args.output}")


if __name__ == "__main__":
    main()
