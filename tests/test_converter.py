"""Tests for the Twos markdown converter."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Allow importing from the local src/ directory without installing the package.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from convert_to_json import parse_twos_file  # noqa: E402


def test_parse_twos_file(tmp_path: Path) -> None:
    """
    Parser extracts tasks, tags, and people.

    Args:
        tmp_path: Pytest-provided temporary directory for file I/O.

    This test writes a small Markdown sample to a temp directory and
    asserts that the parsed JSON structure matches expectations.

    Returns:
        None.
    """
    content = (
        "# ⌛️ Mon, Jan 01, 2024 (01/01/24 08:00 am)\n"
        "• Meet Alice about plans #work# 01/01/24 09:00 am\n"
        "- [x] Review checklist 01/01/24 10:15 am\n"
    )

    # I/O boundary: write a temporary Markdown file for the parser.
    sample_path = tmp_path / "sample.md"
    sample_path.write_text(content, encoding="utf-8")

    data = parse_twos_file(sample_path)
    tasks = data["tasks"]

    assert len(tasks) == 2
    assert tasks[0]["content"] == "Meet Alice about plans #work#"
    assert "work" in tasks[0]["tags"]
    assert "Alice" in tasks[0]["people_mentioned"]
