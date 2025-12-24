#!/usr/bin/env python3
"""
Interactive setup wizard for memex-twos-mcp.
Guides users through installation and configuration.
"""

from __future__ import annotations

import argparse
import os
import select
import shutil
import subprocess
import sys
from pathlib import Path


def print_header(text: str) -> None:
    """
    Print a section header for the console UI.

    Args:
        text: Title text to display.
    """
    print("\n" + "=" * 60)
    print(f"{text}")
    print("=" * 60 + "\n")


def input_with_timeout(prompt: str, default: str, timeout: int = 10) -> str:
    """
    Get user input with a timeout and default value.

    Args:
        prompt: The prompt to display
        default: Default value if timeout or empty input
        timeout: Seconds to wait before using default

    Returns:
        User input or default value
    """
    print(f"{prompt} [{default}]: ", end="", flush=True)

    # Platform-specific timeout handling.
    if sys.platform == "win32":
        # Windows doesn't support select on stdin, fall back to simple input
        try:
            response = input().strip()
            return response if response else default
        except EOFError:
            return default
    else:
        # Unix-like systems: use select for timeout on stdin.
        ready, _, _ = select.select([sys.stdin], [], [], timeout)

        if ready:
            response = sys.stdin.readline().strip()
            return response if response else default
        else:
            print(f"\n(timeout - using default: {default})")
            return default


def check_python_version() -> None:
    """
    Check Python version is 3.10+ and exit on failure.

    Returns:
        None. Exits the process on failure.
    """
    print_header("Step 1: Checking Python Version")

    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10+ required, you have {sys.version}")
        print("Please upgrade Python and try again.")
        sys.exit(1)

    print(f"OK: Python {sys.version_info.major}.{sys.version_info.minor} detected")


def setup_virtual_environment() -> None:
    """
    Create a virtual environment in .venv if requested.

    Uses the built-in `venv` module via a subprocess call.

    Returns:
        None.
    """
    print_header("Step 2: Virtual Environment")

    venv_path = Path.cwd() / ".venv"

    if venv_path.exists():
        print("OK: Virtual environment already exists")
        return

    response = input("Create virtual environment? (y/n): ")
    if response.lower() != "y":
        print("SKIP: Virtual environment creation")
        return

    print("Creating virtual environment...")
    # Side effect: writes files to the .venv directory.
    subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
    print("OK: Virtual environment created")
    print("\nActivate it with:")
    print("  source .venv/bin/activate  (Linux/Mac)")
    print("  .venv\\Scripts\\activate     (Windows)")


def install_dependencies() -> None:
    """
    Install Python dependencies via pip.

    Side effects:
        Downloads packages and installs them into the active environment.

    Returns:
        None.
    """
    print_header("Step 3: Installing Dependencies")

    response = input("Install dependencies now? (y/n): ")
    if response.lower() != "y":
        print("SKIP: Dependency installation")
        print("Run: pip install -e . later")
        return

    print("Installing dependencies...")
    # Uses pip to install the project in editable mode.
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    print("OK: Dependencies installed")


def setup_data_files(export_file: str | None) -> str | None:
    """
    Set up data directories and get Twos export.

    Returns:
        A string describing data type ("real" or "sample") or None if skipped.
    """
    print_header("Step 4: Data Setup")

    # Ensure expected directories exist for input and output data files.
    (Path.cwd() / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "data" / "processed").mkdir(parents=True, exist_ok=True)

    if export_file:
        export_path = Path(export_file)
        if export_path.exists():
            # I/O boundary: copy user's export file into the project data directory.
            shutil.copy(export_path, "data/raw/twos_export.md")
            print("OK: Export file copied to data/raw/twos_export.md")
            return "real"
        print(f"ERROR: File not found: {export_path}")
        return None

    print("Do you have a Twos export file? (Markdown with timestamps format)")
    print("In Twos app: Settings -> Export -> Markdown with timestamps")
    print("")
    print("1. Yes, I have an export file")
    print("2. No, use sample data for testing")
    print("3. Skip for now")

    choice = input("Choice (1/2/3): ").strip()

    if choice == "1":
        export_path = input("Path to your Twos export file: ").strip()
        if Path(export_path).exists():
            # I/O boundary: copy user's export file into the project data directory.
            shutil.copy(export_path, "data/raw/twos_export.md")
            print("OK: Export file copied to data/raw/twos_export.md")
            return "real"
        print(f"ERROR: File not found: {export_path}")
        return None

    if choice == "2":
        sample_path = Path("data/sample/sample_export.md")
        if sample_path.exists():
            # I/O boundary: copy sample data into place for testing.
            shutil.copy(sample_path, "data/raw/twos_export.md")
            print("OK: Sample data copied to data/raw/twos_export.md")
            return "sample"
        print("ERROR: Sample data not found. Run Phase 4 to add it.")
        return None

    print("SKIP: Data setup")
    return None


def convert_and_load_data(data_type: str | None) -> None:
    """
    Convert markdown to JSON and load to SQLite.

    This runs the converter, optional grooming, and the SQLite loader.

    Args:
        data_type: The chosen data type ("real", "sample", or None).

    Returns:
        None.
    """
    if not data_type:
        return

    print_header("Step 5: Converting Data")

    # Step 5a: Convert to JSON
    print("Converting Twos export to JSON...")
    # Side effect: writes a JSON file into data/processed.
    subprocess.run(
        [
            sys.executable,
            "src/convert_to_json.py",
            "data/raw/twos_export.md",
            "-o",
            "data/processed/twos_data.json",
            "--pretty",
        ],
        check=True,
    )
    print("OK: Conversion complete")

    # Step 5b: Optional grooming
    print("\n" + "-" * 60)
    print("Data Grooming")
    print("-" * 60)
    print("Grooming removes exact duplicates and fixes data quality issues.")
    print("Recommended for first-time setup and regular data refreshes.")
    print()

    response = input_with_timeout(
        "Run data grooming? (Y/n)",
        default="y",
        timeout=10
    )

    json_file = "data/processed/twos_data.json"  # Default: use original

    if response.lower() in ["y", "yes", ""]:
        # Run Python auto-fix
        print("\nGrooming data (removing duplicates, fixing broken refs)...")
        groom_args = [
            sys.executable,
            "scripts/groom_data.py",
            "data/processed/twos_data.json",
        ]

        # Ask about AI analysis
        print("\n" + "-" * 60)
        print("AI Semantic Analysis (Optional)")
        print("-" * 60)
        print("Provides pattern detection, theme categorization, and schema recommendations.")
        print("WARNING: Uses Claude Code subscription quota and takes 1-2 minutes.")
        print()

        ai_response = input_with_timeout(
            "Run AI semantic analysis? (y/N)",
            default="n",
            timeout=10
        )

        if ai_response.lower() in ["y", "yes"]:
            groom_args.append("--ai-analysis")
            print("\nRunning full grooming (Python + AI)...")
        else:
            print("\nRunning basic grooming (Python only)...")

        # Side effect: writes cleaned JSON and report files.
        subprocess.run(groom_args, check=True)
        print("OK: Data groomed")
        json_file = "data/processed/twos_data_cleaned.json"
        print(f"   Using cleaned data: {json_file}")

        # Step 5b2: Optional entity classification (only if AI was used)
        if ai_response.lower() in ["y", "yes"]:
            print("\n" + "-" * 60)
            print("Entity Classification (Optional)")
            print("-" * 60)
            print("Classifies 'people' into: person, place, project, verb, other.")
            print("Filters misclassifications (e.g., 'New', 'Put', 'Seattle').")
            print("Improves query accuracy for 'things with [person name]'.")
            print("Uses Claude Code quota, takes 2-3 minutes.")
            print()

            # No timeout here - user just waited 2-5 min for AI analysis
            # They need to make an explicit choice, not auto-default
            classify_response = input("Run entity classification? (y/N): ").strip().lower()

            if classify_response.lower() in ["y", "yes"]:
                print("\nClassifying and normalizing entities...")
                # Side effect: writes entity_mappings.json and normalized JSON
                subprocess.run([
                    sys.executable,
                    "scripts/classify_entities.py",
                    json_file,
                    "--ai-classify",
                    "--apply-mappings"
                ], check=True)
                json_file = "data/processed/twos_data_cleaned_normalized.json"
                print(f"OK: Entities classified and normalized")
                print(f"   Using normalized data: {json_file}")
            else:
                print("SKIP: Entity classification")
                print("   (You can run this later with: python scripts/classify_entities.py --ai-classify)")
    else:
        print("SKIP: Data grooming (using original data)")

    # Step 5c: Load to SQLite
    print(f"\nLoading data into SQLite...")
    # Side effect: creates/overwrites the SQLite database file.
    load_args = [
        sys.executable,
        "scripts/load_to_sqlite.py",
        json_file,
    ]
    db_path = Path("data/processed/twos.db")
    if db_path.exists():
        overwrite = input_with_timeout(
            "Database exists. Overwrite? (y/N)",
            default="n",
            timeout=10
        )
        if overwrite.lower() in ["y", "yes"]:
            load_args.append("--force")
        else:
            print("SKIP: Database load (existing database kept)")
            return

    subprocess.run(
        load_args,
        check=True,
    )
    print("OK: Database created")


def generate_mcp_config(claude_config_path: str | None) -> None:
    """
    Generate MCP configuration for Claude Desktop.

    This writes to Claude's local config file when confirmed.

    Args:
        claude_config_path: Optional explicit path to Claude's config file.

    Returns:
        None.
    """
    print_header("Step 6: MCP Configuration")

    response = input("Generate Claude Desktop configuration? (y/n): ")
    if response.lower() != "y":
        print("SKIP: MCP config generation")
        print("See MCP_SETUP.md for manual configuration")
        return

    print("Running config generator...")
    # Side effect: updates Claude Desktop configuration file on disk.
    env = None
    if claude_config_path:
        env = {**dict(os.environ), "MEMEX_CLAUDE_CONFIG": claude_config_path}
    subprocess.run(
        [sys.executable, "scripts/generate_mcp_config.py"],
        check=True,
        env=env,
    )


def test_server() -> None:
    """
    Test that server can import.

    This only checks import success; it does not start the server loop.

    Returns:
        None.
    """
    print_header("Step 7: Testing Server")

    print("Testing server import...")
    # Uses a short Python one-liner to ensure importability.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from memex_twos_mcp import server; print('OK: import succeeded')",
        ],
        check=True,
    )

    print("Testing database count script...")
    subprocess.run(
        [
            sys.executable,
            "scripts/db_count.py",
        ],
        check=True,
    )


def main() -> None:
    """
    Run the setup wizard and handle top-level errors.

    Returns:
        None. Exits the process on failure.
    """
    print("Memex Twos MCP Setup Wizard")
    print("Transform your Twos exports into a queryable knowledge base for Claude.")

    parser = argparse.ArgumentParser(description="Memex Twos MCP setup wizard")
    parser.add_argument(
        "--export-file",
        type=str,
        help="Path to a Twos export file (Markdown with timestamps format)",
    )
    parser.add_argument(
        "--claude-config",
        type=str,
        help="Path to Claude Desktop config file (overrides auto-detection)",
    )
    args = parser.parse_args()

    try:
        check_python_version()
        setup_virtual_environment()
        install_dependencies()
        data_type = setup_data_files(args.export_file)
        convert_and_load_data(data_type)
        generate_mcp_config(args.claude_config)
        test_server()

        print_header("Setup Complete")
        print("Next steps:")
        print("1. Restart Claude Desktop")
        print("2. Try asking: 'What is in my task database?'")
        print("3. See MCP_SETUP.md for more examples")
        print("\nHappy querying.")

    except KeyboardInterrupt:
        print("\n\nSetup interrupted")
        sys.exit(1)
    except Exception as exc:
        print(f"\n\nERROR: {exc}")
        print("Check MCP_SETUP.md for troubleshooting")
        sys.exit(1)


if __name__ == "__main__":
    main()
