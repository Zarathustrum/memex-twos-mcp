#!/usr/bin/env python3
"""
Validate memex-twos-mcp installation and configuration.

This script performs a series of checks for Python version, dependencies,
configuration, database contents, and optional Claude Desktop config.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path


def print_header(text: str) -> None:
    """
    Print a section header for readable console output.

    Args:
        text: Title text to display.
    """
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")


def check_python_version(errors: list[str]) -> None:
    """
    Check Python version is 3.10+.

    Args:
        errors: List that collects validation errors.

    Appends a message to errors if the requirement is not met.
    """
    print_header("Python Version")
    if sys.version_info < (3, 10):
        errors.append(f"Python 3.10+ required, found {sys.version}")
        print("ERROR: Python 3.10+ required")
    else:
        print(f"OK: Python {sys.version_info.major}.{sys.version_info.minor}")


def check_dependencies(errors: list[str]) -> None:
    """
    Check required dependencies are installed.

    Args:
        errors: List that collects validation errors.

    Uses importlib to detect whether packages are importable.
    """
    print_header("Dependencies")
    required = ["mcp", "yaml", "dateutil"]
    for dep in required:
        if importlib.util.find_spec(dep) is None:
            errors.append(f"Missing dependency: {dep}")
            print(f"ERROR: Missing dependency {dep}")
        else:
            print(f"OK: {dep}")


def check_config(errors: list[str]) -> Path | None:
    """
    Check configuration file and database path.

    Args:
        errors: List that collects validation errors.

    Returns:
        The configured database path or None if config import failed.
    """
    print_header("Configuration")
    try:
        from memex_twos_mcp.config import get_config
    except Exception as exc:  # pragma: no cover - import failure
        errors.append(f"Failed to import config: {exc}")
        print(f"ERROR: Failed to import config: {exc}")
        return None

    config = get_config()
    config_errors = config.validate()
    if config_errors:
        errors.extend(config_errors)
        for err in config_errors:
            print(f"ERROR: {err}")
    else:
        print(f"OK: Database path {config.db_path}")

    return config.db_path


def check_database(db_path: Path | None, errors: list[str]) -> None:
    """
    Check database exists and has data.

    Args:
        db_path: Path to the SQLite database or None.
        errors: List that collects validation errors.

    Uses sqlite3 to connect and run a simple COUNT query.
    """
    print_header("Database")
    if not db_path:
        print("SKIP: No database path available")
        return

    if not db_path.exists():
        errors.append(f"Database not found: {db_path}")
        print(f"ERROR: Database not found: {db_path}")
        return

    try:
        # I/O boundary: open the SQLite database file and read from it.
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM things")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"OK: things count = {count}")
        if count == 0:
            errors.append("Database contains zero things")
            print("ERROR: Database contains zero things")
    except sqlite3.Error as exc:
        # sqlite3.Error covers common issues like missing tables or locked DBs.
        errors.append(f"SQLite error: {exc}")
        print(f"ERROR: SQLite error: {exc}")


def check_server_import(errors: list[str]) -> None:
    """
    Check server module imports cleanly without executing it.

    Args:
        errors: List that collects validation errors.
    """
    print_header("Server Import")
    try:
        from memex_twos_mcp import server  # noqa: F401

        print("OK: memex_twos_mcp.server import")
    except Exception as exc:
        errors.append(f"Server import failed: {exc}")
        print(f"ERROR: Server import failed: {exc}")


def _is_wsl() -> bool:
    """Return True if running inside Windows Subsystem for Linux."""
    if sys.platform != "linux":
        return False
    try:
        osrelease = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
    except OSError:
        return False
    return "microsoft" in osrelease.lower()


def _claude_config_path() -> Path:
    """
    Return the default Claude Desktop config path for this OS.

    This mirrors Claude Desktop's per-OS config file locations and
    supports an override via the MEMEX_CLAUDE_CONFIG environment variable.

    Returns:
        The expected config file path for this platform.
    """
    override = os.getenv("MEMEX_CLAUDE_CONFIG")
    if override:
        return Path(override)

    if sys.platform.startswith("darwin"):
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if os.name == "nt":
        appdata = os.getenv("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(appdata) / "Claude" / "claude_desktop_config.json"

    if _is_wsl():
        wsl_user = os.getenv("WINUSER") or os.getenv("USER")
        if wsl_user:
            windows_path = (
                Path("/mnt/c")
                / "Users"
                / wsl_user
                / "AppData"
                / "Roaming"
                / "Claude"
                / "claude_desktop_config.json"
            )
            if windows_path.exists():
                return windows_path

    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def check_claude_config() -> None:
    """
    Check Claude Desktop config presence and basic JSON validity.

    This is optional and only warns if missing or malformed.

    Returns:
        None.
    """
    print_header("Claude Desktop Config")
    path = _claude_config_path()
    if path.exists():
        try:
            # I/O boundary: read JSON configuration from disk.
            json.loads(path.read_text(encoding="utf-8"))
            print(f"OK: Found config at {path}")
        except json.JSONDecodeError as exc:
            print(f"WARN: Config exists but is not valid JSON: {exc}")
    else:
        print(f"WARN: Config not found at {path}")


def main() -> None:
    """
    Run setup validation checks and exit with status 1 on failure.

    Returns:
        None. Exits the process on failure.
    """
    errors: list[str] = []

    check_python_version(errors)
    check_dependencies(errors)
    db_path = check_config(errors)
    check_database(db_path, errors)
    check_server_import(errors)
    check_claude_config()

    print_header("Summary")
    if errors:
        print("ERRORS FOUND:")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)

    print("All checks passed.")


if __name__ == "__main__":
    main()
