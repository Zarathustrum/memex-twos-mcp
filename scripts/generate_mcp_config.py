#!/usr/bin/env python3
"""
Generate Claude Desktop MCP configuration for memex-twos-mcp.

This script updates the local Claude Desktop config JSON to register
the MCP server entry pointing at this project.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

from memex_twos_mcp.config import get_config


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
            # Return Windows Claude Desktop config path even if file doesn't exist yet
            # (the script will create it)
            return (
                Path("/mnt/c")
                / "Users"
                / wsl_user
                / "AppData"
                / "Roaming"
                / "Claude"
                / "claude_desktop_config.json"
            )

    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def build_server_config(use_wsl_command: bool) -> dict:
    """
    Build the MCP server config entry.

    Returns:
        A dict matching Claude Desktop's expected `mcpServers` schema.
    """
    config = get_config()
    project_root = config.project_root

    if use_wsl_command:
        wsl_command = (
            f"cd {shlex.quote(str(project_root))} && "
            "source .venv/bin/activate && "
            "PYTHONPATH=src python -m memex_twos_mcp.server"
        )
        return {
            "command": "wsl.exe",
            "args": ["bash", "-lc", wsl_command],
            "env": {},
        }

    # Claude Desktop will execute this command when starting the MCP server.
    # Use sys.executable to get the current Python interpreter (venv if activated)
    return {
        "command": sys.executable,
        "args": ["-m", "memex_twos_mcp.server"],
        "cwd": str(project_root),
    }


def load_existing_config(path: Path) -> dict:
    """
    Load existing Claude config JSON if present.

    Args:
        path: Path to the Claude Desktop config JSON file.

    Returns:
        Parsed JSON data with a guaranteed `mcpServers` key.
    """
    if not path.exists():
        return {"mcpServers": {}}

    # I/O boundary: read the config file from disk.
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if "mcpServers" not in data:
        data["mcpServers"] = {}
    return data


def write_config(path: Path, data: dict) -> None:
    """
    Write config to disk.

    Args:
        path: Path to write the config JSON to.
        data: Parsed config data to serialize.

    Side effects:
        Creates parent directories and overwrites the config file.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def backup_config(path: Path) -> Path | None:
    """
    Create timestamped backup of existing config file.

    Args:
        path: Path to the config file to back up.

    Returns:
        Path to the backup file if created, None if original doesn't exist.
    """
    if not path.exists():
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    backup_name = f"{path.stem}.backup.{timestamp}{path.suffix}"
    backup_path = path.parent / backup_name

    # Copy existing config to backup
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup_path


def main() -> None:
    """
    Generate or update Claude Desktop config.

    Returns:
        None. Exits with status 1 on JSON parsing errors.
    """
    parser = argparse.ArgumentParser(description="Generate Claude Desktop MCP config")
    parser.add_argument(
        "--print-path",
        action="store_true",
        help="Print the resolved Claude config path and exit",
    )
    args = parser.parse_args()

    target_path = _claude_config_path()
    print(f"Claude Desktop config: {target_path}")
    if args.print_path:
        return

    try:
        data = load_existing_config(target_path)
    except json.JSONDecodeError as exc:
        # JSONDecodeError indicates a malformed file that cannot be updated safely.
        print(f"ERROR: Existing config is invalid JSON: {exc}")
        sys.exit(1)

    use_wsl_command = _is_wsl() and target_path.parts[:3] == ("/", "mnt", "c")
    server_config = build_server_config(use_wsl_command)
    data["mcpServers"]["memex-twos-v2"] = server_config

    if target_path.exists():
        response = (
            input("Update existing config with memex-twos-v2? (y/n): ").strip().lower()
        )
        if response != "y":
            print("SKIP: No changes written")
            return

    # Backup existing config before modification
    backup_path = backup_config(target_path)
    if backup_path:
        print(f"Backup created: {backup_path}")

    # I/O boundary: write the updated config back to disk.
    write_config(target_path, data)
    print("OK: Configuration updated")

    if use_wsl_command:
        print("\nNOTE: Windows users must restart Claude Desktop for changes to take effect.")
        print("      If Claude Desktop appears closed but won't restart, you may need to")
        print("      kill the process in Task Manager (look for 'Claude' or 'claude.exe').")


if __name__ == "__main__":
    main()
