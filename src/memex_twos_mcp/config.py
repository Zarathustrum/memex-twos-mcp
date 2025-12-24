"""
Configuration management for memex-twos-mcp.

Supports environment variables and a YAML config file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class MemexConfig:
    """Configuration for Memex Twos MCP server."""

    def __init__(self) -> None:
        """
        Build configuration values from defaults, environment variables,
        and an optional YAML config file.
        """
        self.project_root = self._find_project_root()

        self.db_path = Path(
            os.getenv(
                "MEMEX_DB_PATH",
                self.project_root / "data" / "processed" / "twos.db",
            )
        )

        self.config_file = Path(
            os.getenv(
                "MEMEX_CONFIG",
                Path.home() / ".memex-twos" / "config.yaml",
            )
        )

        if self.config_file.exists():
            self._load_config_file()

    def _find_project_root(self) -> Path:
        """
        Find project root directory by walking up for markers.

        This looks for `pyproject.toml` or `.git` to decide where the
        repository root is, then falls back to the current working directory.

        Returns:
            The detected project root path.
        """
        current = Path(__file__).resolve().parent

        while current != current.parent:
            if (current / "pyproject.toml").exists():
                return current
            if (current / ".git").exists():
                return current
            current = current.parent

        return Path.cwd()

    def _load_config_file(self) -> None:
        """
        Load configuration from a YAML file.

        Side effects:
            Reads from disk. Any parsed values override defaults.

        Returns:
            None.
        """
        with self.config_file.open("r", encoding="utf-8") as handle:
            config: Any = yaml.safe_load(handle)

        if not config:
            return

        database_config = config.get("database")
        if database_config and "path" in database_config:
            self.db_path = Path(database_config["path"])

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            A list of human-readable error strings; empty means valid.
        """
        errors: list[str] = []

        if not self.db_path.exists():
            errors.append(f"Database not found: {self.db_path}")
        elif not self.db_path.is_file():
            errors.append(f"Database path is not a file: {self.db_path}")

        return errors


_config: MemexConfig | None = None


def get_config() -> MemexConfig:
    """
    Get global configuration instance.

    This uses a module-level singleton so configuration is read once.
    """
    global _config
    if _config is None:
        _config = MemexConfig()
    return _config
