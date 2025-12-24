#!/usr/bin/env python3
"""
Print a minimal database health check payload.

Outputs the configured database path, total thing count, and any
available load metadata (source file and last load time).
"""

from __future__ import annotations

import json
import sys

from memex_twos_mcp.config import get_config
from memex_twos_mcp.database import TwosDatabase


def main() -> int:
    """CLI entry point for printing database count info."""
    config = get_config()
    errors = config.validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    db = TwosDatabase(config.db_path)
    info = db.get_count_info()
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
