#!/usr/bin/env python3
"""
Unified builder orchestrator for derived indices.

Coordinates execution of multiple index builders with proper dependency
management, error handling, and progress reporting.

Usage:
    # Run default builders (timepacks + threads)
    python scripts/build_all.py --db data/processed/twos.db

    # Include LLM-powered monthly summaries
    python scripts/build_all.py --db data/processed/twos.db --with-llm

    # Run specific builders only
    python scripts/build_all.py --db data/processed/twos.db --builders timepacks,threads

    # Force rebuild (ignore incremental)
    python scripts/build_all.py --db data/processed/twos.db --force
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ============================================================================
# Builder Configuration
# ============================================================================

BUILDER_ORDER = ["timepacks", "threads", "summaries"]

# Builder dependencies: key depends on all values in list
BUILDER_DEPENDENCIES = {
    "summaries": ["timepacks"]  # Monthly summaries require timepacks data
}

# Builder metadata for display
BUILDER_METADATA = {
    "timepacks": {
        "name": "TimePacks",
        "description": "day/week/month rollups",
        "module": "build_timepacks",
    },
    "threads": {
        "name": "ThreadPacks",
        "description": "tag/person activity indices",
        "module": "build_threads",
    },
    "summaries": {
        "name": "MonthlySummaries",
        "description": "LLM semantic summaries",
        "module": "build_month_summaries",
    },
}


# ============================================================================
# Core Orchestration Function
# ============================================================================


def build_derived_indices(
    db_path: Path,
    with_llm: bool = False,
    force: bool = False,
    builders: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build derived indices (rollups, threads, summaries).

    Orchestration strategy:
    - Runs builders in dependency order (timepacks → threads, summaries)
    - Partial success allowed: if TimePacks fails, ThreadPacks still runs
    - Dependency blocking: if TimePacks fails, MonthlySummaries is skipped
    - Each builder runs independently (isolated error handling)
    - Returns detailed status for each builder (success/failed + stats/errors)

    Error handling:
    - Builder failure doesn't stop execution (partial success)
    - Errors captured per-builder and returned in result dict
    - Exit code 0 only if ALL requested builders succeeded
    - Exit code 1 if any builder failed (even if others succeeded)

    Dependency chain:
    - TimePacks: no dependencies (runs first)
    - ThreadPacks: no dependencies (runs after TimePacks for simplicity)
    - MonthlySummaries: depends on TimePacks (skipped if TimePacks failed)

    Args:
        db_path: Path to SQLite database
        with_llm: Include LLM-powered monthly summaries (uses API quota)
        force: Force rebuild (ignore incremental src_hash check)
        builders: Optional list of builders to run (default: timepacks + threads)
        verbose: Print progress to stdout

    Returns:
        {
            "success": bool,  # True only if ALL requested builders succeeded
            "builders_run": ["timepacks", "threads"],
            "builders_succeeded": ["timepacks", "threads"],
            "builders_failed": [],
            "duration_seconds": 5.0,
            "stats": {
                "timepacks": {"rollup_count": 858, "day_count": 730, ...},
                "threads": {"thread_count": 127, "tag_threads": 45, ...}
            },
            "errors": {}  # builder_name → error message
        }
    """
    start_time = time.time()

    # Determine which builders to run
    if builders is None:
        builders_to_run = ["timepacks", "threads"]
        if with_llm:
            builders_to_run.append("summaries")
    else:
        # Validate builder names
        invalid = [b for b in builders if b not in BUILDER_METADATA]
        if invalid:
            return {
                "success": False,
                "builders_run": [],
                "builders_succeeded": [],
                "builders_failed": builders,
                "duration_seconds": 0.0,
                "stats": {},
                "errors": {"validation": f"Unknown builders: {', '.join(invalid)}"},
            }
        builders_to_run = builders

    # Validate database exists
    if not db_path.exists():
        return {
            "success": False,
            "builders_run": [],
            "builders_succeeded": [],
            "builders_failed": builders_to_run,
            "duration_seconds": 0.0,
            "stats": {},
            "errors": {"db": f"Database not found: {db_path}"},
        }

    # Import builder modules
    builder_funcs: Dict[str, Callable] = {}
    for builder_name in builders_to_run:
        try:
            module_name = BUILDER_METADATA[builder_name]["module"]
            # Dynamic import
            if builder_name == "timepacks":
                from build_timepacks import build  # type: ignore

                builder_funcs["timepacks"] = build
            elif builder_name == "threads":
                from build_threads import build  # type: ignore

                builder_funcs["threads"] = build
            elif builder_name == "summaries":
                from build_month_summaries import build  # type: ignore

                builder_funcs["summaries"] = build
        except ImportError as e:
            if verbose:
                print(
                    f"⚠️  Failed to import {builder_name} ({module_name}): {e}",
                    file=sys.stderr,
                )

    # Execute builders in dependency order
    results: Dict[str, Any] = {}
    succeeded: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []
    errors: Dict[str, str] = {}

    if verbose:
        print("\nBuilding derived indices...\n")

    for i, builder_name in enumerate(builders_to_run, 1):
        total = len(builders_to_run)
        meta = BUILDER_METADATA[builder_name]

        if verbose:
            print(f"[{i}/{total}] {meta['name']} ({meta['description']})...")

        # Check dependencies
        deps = BUILDER_DEPENDENCIES.get(builder_name, [])
        missing_deps = [dep for dep in deps if dep in failed]
        if missing_deps:
            skipped.append(builder_name)
            if verbose:
                dep_names = ", ".join(BUILDER_METADATA[d]["name"] for d in missing_deps)
                print(f"  ⏭️  Skipped (dependency {dep_names} failed)")
            continue

        # Check if builder loaded
        if builder_name not in builder_funcs:
            failed.append(builder_name)
            errors[builder_name] = "Failed to import builder module"
            if verbose:
                print("  ❌ Failed (not loaded)")
            continue

        # Run builder
        try:
            result = builder_funcs[builder_name](db_path=db_path, force=force)

            if result.get("success"):
                succeeded.append(builder_name)
                results[builder_name] = result.get("stats", {})
                duration = result.get("duration_seconds", 0.0)
                if verbose:
                    print(f"  ✅ Success in {duration:.1f}s")
                    # Print builder-specific stats if available
                    stats = result.get("stats", {})
                    if stats:
                        stats_str = ", ".join(
                            f"{k}={v}"
                            for k, v in stats.items()
                            if isinstance(v, (int, float))
                        )
                        if stats_str:
                            print(f"     {stats_str}")
            else:
                failed.append(builder_name)
                error_msg = result.get("error", "Unknown error")
                errors[builder_name] = error_msg
                if verbose:
                    print(f"  ❌ Failed: {error_msg}")

        except Exception as e:
            failed.append(builder_name)
            errors[builder_name] = str(e)
            if verbose:
                print("  ❌ Failed with exception:")
                traceback.print_exc()

    # Update metadata for successful builds
    if succeeded:
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

            # Update last build timestamp
            cur.execute(
                "INSERT OR REPLACE INTO metadata (key, value, updated_at) "
                "VALUES (?, ?, ?)",
                ("last_derived_build", timestamp, timestamp),
            )

            # Update version for each successful builder
            for builder_name in succeeded:
                version_key = f"{builder_name}_version"
                cur.execute(
                    "INSERT OR REPLACE INTO metadata (key, value, updated_at) "
                    "VALUES (?, ?, ?)",
                    (version_key, "1.0", timestamp),
                )

            conn.commit()
            conn.close()
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to update metadata: {e}", file=sys.stderr)

    duration = time.time() - start_time

    # Print summary
    if verbose:
        success_count = len(succeeded)
        total_requested = len(builders_to_run) - len(skipped)
        print(
            f"\nSummary: {success_count}/{total_requested} builders succeeded in {duration:.1f}s"
        )

        if skipped:
            print(f"Skipped: {len(skipped)} (dependency failures)")

    return {
        "success": len(failed) == 0 and len(succeeded) > 0,
        "builders_run": builders_to_run,
        "builders_succeeded": succeeded,
        "builders_failed": failed,
        "builders_skipped": skipped,
        "duration_seconds": duration,
        "stats": results,
        "errors": errors,
    }


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build derived indices (rollups, threads, summaries)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default builders (timepacks + threads)
  %(prog)s --db data/processed/twos.db

  # Include LLM-powered monthly summaries
  %(prog)s --db data/processed/twos.db --with-llm

  # Run specific builders only
  %(prog)s --db data/processed/twos.db --builders timepacks,threads

  # Force rebuild (ignore incremental)
  %(prog)s --db data/processed/twos.db --force
        """,
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/processed/twos.db"),
        help="Path to SQLite database (default: data/processed/twos.db)",
    )

    parser.add_argument(
        "--builders",
        type=str,
        help="Comma-separated list: timepacks,threads,summaries (default: timepacks,threads)",
    )

    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Include LLM-powered monthly summaries (uses Claude API quota)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild (ignore incremental)",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Parse builders list
    builders: Optional[List[str]] = None
    if args.builders:
        builders = [b.strip() for b in args.builders.split(",")]

    # Run orchestrator
    result = build_derived_indices(
        db_path=args.db,
        with_llm=args.with_llm,
        force=args.force,
        builders=builders,
        verbose=not args.quiet,
    )

    # Exit with appropriate code
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
