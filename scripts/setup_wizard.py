#!/usr/bin/env python3
"""
Interactive setup wizard for memex-twos-mcp.
Guides users through installation and configuration with professional console UX.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Rich imports with graceful degradation
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ============================================================================
# Configuration Data Structure
# ============================================================================


@dataclass
class StageResult:
    """Result from executing a stage."""

    success: bool
    duration: float
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class WizardConfig:
    """Complete configuration for the setup wizard run."""

    # Interaction mode
    non_interactive: bool = False
    verbose: bool = False
    no_color: bool = False

    # Stage control flags
    create_venv: bool = False
    install_deps: bool = True
    run_grooming: bool = True
    run_ai_analysis: bool = False
    run_entity_classification: bool = False
    overwrite_db: bool = True
    build_derived_indices: bool = True
    build_with_llm: bool = False
    generate_mcp_config: bool = True

    # Input/output paths
    export_file: str | None = None
    data_mode: str = "skip"  # "real", "sample", "skip"
    claude_config_path: str | None = None

    # Results tracking (populated during execution)
    json_file: str = "data/processed/twos_data.json"
    db_path: str = "data/processed/twos.db"
    stage_results: dict[str, StageResult] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)


# ============================================================================
# Console Rendering Helpers
# ============================================================================


class ConsoleRenderer:
    """Handles all console output with Rich or fallback rendering."""

    def __init__(self, config: WizardConfig):
        self.config = config
        self.use_rich = RICH_AVAILABLE and not config.no_color and sys.stdout.isatty()
        if self.use_rich:
            self.console = Console(no_color=config.no_color)
        else:
            self.console = None

    def banner(self, stage_num: int, title: str, total_stages: int = 15) -> None:
        """Print a stage banner."""
        if self.use_rich:
            header = f"[bold cyan]{stage_num:02d} / {total_stages:02d}[/bold cyan]  [bold white]{title}[/bold white]"
            self.console.print()
            self.console.print(Panel(header, border_style="cyan", padding=(0, 2)))
        else:
            print(f"\n{'=' * 70}")
            print(f"  {stage_num:02d} / {total_stages:02d}  {title}")
            print(f"{'=' * 70}\n")

    def status(
        self, level: str, message: str, detail: str = "", indent: int = 0
    ) -> None:
        """Print a status line (OK, WARN, FAIL, INFO)."""
        prefix = " " * (indent * 2)

        if self.use_rich:
            colors = {
                "OK": "green",
                "WARN": "yellow",
                "FAIL": "red",
                "INFO": "blue",
                "SKIP": "dim",
            }
            color = colors.get(level, "white")
            text = Text()
            text.append(prefix)
            text.append(f"[{level:4s}] ", style=f"bold {color}")
            text.append(message)
            if detail:
                text.append(f"\n{prefix}      {detail}", style="dim")
            self.console.print(text)
        else:
            print(f"{prefix}[{level:4s}] {message}")
            if detail:
                print(f"{prefix}      {detail}")

    def rule(self, text: str = "") -> None:
        """Print a horizontal rule."""
        if self.use_rich:
            self.console.print(Rule(text, style="dim"))
        else:
            if text:
                print(f"\n{'-' * 35} {text} {'-' * 35}\n")
            else:
                print(f"{'-' * 70}")

    def summary_table(self, config: WizardConfig) -> None:
        """Print final summary with metrics and timing."""
        elapsed = time.time() - config.start_time

        if self.use_rich:
            # Summary panel
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Key", style="bold cyan", width=25)
            table.add_column("Value", style="white")

            # Configuration
            table.add_row("Mode", "Non-interactive" if config.non_interactive else "Interactive")
            table.add_row("Data source", config.data_mode.title())
            table.add_row("Export file", config.export_file or "N/A")

            # Outputs
            table.add_row("", "")  # Spacer
            table.add_row("JSON output", config.json_file)
            table.add_row("Database", config.db_path)

            # Options selected
            table.add_row("", "")  # Spacer
            table.add_row("Grooming", "Yes" if config.run_grooming else "No")
            table.add_row("AI analysis", "Yes" if config.run_ai_analysis else "No")
            table.add_row(
                "Entity classification", "Yes" if config.run_entity_classification else "No"
            )
            table.add_row("MCP config", "Yes" if config.generate_mcp_config else "No")

            # Metrics from stages
            total_things = 0
            total_people = 0
            total_tags = 0
            duplicates_removed = 0

            for stage_name, result in config.stage_results.items():
                if "total_things" in result.metrics:
                    total_things = result.metrics["total_things"]
                if "total_people" in result.metrics:
                    total_people = result.metrics["total_people"]
                if "total_tags" in result.metrics:
                    total_tags = result.metrics["total_tags"]
                if "duplicates_removed" in result.metrics:
                    duplicates_removed = result.metrics["duplicates_removed"]

            if total_things > 0:
                table.add_row("", "")  # Spacer
                table.add_row("Things", f"{total_things:,}")
                table.add_row("People", f"{total_people:,}")
                table.add_row("Tags", f"{total_tags:,}")
                if duplicates_removed > 0:
                    table.add_row("Duplicates removed", f"{duplicates_removed:,}")

            # Timing
            table.add_row("", "")  # Spacer
            table.add_row("Total elapsed", f"{elapsed:.1f}s")

            # Stage timing
            for stage_name, result in config.stage_results.items():
                if result.success and result.duration > 0:
                    table.add_row(
                        f"  {stage_name}", f"{result.duration:.1f}s", style="dim"
                    )

            self.console.print()
            self.console.print(Panel(table, title="[bold white]Setup Summary[/bold white]", border_style="green", padding=(1, 2)))
            self.console.print()

        else:
            # Fallback plain text summary
            print("\n" + "=" * 70)
            print("  SETUP SUMMARY")
            print("=" * 70)
            print(f"  Mode: {'Non-interactive' if config.non_interactive else 'Interactive'}")
            print(f"  Data source: {config.data_mode.title()}")
            print(f"  Export file: {config.export_file or 'N/A'}")
            print()
            print(f"  JSON output: {config.json_file}")
            print(f"  Database: {config.db_path}")
            print()
            print(f"  Grooming: {'Yes' if config.run_grooming else 'No'}")
            print(f"  AI analysis: {'Yes' if config.run_ai_analysis else 'No'}")
            print(f"  Entity classification: {'Yes' if config.run_entity_classification else 'No'}")
            print(f"  MCP config: {'Yes' if config.generate_mcp_config else 'No'}")
            print()
            print(f"  Total elapsed: {elapsed:.1f}s")
            print("=" * 70 + "\n")

    def spinner(self, message: str):
        """Return a context manager for a spinner (if Rich available)."""
        if self.use_rich:
            return self.console.status(f"[bold blue]{message}...", spinner="dots")
        else:
            # Fallback: just print the message
            class FakeSpinner:
                def __enter__(self):
                    print(f"{message}...")
                    return self

                def __exit__(self, *args):
                    pass

            return FakeSpinner()


# ============================================================================
# User Input Helpers
# ============================================================================


def prompt_yes_no(
    renderer: ConsoleRenderer, question: str, default: bool = False
) -> bool:
    """Prompt user for yes/no answer."""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{question} ({default_str}): ").strip().lower()

    if not response:
        return default

    return response in ["y", "yes"]


# ============================================================================
# Configuration Stage
# ============================================================================


def collect_configuration(
    renderer: ConsoleRenderer, args: argparse.Namespace
) -> WizardConfig:
    """Collect all configuration choices upfront."""
    renderer.banner(1, "CONFIGURATION", total_stages=14)

    config = WizardConfig(
        non_interactive=args.yes or args.non_interactive,
        verbose=args.verbose,
        no_color=args.no_color,
        export_file=args.export_file,
        claude_config_path=args.claude_config,
    )

    # If non-interactive, apply CLI flag defaults
    if config.non_interactive:
        renderer.status("INFO", "Running in non-interactive mode")
        config.create_venv = args.create_venv
        config.install_deps = args.install_deps
        config.run_grooming = args.groom
        config.run_ai_analysis = args.ai_analysis
        config.run_entity_classification = args.classify_entities
        config.overwrite_db = args.overwrite_db
        config.generate_mcp_config = args.generate_mcp_config

        # Determine data mode
        if args.export_file:
            config.data_mode = "real"
        elif args.use_sample:
            config.data_mode = "sample"
        else:
            config.data_mode = "skip"

        renderer.status("OK", "Configuration loaded from CLI flags")
        return config

    # Interactive mode: ask questions
    renderer.status("INFO", "Interactive mode - configure your setup")
    print()

    # Virtual environment
    venv_path = Path.cwd() / ".venv"
    if not venv_path.exists():
        config.create_venv = prompt_yes_no(
            renderer, "Create virtual environment?", default=True
        )

    # Dependencies
    config.install_deps = prompt_yes_no(
        renderer, "Install dependencies?", default=True
    )

    # Data setup
    if not args.export_file:
        print("\nData Setup:")
        print("  1. I have a Twos export file")
        print("  2. Use sample data for testing")
        print("  3. Skip data setup")
        choice = input("Choice (1/2/3) [1]: ").strip() or "1"

        if choice == "1":
            export_path = input("Path to Twos export file: ").strip()
            if export_path and Path(export_path).exists():
                config.export_file = export_path
                config.data_mode = "real"
            else:
                renderer.status("WARN", "File not found, skipping data setup")
                config.data_mode = "skip"
        elif choice == "2":
            config.data_mode = "sample"
        else:
            config.data_mode = "skip"
    else:
        config.data_mode = "real" if Path(args.export_file).exists() else "skip"

    # Only ask about processing if we have data
    if config.data_mode != "skip":
        print()
        config.run_grooming = prompt_yes_no(
            renderer, "Run data grooming (remove duplicates, fix issues)?", default=True
        )

        if config.run_grooming:
            config.run_ai_analysis = prompt_yes_no(
                renderer,
                "Run AI semantic analysis (Developer/uses Claude Code subscription)?",
                default=False,
            )

        # Entity classification is independent of AI semantic analysis
        config.run_entity_classification = prompt_yes_no(
            renderer,
            "Run entity classification (filter misclassified entities, uses Claude Code)?",
            default=False,
        )

        config.build_derived_indices = prompt_yes_no(
            renderer,
            "Build derived indices (rollups, threads - recommended for analysis)?",
            default=True,
        )

        if config.build_derived_indices:
            config.build_with_llm = prompt_yes_no(
                renderer,
                "Include LLM-powered monthly summaries (uses Claude Code subscription)?",
                default=False,
            )

        db_path = Path(config.db_path)
        if db_path.exists():
            config.overwrite_db = prompt_yes_no(
                renderer, "Database exists. Overwrite?", default=True
            )

    # MCP config
    print()
    config.generate_mcp_config = prompt_yes_no(
        renderer, "Generate Claude Desktop MCP configuration?", default=True
    )

    # Show confirmation
    print()
    renderer.rule("Configuration Summary")
    print(f"  Virtual environment: {'Create' if config.create_venv else 'Skip'}")
    print(f"  Dependencies: {'Install' if config.install_deps else 'Skip'}")
    print(f"  Data mode: {config.data_mode.title()}")
    if config.data_mode != "skip":
        print(f"  Grooming: {'Yes' if config.run_grooming else 'No'}")
        print(f"  AI analysis: {'Yes' if config.run_ai_analysis else 'No'}")
        print(f"  Entity classification: {'Yes' if config.run_entity_classification else 'No'}")
        print(f"  Derived indices: {'Yes' if config.build_derived_indices else 'No'}")
        if config.build_derived_indices:
            print(f"  With LLM summaries: {'Yes' if config.build_with_llm else 'No'}")
    print(f"  MCP config: {'Yes' if config.generate_mcp_config else 'No'}")
    renderer.rule()
    print()

    if not prompt_yes_no(renderer, "Proceed with this configuration?", default=True):
        renderer.status("INFO", "Setup cancelled by user")
        sys.exit(0)

    return config


# ============================================================================
# Stage Implementations
# ============================================================================


def stage_check_python(
    renderer: ConsoleRenderer, config: WizardConfig
) -> StageResult:
    """Check Python version."""
    renderer.banner(2, "ENVIRONMENT CHECK")
    start = time.time()

    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    if sys.version_info < (3, 10):
        renderer.status("FAIL", f"Python 3.10+ required, you have {py_version}")
        return StageResult(success=False, duration=time.time() - start)

    renderer.status("OK", f"Python {py_version} detected")
    return StageResult(success=True, duration=time.time() - start)


def stage_setup_venv(renderer: ConsoleRenderer, config: WizardConfig) -> StageResult:
    """Create virtual environment if requested."""
    renderer.banner(3, "VIRTUAL ENVIRONMENT")
    start = time.time()

    venv_path = Path.cwd() / ".venv"

    if venv_path.exists():
        renderer.status("OK", "Virtual environment already exists")
        return StageResult(success=True, duration=time.time() - start)

    if not config.create_venv:
        renderer.status("SKIP", "Virtual environment creation")
        return StageResult(success=True, duration=time.time() - start)

    try:
        with renderer.spinner("Creating virtual environment"):
            subprocess.run(
                [sys.executable, "-m", "venv", ".venv"],
                check=True,
                capture_output=not config.verbose,
            )
        renderer.status("OK", "Virtual environment created")
        renderer.status(
            "INFO",
            "Activate with:",
            detail="source .venv/bin/activate  (Linux/Mac)\n.venv\\Scripts\\activate     (Windows)",
        )
        return StageResult(success=True, duration=time.time() - start)
    except subprocess.CalledProcessError as e:
        renderer.status("FAIL", f"Failed to create virtual environment: {e}")
        return StageResult(success=False, duration=time.time() - start)


def stage_install_deps(
    renderer: ConsoleRenderer, config: WizardConfig
) -> StageResult:
    """Install Python dependencies."""
    renderer.banner(4, "DEPENDENCIES")
    start = time.time()

    if not config.install_deps:
        renderer.status("SKIP", "Dependency installation")
        renderer.status("INFO", "Run manually: pip install -e .")
        return StageResult(success=True, duration=time.time() - start)

    try:
        with renderer.spinner("Installing dependencies (this may take 1-2 minutes)"):
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                check=True,
                capture_output=not config.verbose,
                text=True,
            )
            if config.verbose and result.stdout:
                print(result.stdout)

        renderer.status("OK", "Dependencies installed")
        return StageResult(success=True, duration=time.time() - start)
    except subprocess.CalledProcessError as e:
        renderer.status("FAIL", f"Failed to install dependencies: {e}")
        if e.stderr and config.verbose:
            print(e.stderr)
        return StageResult(success=False, duration=time.time() - start)


def stage_data_setup(renderer: ConsoleRenderer, config: WizardConfig) -> StageResult:
    """Set up data files."""
    renderer.banner(5, "DATA INGEST")
    start = time.time()

    # Ensure directories exist
    (Path.cwd() / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "data" / "processed").mkdir(parents=True, exist_ok=True)

    if config.data_mode == "skip":
        renderer.status("SKIP", "Data setup")
        return StageResult(success=True, duration=time.time() - start)

    try:
        if config.data_mode == "real":
            export_path = Path(config.export_file)
            if not export_path.exists():
                renderer.status("FAIL", f"Export file not found: {export_path}")
                return StageResult(success=False, duration=time.time() - start)

            shutil.copy(export_path, "data/raw/twos_export.md")
            renderer.status(
                "OK", "Export file copied", detail="data/raw/twos_export.md"
            )

        elif config.data_mode == "sample":
            sample_path = Path("data/sample/sample_export.md")
            if not sample_path.exists():
                renderer.status("FAIL", "Sample data not found")
                return StageResult(success=False, duration=time.time() - start)

            shutil.copy(sample_path, "data/raw/twos_export.md")
            renderer.status("OK", "Sample data copied", detail="data/raw/twos_export.md")

        return StageResult(success=True, duration=time.time() - start)

    except Exception as e:
        renderer.status("FAIL", f"Data setup failed: {e}")
        return StageResult(success=False, duration=time.time() - start)


def stage_convert(renderer: ConsoleRenderer, config: WizardConfig) -> StageResult:
    """Convert markdown to JSON."""
    renderer.banner(6, "CONVERSION")
    start = time.time()

    if config.data_mode == "skip":
        renderer.status("SKIP", "No data to convert")
        return StageResult(success=True, duration=time.time() - start)

    try:
        with renderer.spinner("Converting Twos export to JSON"):
            result = subprocess.run(
                [
                    sys.executable,
                    "src/convert_to_json.py",
                    "data/raw/twos_export.md",
                    "-o",
                    "data/processed/twos_data.json",
                    "--pretty",
                ],
                check=True,
                capture_output=not config.verbose,
                text=True,
            )
            if config.verbose and result.stdout:
                print(result.stdout)

        renderer.status("OK", "Conversion complete", detail=config.json_file)
        return StageResult(success=True, duration=time.time() - start)

    except subprocess.CalledProcessError as e:
        renderer.status("FAIL", f"Conversion failed: {e}")
        if e.stderr and config.verbose:
            print(e.stderr)
        return StageResult(success=False, duration=time.time() - start)


def stage_grooming(renderer: ConsoleRenderer, config: WizardConfig) -> StageResult:
    """Run data grooming (optional)."""
    renderer.banner(7, "GROOMING (OPTIONAL)")
    start = time.time()

    if not config.run_grooming or config.data_mode == "skip":
        renderer.status("SKIP", "Data grooming")
        return StageResult(success=True, duration=time.time() - start)

    try:
        groom_args = [
            sys.executable,
            "scripts/groom_data.py",
            config.json_file,
        ]

        if config.run_ai_analysis:
            groom_args.append("--ai-analysis")
            with renderer.spinner(
                "Running grooming (Python + AI analysis - may take 2-3 minutes)"
            ):
                result = subprocess.run(
                    groom_args,
                    check=True,
                    capture_output=not config.verbose,
                    text=True,
                )
                if config.verbose and result.stdout:
                    print(result.stdout)
            renderer.status("OK", "Grooming complete (with AI analysis)")
        else:
            with renderer.spinner("Running grooming (Python auto-fix)"):
                result = subprocess.run(
                    groom_args,
                    check=True,
                    capture_output=not config.verbose,
                    text=True,
                )
                if config.verbose and result.stdout:
                    print(result.stdout)
            renderer.status("OK", "Grooming complete (auto-fix only)")

        # Update JSON file path to use cleaned version
        config.json_file = "data/processed/twos_data_cleaned.json"
        renderer.status("INFO", "Using cleaned data", detail=config.json_file)

        # TODO: Parse grooming output for duplicates_removed metric
        return StageResult(
            success=True,
            duration=time.time() - start,
            metrics={"duplicates_removed": 0},  # Could parse from groom output
        )

    except subprocess.CalledProcessError as e:
        renderer.status("FAIL", f"Grooming failed: {e}")
        if e.stderr and config.verbose:
            print(e.stderr)
        return StageResult(success=False, duration=time.time() - start)


def stage_entity_classification(
    renderer: ConsoleRenderer, config: WizardConfig
) -> StageResult:
    """Run entity classification (optional)."""
    renderer.banner(8, "ENTITY CLASSIFICATION (OPTIONAL)")
    start = time.time()

    if not config.run_entity_classification or config.data_mode == "skip":
        renderer.status("SKIP", "Entity classification")
        return StageResult(success=True, duration=time.time() - start)

    try:
        with renderer.spinner(
            "Classifying entities (may take 2-3 minutes)"
        ):
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/classify_entities.py",
                    config.json_file,
                    "--ai-classify",
                    "--apply-mappings",
                ],
                check=True,
                capture_output=not config.verbose,
                text=True,
            )
            if config.verbose and result.stdout:
                print(result.stdout)

        # Update JSON file path to use normalized version
        config.json_file = "data/processed/twos_data_cleaned_normalized.json"
        renderer.status("OK", "Entity classification complete")
        renderer.status("INFO", "Using normalized data", detail=config.json_file)
        return StageResult(success=True, duration=time.time() - start)

    except subprocess.CalledProcessError as e:
        renderer.status("FAIL", f"Entity classification failed: {e}")
        if e.stderr and config.verbose:
            print(e.stderr)
        return StageResult(success=False, duration=time.time() - start)


def stage_load_sqlite(renderer: ConsoleRenderer, config: WizardConfig) -> StageResult:
    """Load data into SQLite."""
    renderer.banner(9, "SQLITE LOAD")
    start = time.time()

    if config.data_mode == "skip":
        renderer.status("SKIP", "No data to load")
        return StageResult(success=True, duration=time.time() - start)

    try:
        load_args = [
            sys.executable,
            "scripts/load_to_sqlite.py",
            config.json_file,
        ]

        db_path = Path(config.db_path)
        if db_path.exists() and config.overwrite_db:
            load_args.append("--force")

        with renderer.spinner("Loading data into SQLite database"):
            result = subprocess.run(
                load_args,
                check=True,
                capture_output=not config.verbose,
                text=True,
            )
            if config.verbose and result.stdout:
                print(result.stdout)

        renderer.status("OK", "Database created", detail=config.db_path)
        return StageResult(success=True, duration=time.time() - start)

    except subprocess.CalledProcessError as e:
        renderer.status("FAIL", f"Database load failed: {e}")
        if e.stderr and config.verbose:
            print(e.stderr)
        return StageResult(success=False, duration=time.time() - start)


def stage_embeddings(renderer: ConsoleRenderer, config: WizardConfig) -> StageResult:
    """Generate embeddings (part of load_to_sqlite)."""
    renderer.banner(10, "EMBEDDINGS")
    start = time.time()

    if config.data_mode == "skip":
        renderer.status("SKIP", "No data for embeddings")
        return StageResult(success=True, duration=time.time() - start)

    # Embeddings are generated during load_to_sqlite, so this is just a status check
    renderer.status(
        "INFO",
        "Embeddings generated during database load",
        detail="384-dimensional vectors for hybrid search",
    )
    return StageResult(success=True, duration=time.time() - start)


def stage_derived_indices(renderer: ConsoleRenderer, config: WizardConfig) -> StageResult:
    """Build derived indices (rollups, threads, optional LLM summaries)."""
    renderer.banner(11, "DERIVED INDICES (OPTIONAL)")
    start = time.time()

    if config.data_mode == "skip" or not config.build_derived_indices:
        renderer.status("SKIP", "Derived indices")
        return StageResult(success=True, duration=time.time() - start)

    try:
        # Import the orchestrator function
        sys.path.insert(0, str(Path(__file__).parent))
        from build_all import build_derived_indices

        db_path = Path(config.db_path)

        # Run orchestrator
        with renderer.spinner(
            "Building derived indices (timepacks, threads"
            + (", summaries" if config.build_with_llm else "")
            + ")..."
        ):
            result = build_derived_indices(
                db_path=db_path,
                with_llm=config.build_with_llm,
                force=False,  # Never force in wizard
                verbose=config.verbose,
            )

        if result["success"]:
            succeeded = result["builders_succeeded"]
            duration = result["duration_seconds"]

            # Format stats for display
            stats_parts = []
            if "timepacks" in succeeded:
                stats_parts.append("rollups")
            if "threads" in succeeded:
                stats_parts.append("threads")
            if "summaries" in succeeded:
                stats_parts.append("summaries")

            stats_str = ", ".join(stats_parts)
            renderer.status("OK", f"Built {stats_str} in {duration:.1f}s")

            # Store metrics
            metrics = {}
            for builder, stats in result.get("stats", {}).items():
                for key, value in stats.items():
                    metrics[f"{builder}_{key}"] = value

            return StageResult(
                success=True, duration=time.time() - start, metrics=metrics
            )
        else:
            # Partial success is OK - some builders may have failed
            failed = result["builders_failed"]
            errors = result["errors"]
            error_msg = "; ".join(f"{b}: {errors.get(b, 'unknown')}" for b in failed)

            if result["builders_succeeded"]:
                # Some succeeded, some failed - warn but continue
                renderer.status(
                    "WARN",
                    f"Some builders failed: {', '.join(failed)}",
                    detail=error_msg,
                )
                return StageResult(success=True, duration=time.time() - start)
            else:
                # All failed - this is a real error
                renderer.status("FAIL", f"All builders failed: {error_msg}")
                return StageResult(success=False, duration=time.time() - start)

    except Exception as e:
        renderer.status("FAIL", f"Derived indices build failed: {e}")
        if config.verbose:
            traceback.print_exc()
        return StageResult(success=False, duration=time.time() - start)


def stage_validation(renderer: ConsoleRenderer, config: WizardConfig) -> StageResult:
    """Validate database and get stats."""
    renderer.banner(12, "VALIDATION")
    start = time.time()

    if config.data_mode == "skip":
        renderer.status("SKIP", "No database to validate")
        return StageResult(success=True, duration=time.time() - start)

    try:
        # Get database stats
        result = subprocess.run(
            [sys.executable, "scripts/db_count.py"],
            check=True,
            capture_output=True,
            text=True,
        )

        # Parse output for metrics (basic parsing)
        output = result.stdout
        metrics = {}

        # Try to extract counts from output
        for line in output.split("\n"):
            if "things" in line.lower():
                try:
                    metrics["total_things"] = int("".join(filter(str.isdigit, line)))
                except ValueError:
                    pass
            elif "people" in line.lower():
                try:
                    metrics["total_people"] = int("".join(filter(str.isdigit, line)))
                except ValueError:
                    pass
            elif "tags" in line.lower():
                try:
                    metrics["total_tags"] = int("".join(filter(str.isdigit, line)))
                except ValueError:
                    pass

        renderer.status("OK", "Database validated")
        if metrics.get("total_things"):
            renderer.status(
                "INFO",
                f"Loaded {metrics['total_things']:,} things, "
                f"{metrics.get('total_people', 0):,} people, "
                f"{metrics.get('total_tags', 0):,} tags",
            )

        if config.verbose:
            print(output)

        return StageResult(
            success=True, duration=time.time() - start, metrics=metrics
        )

    except subprocess.CalledProcessError as e:
        renderer.status("WARN", "Validation failed (non-critical)", detail=str(e))
        return StageResult(success=True, duration=time.time() - start)


def stage_mcp_config(renderer: ConsoleRenderer, config: WizardConfig) -> StageResult:
    """Generate MCP configuration."""
    renderer.banner(13, "MCP CONFIGURATION (OPTIONAL)")
    start = time.time()

    if not config.generate_mcp_config:
        renderer.status("SKIP", "MCP configuration")
        renderer.status("INFO", "See MCP_SETUP.md for manual configuration")
        return StageResult(success=True, duration=time.time() - start)

    try:
        env = None
        if config.claude_config_path:
            env = {
                **dict(os.environ),
                "MEMEX_CLAUDE_CONFIG": config.claude_config_path,
            }

        with renderer.spinner("Generating Claude Desktop configuration"):
            result = subprocess.run(
                [sys.executable, "scripts/generate_mcp_config.py"],
                check=True,
                capture_output=not config.verbose,
                text=True,
                env=env,
            )
            if config.verbose and result.stdout:
                print(result.stdout)

        renderer.status("OK", "MCP configuration generated")
        return StageResult(success=True, duration=time.time() - start)

    except subprocess.CalledProcessError as e:
        renderer.status("FAIL", f"MCP config generation failed: {e}")
        if e.stderr and config.verbose:
            print(e.stderr)
        return StageResult(success=False, duration=time.time() - start)


def stage_testing(renderer: ConsoleRenderer, config: WizardConfig) -> StageResult:
    """Test server import and basic functionality."""
    renderer.banner(14, "TESTING")
    start = time.time()

    try:
        # Test import
        with renderer.spinner("Testing server import"):
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from memex_twos_mcp import server; print('Import: OK')",
                ],
                check=True,
                capture_output=not config.verbose,
                text=True,
            )
            if config.verbose and result.stdout:
                print(result.stdout)

        renderer.status("OK", "Server import successful")

        # Test database count (if we have a database)
        if config.data_mode != "skip":
            result = subprocess.run(
                [sys.executable, "scripts/db_count.py"],
                check=True,
                capture_output=True,
                text=True,
            )
            renderer.status("OK", "Database connectivity verified")

        return StageResult(success=True, duration=time.time() - start)

    except subprocess.CalledProcessError as e:
        renderer.status("FAIL", f"Testing failed: {e}")
        if e.stderr and config.verbose:
            print(e.stderr)
        return StageResult(success=False, duration=time.time() - start)


# ============================================================================
# Stage Selection (Jump Mode)
# ============================================================================


def prompt_stage_selection(renderer: ConsoleRenderer, stages: list) -> int:
    """
    Display numbered stage list and prompt user to select which stage to jump to.

    Args:
        renderer: Console renderer for output
        stages: List of (stage_name, stage_func) tuples

    Returns:
        Index of selected stage (0-based)
    """
    # Build stage display list with human-readable names
    stage_names = {
        "environment_check": "Environment Check",
        "virtual_environment": "Virtual Environment",
        "dependencies": "Dependencies",
        "data_ingest": "Data Ingest",
        "conversion": "Conversion (MD → JSON)",
        "grooming": "Grooming (optional)",
        "entity_classification": "Entity Classification (optional)",
        "sqlite_load": "SQLite Load",
        "embeddings": "Embeddings",
        "derived_indices": "Derived Indices (optional)",
        "validation": "Validation",
        "mcp_config": "MCP Configuration (optional)",
        "testing": "Testing",
    }

    print("\n" + "=" * 70)
    print("  PIPELINE STAGES - Select stage to jump to:")
    print("=" * 70)
    print()

    for i, (stage_key, _) in enumerate(stages, 1):
        display_name = stage_names.get(stage_key, stage_key)
        print(f"  {i:2d}. {display_name}")

    print()
    print("=" * 70)
    print()

    while True:
        try:
            choice = input(f"Jump to stage (1-{len(stages)}) [1]: ").strip()

            if not choice:
                return 0  # Default to first stage

            stage_num = int(choice)
            if 1 <= stage_num <= len(stages):
                return stage_num - 1  # Convert to 0-based index
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(stages)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled by user")
            sys.exit(0)


# ============================================================================
# Main Execution
# ============================================================================


def main() -> None:
    """Run the setup wizard."""
    parser = argparse.ArgumentParser(
        description="Memex Twos MCP setup wizard - Transform your Twos exports into a queryable knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for first-time setup)
  python scripts/setup_wizard.py

  # Non-interactive with defaults
  python scripts/setup_wizard.py --yes --export-file /path/to/export.md

  # Non-interactive with custom options
  python scripts/setup_wizard.py --non-interactive --export-file data.md \\
      --groom --ai-analysis --generate-mcp-config

  # CI/CD mode (minimal, no AI features)
  python scripts/setup_wizard.py --yes --no-color --skip-ai-analysis \\
      --skip-classify-entities --export-file export.md

  # Jump to specific stage (for testing/debugging)
  python scripts/setup_wizard.py --jump
  # Shows numbered stage list, then prompts for selection (e.g., "10" for derived indices)
  # Jump mode enables all execution flags by default (AI analysis, entity classification, LLM summaries)
        """,
    )

    # Input files
    parser.add_argument(
        "--export-file", type=str, help="Path to Twos export file (Markdown with timestamps)"
    )
    parser.add_argument(
        "--claude-config",
        type=str,
        help="Path to Claude Desktop config file (overrides auto-detection)",
    )

    # Mode control
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Non-interactive mode with default options (alias for --non-interactive)",
    )
    parser.add_argument(
        "--non-interactive", action="store_true", help="Run without prompts using CLI flags"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output from subcommands"
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )

    # Stage control
    parser.add_argument(
        "--create-venv",
        action="store_true",
        default=False,
        help="Create virtual environment (default: no in non-interactive mode)",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        default=True,
        help="Install dependencies (default: yes)",
    )
    parser.add_argument(
        "--skip-install-deps",
        action="store_false",
        dest="install_deps",
        help="Skip dependency installation",
    )

    # Data source
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use sample data instead of real export",
    )

    # Processing options
    parser.add_argument(
        "--groom",
        action="store_true",
        default=True,
        help="Run data grooming (default: yes)",
    )
    parser.add_argument(
        "--skip-groom", action="store_false", dest="groom", help="Skip data grooming"
    )
    parser.add_argument(
        "--ai-analysis",
        action="store_true",
        default=False,
        help="Run AI semantic analysis (Developer/uses Claude Code subscription)",
    )
    parser.add_argument(
        "--skip-ai-analysis",
        action="store_false",
        dest="ai_analysis",
        help="Skip AI semantic analysis (default)",
    )
    parser.add_argument(
        "--classify-entities",
        action="store_true",
        default=False,
        help="Run entity classification (requires --ai-analysis)",
    )
    parser.add_argument(
        "--skip-classify-entities",
        action="store_false",
        dest="classify_entities",
        help="Skip entity classification (default)",
    )

    # Database
    parser.add_argument(
        "--overwrite-db",
        action="store_true",
        default=True,
        help="Overwrite existing database (default: yes)",
    )
    parser.add_argument(
        "--no-overwrite-db",
        action="store_false",
        dest="overwrite_db",
        help="Keep existing database",
    )

    # MCP
    parser.add_argument(
        "--generate-mcp-config",
        action="store_true",
        default=True,
        help="Generate Claude Desktop MCP configuration (default: yes)",
    )
    parser.add_argument(
        "--skip-mcp-config",
        action="store_false",
        dest="generate_mcp_config",
        help="Skip MCP configuration generation",
    )

    # Stage jumping (for testing/debugging)
    parser.add_argument(
        "--jump",
        action="store_true",
        help="Jump to a specific pipeline stage (shows numbered list for selection)",
    )

    args = parser.parse_args()

    # Check if Rich is available but not if --no-color
    if not RICH_AVAILABLE and not args.no_color:
        print("Note: 'rich' library not found. Using plain text output.")
        print("Install with: pip install rich\n")

    # Initialize renderer (placeholder config for initial rendering)
    temp_config = WizardConfig(no_color=args.no_color)
    renderer = ConsoleRenderer(temp_config)

    # Print welcome
    if renderer.use_rich:
        renderer.console.print()
        renderer.console.print(
            Panel.fit(
                "[bold white]Memex Twos MCP Setup Wizard[/bold white]\n"
                "Transform your Twos exports into a queryable knowledge base for Claude.",
                border_style="cyan",
                padding=(1, 4),
            )
        )
    else:
        print("\n" + "=" * 70)
        print("  Memex Twos MCP Setup Wizard")
        print("  Transform your Twos exports into a queryable knowledge base for Claude.")
        print("=" * 70 + "\n")

    try:
        # Define all pipeline stages
        stages = [
            ("environment_check", stage_check_python),
            ("virtual_environment", stage_setup_venv),
            ("dependencies", stage_install_deps),
            ("data_ingest", stage_data_setup),
            ("conversion", stage_convert),
            ("grooming", stage_grooming),
            ("entity_classification", stage_entity_classification),
            ("sqlite_load", stage_load_sqlite),
            ("embeddings", stage_embeddings),
            ("derived_indices", stage_derived_indices),
            ("validation", stage_validation),
            ("mcp_config", stage_mcp_config),
            ("testing", stage_testing),
        ]

        # Handle jump mode
        start_index = 0
        if args.jump:
            start_index = prompt_stage_selection(renderer, stages)
            print(f"\n✓ Jumping to stage {start_index + 1}: {stages[start_index][0]}")
            print(f"  Skipping stages 1-{start_index}\n")

        # Stage 1: Collect configuration (unless jumping past it)
        if start_index == 0:
            config = collect_configuration(renderer, args)
            renderer = ConsoleRenderer(config)  # Recreate with actual config
        else:
            # Jumping mode: Developer/testing mode - default all execution flags to True
            # unless explicitly disabled via CLI flags
            # This ensures stages actually run when jumping to them for testing
            config = WizardConfig(
                non_interactive=True,
                verbose=args.verbose,
                no_color=args.no_color,
                export_file=args.export_file,
                claude_config_path=args.claude_config,
                create_venv=args.create_venv,  # Respect explicit choice
                install_deps=args.install_deps,  # Defaults to True
                run_grooming=args.groom,  # Defaults to True
                run_ai_analysis=True,  # Jump mode: enable AI by default for testing
                run_entity_classification=True,  # Jump mode: enable entities by default
                overwrite_db=args.overwrite_db,  # Defaults to True
                build_derived_indices=True,  # Jump mode: always build indices
                build_with_llm=True,  # Jump mode: enable LLM summaries for testing
                generate_mcp_config=args.generate_mcp_config,  # Defaults to True
            )
            renderer = ConsoleRenderer(config)
            print("  Jump mode (dev/testing): All execution flags enabled by default")
            print("  Use --skip-* flags to disable specific stages if needed\n")

        # Execute stages starting from selected index
        failed = False
        for stage_name, stage_func in stages[start_index:]:
            result = stage_func(renderer, config)
            config.stage_results[stage_name] = result

            if not result.success:
                failed = True
                break

        # Stage 15: Complete
        renderer.banner(15, "COMPLETE")

        if failed:
            renderer.status("FAIL", "Setup failed")
            renderer.summary_table(config)
            sys.exit(1)

        renderer.status("OK", "Setup completed successfully")
        renderer.summary_table(config)

        # Next steps
        print()
        renderer.rule("Next Steps")
        print("  1. Restart Claude Desktop")
        print("  2. Try asking: 'What is in my task database?'")
        print("  3. See MCP_SETUP.md for more examples")
        print()

    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as exc:
        print(f"\n\nERROR: {exc}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        print("\nCheck MCP_SETUP.md for troubleshooting")
        sys.exit(1)


if __name__ == "__main__":
    main()
