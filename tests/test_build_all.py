"""Tests for the build_all orchestrator."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path for importing build_all
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_all import build_derived_indices  # type: ignore  # noqa: E402


def create_mock_builder_module(build_func: MagicMock) -> ModuleType:
    """Create a mock builder module with a build function."""
    module = ModuleType("mock_builder")
    module.build = build_func
    return module


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Create a minimal test database with metadata table."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create metadata table (minimal schema)
    cur.execute(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    conn.commit()
    conn.close()

    return db_path


@pytest.fixture
def mock_builder_success() -> MagicMock:
    """Mock builder function that succeeds."""

    def builder(db_path: Path, force: bool = False) -> Dict[str, Any]:
        return {
            "success": True,
            "stats": {"items_processed": 100, "duration_ms": 500},
            "duration_seconds": 0.5,
            "error": None,
        }

    return MagicMock(side_effect=builder)


@pytest.fixture
def mock_builder_failure() -> MagicMock:
    """Mock builder function that fails."""

    def builder(db_path: Path, force: bool = False) -> Dict[str, Any]:
        return {
            "success": False,
            "stats": {},
            "duration_seconds": 0.1,
            "error": "Simulated builder failure",
        }

    return MagicMock(side_effect=builder)


@pytest.fixture
def mock_builder_exception() -> MagicMock:
    """Mock builder function that raises exception."""

    def builder(db_path: Path, force: bool = False) -> Dict[str, Any]:
        raise RuntimeError("Simulated exception")

    return MagicMock(side_effect=builder)


# ============================================================================
# Test Cases
# ============================================================================


def test_all_builders_succeed(test_db: Path, mock_builder_success: MagicMock) -> None:
    """Test orchestrator when all builders succeed."""
    # Create mock modules
    mock_timepacks = create_mock_builder_module(mock_builder_success)
    mock_threads = create_mock_builder_module(mock_builder_success)

    # Patch sys.modules to inject our mocks
    with patch.dict(
        "sys.modules",
        {
            "build_timepacks": mock_timepacks,
            "build_threads": mock_threads,
        },
    ):
        result = build_derived_indices(
            db_path=test_db, builders=["timepacks", "threads"], verbose=False
        )

    assert result["success"] is True
    assert result["builders_succeeded"] == ["timepacks", "threads"]
    assert result["builders_failed"] == []
    assert len(result["stats"]) == 2
    assert "timepacks" in result["stats"]
    assert "threads" in result["stats"]

    # Verify metadata was updated
    conn = sqlite3.connect(test_db)
    cur = conn.cursor()

    metadata = dict(cur.execute("SELECT key, value FROM metadata").fetchall())
    assert "last_derived_build" in metadata
    assert "timepacks_version" in metadata
    assert "threads_version" in metadata
    assert metadata["timepacks_version"] == "1.0"

    conn.close()


def test_one_builder_fails_others_continue(
    test_db: Path, mock_builder_success: MagicMock, mock_builder_failure: MagicMock
) -> None:
    """Test that other builders continue when one fails."""
    mock_timepacks = create_mock_builder_module(mock_builder_failure)
    mock_threads = create_mock_builder_module(mock_builder_success)

    with patch.dict(
        "sys.modules",
        {
            "build_timepacks": mock_timepacks,
            "build_threads": mock_threads,
        },
    ):
        result = build_derived_indices(
            db_path=test_db, builders=["timepacks", "threads"], verbose=False
        )

    assert result["success"] is False
    assert result["builders_succeeded"] == ["threads"]
    assert result["builders_failed"] == ["timepacks"]
    assert "timepacks" in result["errors"]
    assert result["errors"]["timepacks"] == "Simulated builder failure"

    # Verify only successful builder updated metadata
    conn = sqlite3.connect(test_db)
    cur = conn.cursor()

    metadata = dict(cur.execute("SELECT key, value FROM metadata").fetchall())
    assert "threads_version" in metadata
    assert "timepacks_version" not in metadata

    conn.close()


def test_dependency_handling(
    test_db: Path, mock_builder_success: MagicMock, mock_builder_failure: MagicMock
) -> None:
    """Test that dependent builders are skipped when dependencies fail."""
    # Timepacks fails, summaries should be skipped
    mock_timepacks = create_mock_builder_module(mock_builder_failure)
    mock_threads = create_mock_builder_module(mock_builder_success)
    mock_summaries = create_mock_builder_module(mock_builder_success)

    with patch.dict(
        "sys.modules",
        {
            "build_timepacks": mock_timepacks,
            "build_threads": mock_threads,
            "build_month_summaries": mock_summaries,
        },
    ):
        result = build_derived_indices(
            db_path=test_db,
            builders=["timepacks", "threads", "summaries"],
            verbose=False,
        )

    assert result["success"] is False
    assert "timepacks" in result["builders_failed"]
    assert "threads" in result["builders_succeeded"]
    assert "summaries" in result["builders_skipped"]


def test_builder_exception_handling(
    test_db: Path, mock_builder_success: MagicMock, mock_builder_exception: MagicMock
) -> None:
    """Test that exceptions from builders are caught and logged."""
    mock_timepacks = create_mock_builder_module(mock_builder_exception)
    mock_threads = create_mock_builder_module(mock_builder_success)

    with patch.dict(
        "sys.modules",
        {
            "build_timepacks": mock_timepacks,
            "build_threads": mock_threads,
        },
    ):
        result = build_derived_indices(
            db_path=test_db, builders=["timepacks", "threads"], verbose=False
        )

    assert result["success"] is False
    assert "timepacks" in result["builders_failed"]
    assert "timepacks" in result["errors"]
    assert "Simulated exception" in result["errors"]["timepacks"]
    assert "threads" in result["builders_succeeded"]


def test_database_not_found() -> None:
    """Test error handling when database doesn't exist."""
    fake_db = Path("/nonexistent/path/fake.db")

    result = build_derived_indices(db_path=fake_db, verbose=False)

    assert result["success"] is False
    assert len(result["builders_failed"]) > 0
    assert "db" in result["errors"]
    assert "not found" in result["errors"]["db"].lower()


def test_invalid_builder_names(test_db: Path) -> None:
    """Test validation of builder names."""
    result = build_derived_indices(
        db_path=test_db, builders=["invalid_builder", "another_fake"], verbose=False
    )

    assert result["success"] is False
    assert "validation" in result["errors"]
    assert "Unknown builders" in result["errors"]["validation"]


def test_default_builders_no_llm(
    test_db: Path, mock_builder_success: MagicMock
) -> None:
    """Test that default builders are timepacks + threads (no LLM)."""
    mock_timepacks = create_mock_builder_module(mock_builder_success)
    mock_threads = create_mock_builder_module(mock_builder_success)

    with patch.dict(
        "sys.modules",
        {
            "build_timepacks": mock_timepacks,
            "build_threads": mock_threads,
        },
    ):
        result = build_derived_indices(db_path=test_db, verbose=False)

    assert result["builders_run"] == ["timepacks", "threads"]
    assert "summaries" not in result["builders_run"]


def test_with_llm_flag(test_db: Path, mock_builder_success: MagicMock) -> None:
    """Test that --with-llm adds summaries builder."""
    mock_timepacks = create_mock_builder_module(mock_builder_success)
    mock_threads = create_mock_builder_module(mock_builder_success)
    mock_summaries = create_mock_builder_module(mock_builder_success)

    with patch.dict(
        "sys.modules",
        {
            "build_timepacks": mock_timepacks,
            "build_threads": mock_threads,
            "build_month_summaries": mock_summaries,
        },
    ):
        result = build_derived_indices(db_path=test_db, with_llm=True, verbose=False)

    assert "summaries" in result["builders_run"]
    assert len(result["builders_run"]) == 3


def test_specific_builders_selection(
    test_db: Path, mock_builder_success: MagicMock
) -> None:
    """Test running specific builders only."""
    mock_timepacks = create_mock_builder_module(mock_builder_success)

    with patch.dict(
        "sys.modules",
        {
            "build_timepacks": mock_timepacks,
        },
    ):
        result = build_derived_indices(
            db_path=test_db, builders=["timepacks"], verbose=False
        )

    assert result["builders_run"] == ["timepacks"]
    assert result["success"] is True
    assert len(result["builders_succeeded"]) == 1


def test_force_flag_passed_to_builders(
    test_db: Path, mock_builder_success: MagicMock
) -> None:
    """Test that force flag is passed to builder functions."""
    mock_timepacks = create_mock_builder_module(mock_builder_success)

    with patch.dict(
        "sys.modules",
        {
            "build_timepacks": mock_timepacks,
        },
    ):
        build_derived_indices(
            db_path=test_db, builders=["timepacks"], force=True, verbose=False
        )

    # Verify force=True was passed to builder
    mock_builder_success.assert_called_once()
    call_args = mock_builder_success.call_args
    assert call_args.kwargs["force"] is True


def test_import_failure_handling(test_db: Path) -> None:
    """Test handling when builder module cannot be imported."""
    # Don't patch any imports - they will fail naturally
    result = build_derived_indices(
        db_path=test_db, builders=["timepacks"], verbose=False
    )

    # Should fail because build_timepacks doesn't exist
    assert result["success"] is False
    assert "timepacks" in result["builders_failed"]


def test_metadata_timestamp_format(
    test_db: Path, mock_builder_success: MagicMock
) -> None:
    """Test that metadata timestamps are in ISO format."""
    mock_timepacks = create_mock_builder_module(mock_builder_success)

    with patch.dict(
        "sys.modules",
        {
            "build_timepacks": mock_timepacks,
        },
    ):
        build_derived_indices(db_path=test_db, builders=["timepacks"], verbose=False)

    conn = sqlite3.connect(test_db)
    cur = conn.cursor()

    timestamp = cur.execute(
        "SELECT value FROM metadata WHERE key = 'last_derived_build'"
    ).fetchone()[0]

    conn.close()

    # Check ISO format (YYYY-MM-DDTHH:MM:SS)
    assert "T" in timestamp
    assert len(timestamp) == 19  # ISO timestamp without timezone


def test_empty_builders_list(test_db: Path) -> None:
    """Test behavior with empty builders list."""
    result = build_derived_indices(db_path=test_db, builders=[], verbose=False)

    # No builders to run, but not a failure
    assert result["success"] is False  # False because no builders succeeded
    assert len(result["builders_run"]) == 0
    assert len(result["builders_succeeded"]) == 0


def test_verbose_output(
    test_db: Path, mock_builder_success: MagicMock, capsys: pytest.CaptureFixture
) -> None:
    """Test that verbose mode produces output."""
    mock_timepacks = create_mock_builder_module(mock_builder_success)

    with patch.dict(
        "sys.modules",
        {
            "build_timepacks": mock_timepacks,
        },
    ):
        build_derived_indices(db_path=test_db, builders=["timepacks"], verbose=True)

    captured = capsys.readouterr()
    assert "Building derived indices" in captured.out
    assert "TimePacks" in captured.out
    assert "Success" in captured.out


def test_quiet_mode(
    test_db: Path, mock_builder_success: MagicMock, capsys: pytest.CaptureFixture
) -> None:
    """Test that verbose=False suppresses output."""
    mock_timepacks = create_mock_builder_module(mock_builder_success)

    with patch.dict(
        "sys.modules",
        {
            "build_timepacks": mock_timepacks,
        },
    ):
        build_derived_indices(db_path=test_db, builders=["timepacks"], verbose=False)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_result_structure(test_db: Path, mock_builder_success: MagicMock) -> None:
    """Test that result dictionary has correct structure."""
    mock_timepacks = create_mock_builder_module(mock_builder_success)

    with patch.dict(
        "sys.modules",
        {
            "build_timepacks": mock_timepacks,
        },
    ):
        result = build_derived_indices(
            db_path=test_db, builders=["timepacks"], verbose=False
        )

    # Verify all expected keys are present
    assert "success" in result
    assert "builders_run" in result
    assert "builders_succeeded" in result
    assert "builders_failed" in result
    assert "builders_skipped" in result
    assert "duration_seconds" in result
    assert "stats" in result
    assert "errors" in result

    # Verify types
    assert isinstance(result["success"], bool)
    assert isinstance(result["builders_run"], list)
    assert isinstance(result["builders_succeeded"], list)
    assert isinstance(result["builders_failed"], list)
    assert isinstance(result["builders_skipped"], list)
    assert isinstance(result["duration_seconds"], float)
    assert isinstance(result["stats"], dict)
    assert isinstance(result["errors"], dict)
