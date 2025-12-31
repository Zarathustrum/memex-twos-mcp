#!/usr/bin/env python3
"""
Unified LLM Backend Abstraction Layer

Provides a consistent interface for LLM invocations across the memex-twos-mcp
project, with pluggable backends (Claude Code CLI, Anthropic API).

This centralizes:
- Backend detection and selection
- Error handling and retry logic
- Response parsing (JSON, text, markdown)
- Timeout and rate limit management
- Billing/quota awareness

Usage:
    from llm_utils import invoke_llm, check_llm_available

    # Simple JSON invocation
    response = invoke_llm(prompt="Analyze this data...", response_format="json")

    # With backend selection
    response = invoke_llm(prompt="...", backend="claude-cli", timeout=300)

    # Check availability before use
    if not check_llm_available():
        print("No LLM backend available")

Backend Priority:
1. Anthropic API (if ANTHROPIC_API_KEY env var set) - future
2. Claude Code CLI (if `claude` executable on PATH)
3. Error if none available

Billing Notes:
- Claude Code CLI: Uses user's Claude subscription quota
- Anthropic API: Direct API billing (future)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Literal, Optional

# ============================================================================
# Backend Detection
# ============================================================================


def check_claude_cli_installed() -> bool:
    """
    Check if Claude Code CLI is installed and available on PATH.

    Returns:
        True if `claude` executable is found.
    """
    result = subprocess.run(
        ["which", "claude"],
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def detect_backend() -> str:
    """
    Auto-detect which LLM backend is available.

    Priority:
    1. Anthropic API (if ANTHROPIC_API_KEY set) - future
    2. Claude Code CLI (if `claude` on PATH)

    Returns:
        Backend name: "api" or "claude-cli"

    Raises:
        RuntimeError: If no backend is available
    """
    # Future: Check for API key
    # if os.getenv("ANTHROPIC_API_KEY"):
    #     return "api"

    if check_claude_cli_installed():
        return "claude-cli"

    raise RuntimeError(
        "No LLM backend available. Install Claude Code CLI: "
        "https://code.claude.com/docs/en/quickstart"
    )


def check_llm_available() -> bool:
    """
    Check if any LLM backend is available.

    Returns:
        True if at least one backend is usable.
    """
    try:
        detect_backend()
        return True
    except RuntimeError:
        return False


# ============================================================================
# Response Parsing
# ============================================================================


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response text.

    Handles:
    - Markdown code blocks: ```json { ... } ```
    - Raw JSON objects: { ... }
    - Removes C-style comments (// and /* */)
    - Extracts first balanced brace pair

    Args:
        text: Raw LLM response text

    Returns:
        Parsed JSON as dictionary

    Raises:
        ValueError: If no valid JSON found or braces unbalanced
        json.JSONDecodeError: If JSON is malformed
    """
    # Step 1: Strip markdown fences
    text_clean = re.sub(r'```json\s*', '', text)
    text_clean = re.sub(r'```\s*$', '', text_clean, flags=re.MULTILINE)

    # Step 2: Remove comments
    text_clean = re.sub(r'//.*$', '', text_clean, flags=re.MULTILINE)
    text_clean = re.sub(r'/\*.*?\*/', '', text_clean, flags=re.DOTALL)

    # Step 3: Find first balanced JSON object
    start_idx = text_clean.find('{')
    if start_idx == -1:
        raise ValueError(f"No JSON object found in response")

    brace_count = 0
    end_idx = start_idx
    for i, char in enumerate(text_clean[start_idx:], start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if brace_count != 0:
        raise ValueError(f"Unbalanced braces in JSON (count: {brace_count})")

    json_str = text_clean[start_idx:end_idx].strip()

    # Step 4: Parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Enhanced error reporting
        problem_start = max(0, e.pos - 50)
        problem_end = min(len(json_str), e.pos + 50)
        context = json_str[problem_start:problem_end]

        raise ValueError(
            f"Invalid JSON at line {e.lineno}, col {e.colno}: {e.msg}\n"
            f"Context: ...{context}..."
        )


# ============================================================================
# Backend Implementations
# ============================================================================


def invoke_claude_cli(
    prompt: str,
    model: str = "sonnet",
    timeout: int = 120,
    response_format: str = "json"
) -> str:
    """
    Invoke Claude Code CLI for LLM processing.

    External dependency: Requires `claude` CLI installed and authenticated.
    Billing: Consumes user's Claude subscription quota.

    Security considerations:
    - Temp file used to avoid command-line injection
    - Timeout enforced to prevent indefinite hangs
    - Temp file cleaned up even on error (finally block)

    Failure modes:
    - Claude CLI not installed → RuntimeError
    - API rate limit hit → RuntimeError (retry manually)
    - Timeout exceeded → subprocess.TimeoutExpired → RuntimeError
    - Network failure → RuntimeError

    Args:
        prompt: Analysis prompt (can be large, ~1K-5K chars)
        model: Model name (sonnet, opus, haiku)
        timeout: Timeout in seconds (default 120s)
        response_format: Expected format (json, text, markdown)

    Returns:
        Raw response text from Claude

    Raises:
        RuntimeError: If CLI invocation fails
        subprocess.TimeoutExpired: If timeout exceeded
    """
    # Write prompt to temp file to avoid shell injection
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        prompt_file = Path(f.name)
        f.write(prompt)

    try:
        # Invoke claude with prompt file via stdin
        # Note: Using cat | claude pattern for stdin compatibility
        result = subprocess.run(
            ["bash", "-c", f"cat {prompt_file} | claude --model {model}"],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI invocation failed: {result.stderr}")

        return result.stdout

    finally:
        # Cleanup temp file
        if prompt_file.exists():
            prompt_file.unlink()


def invoke_anthropic_api(
    prompt: str,
    model: str = "claude-sonnet-4-5",
    timeout: int = 120,
    response_format: str = "json"
) -> str:
    """
    Invoke Anthropic API directly for LLM processing.

    FUTURE: Not yet implemented. Placeholder for API-based backend.

    Requires:
    - ANTHROPIC_API_KEY environment variable
    - anthropic Python SDK: pip install anthropic

    Advantages over CLI:
    - Lower latency (no CLI overhead)
    - Better rate limit handling
    - Programmatic retry logic
    - Token counting and cost tracking

    Args:
        prompt: Analysis prompt
        model: Model ID (claude-sonnet-4-5, etc.)
        timeout: Timeout in seconds
        response_format: Expected format (json, text, markdown)

    Returns:
        Raw response text from API

    Raises:
        NotImplementedError: Always (not yet implemented)
    """
    raise NotImplementedError("API backend not yet implemented. Use claude-cli.")


# ============================================================================
# Public Interface
# ============================================================================


def invoke_llm(
    prompt: str,
    response_format: Literal["json", "text", "markdown"] = "json",
    model: str = "sonnet",
    timeout: int = 120,
    backend: Literal["auto", "claude-cli", "api"] = "auto"
) -> Dict[str, Any] | str:
    """
    Unified LLM invocation with pluggable backends.

    This is the primary interface for all LLM calls in the project.
    Handles backend selection, invocation, parsing, and error handling.

    Billing awareness:
    - Claude CLI: Uses subscription quota (~2K-5K tokens per call)
    - API: Direct billing (future)

    Response parsing:
    - response_format="json": Returns parsed dict, raises if invalid JSON
    - response_format="text": Returns raw string
    - response_format="markdown": Returns raw string

    Args:
        prompt: The prompt to send to the LLM
        response_format: Expected response format
        model: Model to use (sonnet, opus, haiku for CLI; full ID for API)
        timeout: Timeout in seconds (default 120s)
        backend: Backend to use (auto-detect, claude-cli, or api)

    Returns:
        Parsed JSON dict (if response_format="json") or raw string

    Raises:
        RuntimeError: If LLM invocation fails
        ValueError: If JSON parsing fails (response_format="json")
        subprocess.TimeoutExpired: If timeout exceeded

    Examples:
        >>> response = invoke_llm("Analyze: {...}", response_format="json")
        >>> print(response["themes"])

        >>> text = invoke_llm("Summarize: ...", response_format="text")
        >>> print(text)
    """
    # Detect backend if auto
    if backend == "auto":
        backend = detect_backend()

    # Invoke backend
    if backend == "claude-cli":
        raw_response = invoke_claude_cli(prompt, model, timeout, response_format)
    elif backend == "api":
        raw_response = invoke_anthropic_api(prompt, model, timeout, response_format)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Parse response based on format
    if response_format == "json":
        try:
            return extract_json_from_response(raw_response)
        except Exception as e:
            # DEBUG: Log raw response on parse failure
            import sys
            print(f"\n[ERROR] JSON parsing failed: {e}", file=sys.stderr)
            print(f"[ERROR] Raw LLM response (first 500 chars):", file=sys.stderr)
            print(raw_response[:500], file=sys.stderr)
            raise
    else:
        return raw_response


# ============================================================================
# Legacy Compatibility Wrappers
# ============================================================================


def invoke_llm_via_claude_code(prompt: str, timeout: int = 120) -> Dict[str, Any]:
    """
    Legacy wrapper for build_month_summaries.py compatibility.

    DEPRECATED: Use invoke_llm() directly.

    Args:
        prompt: Analysis prompt
        timeout: Timeout in seconds

    Returns:
        Parsed JSON response
    """
    return invoke_llm(prompt, response_format="json", timeout=timeout)
