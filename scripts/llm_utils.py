#!/usr/bin/env python3
"""
Unified LLM Backend Abstraction Layer

Provides a consistent interface for LLM invocations across the memex-twos-mcp
project, with pluggable providers (Anthropic API, OpenAI, Gemini, Ollama,
LM Studio, Claude Code CLI).

This centralizes:
- Provider detection and selection
- Error handling and retry logic
- Response parsing (JSON, text, markdown)
- Timeout and rate limit management
- Billing/quota awareness

Usage:
    from llm_utils import invoke_llm, check_llm_available

    # Simple JSON invocation
    response = invoke_llm(prompt="Analyze this data...", response_format="json")

    # With provider selection
    response = invoke_llm(prompt="...", provider="anthropic-api", timeout=300)

    # Check availability before use
    if not check_llm_available():
        print("No LLM provider available")

Provider Priority (Auto-Detection):
1. Anthropic API (if ANTHROPIC_API_KEY env var set)
2. Claude Code CLI (if `claude` executable on PATH)
3. (Future: OpenAI, Gemini, Ollama, LM Studio)

Billing Notes:
- Claude Code CLI: Uses user's Claude subscription quota
- Anthropic API: Direct API billing (pay-per-token)
- (Future: OpenAI, Gemini = API billing; Ollama, LM Studio = local/free)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Literal, Optional

# Import provider infrastructure
try:
    from llm_providers import LLMProvider, LLMProviderError, LLMResponse
    from llm_providers.factory import get_default_provider, get_provider
    from llm_config import load_config
except ImportError as e:
    print(
        f"ERROR: Could not import llm_providers. "
        f"Make sure you're in the correct directory: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

# ============================================================================
# Provider Selection
# ============================================================================


def select_provider(
    provider_name: Optional[str] = None,
    config_path: Optional[Path] = None,
    **provider_kwargs
) -> LLMProvider:
    """
    Select and initialize LLM provider.

    Priority:
    1. Explicit provider_name argument
    2. default_provider from config (YAML or env var)
    3. Auto-detect (first available provider)

    IMPORTANT: If provider_name is explicitly specified (either as argument or in config),
    this function will FAIL LOUDLY if that provider is unavailable. No silent fallback.

    Args:
        provider_name: Provider identifier (e.g., "anthropic-api", "claude-cli")
        config_path: Optional path to YAML config file
        **provider_kwargs: Provider-specific initialization arguments

    Returns:
        LLMProvider instance

    Raises:
        RuntimeError: If explicit provider is unavailable, or no provider found

    Examples:
        >>> provider = select_provider("lmstudio")  # Fails loudly if unavailable
        >>> provider = select_provider()  # Auto-detect
        >>> provider = select_provider(config_path=Path("config.yaml"))
    """
    config = load_config(config_path)

    # 1. Explicit provider argument
    if provider_name:
        try:
            # Get provider config if available
            provider_config = config.get_provider_config(provider_name)
            kwargs = provider_kwargs.copy()

            # Merge config into kwargs (kwargs take precedence)
            if provider_config:
                if provider_config.api_key and "api_key" not in kwargs:
                    kwargs["api_key"] = provider_config.api_key
                if provider_config.endpoint and "endpoint" not in kwargs:
                    kwargs["endpoint"] = provider_config.endpoint
                if (
                    provider_config.default_model
                    and "default_model" not in kwargs
                ):
                    kwargs["default_model"] = provider_config.default_model

            provider = get_provider(provider_name, **kwargs)

            # Validate provider is available
            if not provider.is_available():
                errors = provider.validate_config()
                error_details = "\n  - ".join(errors) if errors else "Unknown reason"
                raise RuntimeError(
                    f"Provider '{provider_name}' not available:\n  - {error_details}"
                )

            return provider

        except (ValueError, ImportError) as e:
            raise RuntimeError(
                f"Failed to initialize provider '{provider_name}': {e}"
            ) from e

    # 2. Try configured default provider - FAIL LOUDLY if configured but unavailable
    if config.default_provider:
        try:
            return select_provider(config.default_provider, config_path, **provider_kwargs)
        except RuntimeError as e:
            # Explicit config means NO SILENT FALLBACK
            print(
                f"\n[ERROR] Configured default provider '{config.default_provider}' is unavailable.",
                file=sys.stderr,
            )
            raise RuntimeError(
                f"Configured default provider '{config.default_provider}' is unavailable. "
                f"Fix the configuration or remove default_provider to auto-detect.\n"
                f"Details: {e}"
            ) from e

    # 3. Auto-detect
    provider = get_default_provider()
    if provider is None:
        raise RuntimeError(
            "No LLM provider available. Install one of:\n"
            "  - Claude CLI: https://code.claude.com/docs/en/quickstart\n"
            "  - Anthropic API: pip install -e '.[llm-anthropic]' + "
            "set ANTHROPIC_API_KEY\n"
            "  - (Future: OpenAI, Gemini, Ollama, LM Studio)"
        )

    return provider


def check_llm_available(config_path: Optional[Path] = None) -> bool:
    """
    Check if any LLM provider is available.

    Args:
        config_path: Optional path to YAML config file

    Returns:
        True if at least one provider is usable.

    Examples:
        >>> if check_llm_available():
        ...     response = invoke_llm("Hello!")
    """
    try:
        select_provider(config_path=config_path)
        return True
    except RuntimeError:
        return False


# ============================================================================
# Response Parsing (PRESERVED FROM ORIGINAL)
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
    text_clean = re.sub(r"```json\s*", "", text)
    text_clean = re.sub(r"```\s*$", "", text_clean, flags=re.MULTILINE)

    # Step 2: Remove comments
    text_clean = re.sub(r"//.*$", "", text_clean, flags=re.MULTILINE)
    text_clean = re.sub(r"/\*.*?\*/", "", text_clean, flags=re.DOTALL)

    # Step 3: Find first balanced JSON object
    start_idx = text_clean.find("{")
    if start_idx == -1:
        raise ValueError("No JSON object found in response")

    brace_count = 0
    end_idx = start_idx
    for i, char in enumerate(text_clean[start_idx:], start_idx):
        if char == "{":
            brace_count += 1
        elif char == "}":
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
# Public Interface
# ============================================================================


def invoke_llm(
    prompt: str,
    response_format: Literal["json", "text", "markdown"] = "json",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 120,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any] | str:
    """
    Unified LLM invocation with pluggable providers.

    This is the primary interface for all LLM calls in the project.
    Handles provider selection, invocation, parsing, and error handling.

    BACKWARD COMPATIBLE: Existing calls work unchanged (new params optional).

    Billing awareness:
    - Claude CLI: Uses subscription quota (~2K-5K tokens per call)
    - Anthropic API: Direct billing (pay-per-token)
    - (Future: OpenAI, Gemini = billing; Ollama, LM Studio = local/free)

    Response parsing:
    - response_format="json": Returns parsed dict, raises if invalid JSON
    - response_format="text": Returns raw string
    - response_format="markdown": Returns raw string

    Args:
        prompt: The prompt to send to the LLM
        response_format: Expected response format (json, text, markdown)
        provider: Provider to use (None = auto-detect)
        model: Model to use (None = provider default)
        timeout: Timeout in seconds (default 120s)
        temperature: Sampling temperature 0-1 (lower = more deterministic)
        max_tokens: Maximum tokens to generate (None = provider default)
        config_path: Optional path to YAML config file

    Returns:
        Parsed JSON dict (if response_format="json") or raw string

    Raises:
        RuntimeError: If LLM invocation fails
        ValueError: If JSON parsing fails (response_format="json")

    Examples:
        >>> response = invoke_llm("Analyze: {...}", response_format="json")
        >>> print(response["themes"])

        >>> text = invoke_llm("Summarize: ...", response_format="text",
        ...                   provider="anthropic-api")
        >>> print(text)
    """
    # Select provider
    try:
        llm_provider = select_provider(provider, config_path=config_path)
    except RuntimeError as e:
        print(f"\n[ERROR] Provider selection failed: {e}", file=sys.stderr)
        raise

    # Generate response
    try:
        llm_response: LLMResponse = llm_provider.generate_text(
            prompt=prompt,
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except LLMProviderError as e:
        # Enhanced error reporting with actionable hints
        error_msg = f"LLM invocation failed ({e.provider}, {e.error_type}): {e}"
        print(f"\n[ERROR] {error_msg}", file=sys.stderr)

        # Provider-specific hints
        if e.error_type == "rate_limit":
            print(
                "[HINT] Rate limit hit. Wait and retry, or switch provider.",
                file=sys.stderr,
            )
        elif e.error_type == "auth":
            print("[HINT] Check API key configuration.", file=sys.stderr)
        elif e.error_type == "timeout":
            print(
                f"[HINT] Timeout after {timeout}s. Try increasing timeout.",
                file=sys.stderr,
            )
        elif e.error_type == "unavailable":
            print(
                "[HINT] Provider not available. Install or configure it.",
                file=sys.stderr,
            )

        raise RuntimeError(error_msg) from e

    # Parse response based on format
    raw_response = llm_response.content

    if response_format == "json":
        try:
            return extract_json_from_response(raw_response)
        except Exception as e:
            # DEBUG: Log raw response on parse failure
            print(f"\n[ERROR] JSON parsing failed: {e}", file=sys.stderr)
            print(
                f"[ERROR] Provider: {llm_response.provider}, "
                f"Model: {llm_response.model}",
                file=sys.stderr,
            )
            print(
                "[ERROR] Raw response (first 500 chars):", file=sys.stderr
            )
            print(raw_response[:500], file=sys.stderr)
            raise
    else:
        return raw_response


# ============================================================================
# Legacy Compatibility Wrappers (PRESERVED FROM ORIGINAL)
# ============================================================================


def invoke_llm_via_claude_code(prompt: str, timeout: int = 120) -> Dict[str, Any]:
    """
    Legacy wrapper for build_month_summaries.py compatibility.

    DEPRECATED: Use invoke_llm() directly.

    Forces Claude CLI provider for exact backward compatibility.

    Args:
        prompt: Analysis prompt
        timeout: Timeout in seconds

    Returns:
        Parsed JSON response

    Examples:
        >>> response = invoke_llm_via_claude_code("Analyze month data...")
    """
    return invoke_llm(
        prompt, response_format="json", provider="claude-cli", timeout=timeout
    )
