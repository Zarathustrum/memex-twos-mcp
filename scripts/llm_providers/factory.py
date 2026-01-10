"""
Provider Factory and Registry

Handles provider selection, registration, and auto-detection.

Usage:
    from llm_providers.factory import get_provider, get_default_provider

    # Get specific provider
    provider = get_provider("anthropic-api")

    # Auto-detect first available provider
    provider = get_default_provider()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from . import LLMProvider


# Provider registry (lazy import to avoid circular dependencies)
def _get_provider_registry() -> Dict[str, Type[LLMProvider]]:
    """
    Get provider registry with lazy imports.

    Returns:
        Dictionary mapping provider names to provider classes
    """
    from .anthropic_api import AnthropicApiProvider
    from .claude_cli import ClaudeCliProvider
    from .lmstudio import LmStudioProvider

    return {
        "claude-cli": ClaudeCliProvider,
        "anthropic-api": AnthropicApiProvider,
        "lmstudio": LmStudioProvider,
        # Note: Additional providers will be added in Phase 2:
        # "openai-api": OpenAiApiProvider,
        # "gemini": GeminiProvider,
        # "ollama": OllamaProvider,
    }


def get_provider(name: str, **kwargs) -> LLMProvider:
    """
    Get provider instance by name.

    Args:
        name: Provider name (e.g., "claude-cli", "anthropic-api")
        **kwargs: Provider-specific initialization arguments

    Returns:
        Provider instance

    Raises:
        ValueError: If provider name is unknown

    Examples:
        >>> provider = get_provider("anthropic-api", api_key="sk-ant-...")
        >>> provider = get_provider("claude-cli", default_model="opus")
    """
    registry = _get_provider_registry()

    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown provider: {name}. Available providers: {available}"
        )

    provider_class = registry[name]
    return provider_class(**kwargs)


def get_default_provider() -> Optional[LLMProvider]:
    """
    Get default provider using auto-detection priority order.

    Priority:
    1. Anthropic API (if ANTHROPIC_API_KEY set)
    2. LM Studio (if running locally)
    3. Claude CLI (if installed)
    4. (Future: OpenAI, Gemini, Ollama)

    Returns:
        First available provider instance, or None if none available

    Notes:
        - Tries providers in priority order
        - Returns first provider where is_available() returns True
        - Silently skips unavailable providers (no exceptions)

    Examples:
        >>> provider = get_default_provider()
        >>> if provider:
        ...     response = provider.generate_text("Hello!")
        ... else:
        ...     print("No LLM provider available")
    """
    # Priority order: Anthropic API > LM Studio (local) > Claude CLI
    priority_order = ["anthropic-api", "lmstudio", "claude-cli"]

    for name in priority_order:
        try:
            provider = get_provider(name)
            if provider.is_available():
                return provider
        except (ValueError, ImportError, Exception):
            # Skip providers that fail to initialize
            continue

    return None


def detect_available_providers() -> List[LLMProvider]:
    """
    Detect all available providers on the system.

    Returns:
        List of provider instances that are available

    Notes:
        - Useful for diagnostic purposes
        - Returns only providers where is_available() returns True
        - Order matches provider registry order

    Examples:
        >>> providers = detect_available_providers()
        >>> for provider in providers:
        ...     print(f"✓ {provider.name} (local={provider.is_local})")
    """
    registry = _get_provider_registry()
    available = []

    for name in sorted(registry.keys()):
        try:
            provider = get_provider(name)
            if provider.is_available():
                available.append(provider)
        except (ValueError, ImportError, Exception):
            # Skip providers that fail
            continue

    return available


def list_all_providers() -> List[str]:
    """
    List all registered provider names.

    Returns:
        List of provider names (sorted alphabetically)

    Examples:
        >>> providers = list_all_providers()
        >>> print(f"Registered providers: {', '.join(providers)}")
    """
    registry = _get_provider_registry()
    return sorted(registry.keys())
