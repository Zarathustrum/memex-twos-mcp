"""
LLM Provider Abstraction Layer

Provides a unified interface for multiple LLM providers (Anthropic API, OpenAI,
Gemini, Ollama, LM Studio, Claude CLI).

Design:
- Protocol-based abstraction (ABC) for provider implementations
- Normalized response format (LLMResponse dataclass)
- Normalized error handling (LLMProviderError)
- Privacy-conscious design (is_local flag)

Usage:
    from llm_providers import get_provider, LLMProvider

    provider = get_provider("anthropic-api")
    response = provider.generate_text("Analyze this data...")
    print(response.content)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LLMProviderError",
]


@dataclass
class LLMResponse:
    """
    Normalized LLM response with metadata.

    All providers return this format for consistency.

    Attributes:
        content: Raw response text from the LLM
        model: Model identifier used (e.g., "claude-sonnet-4-5-20250929")
        provider: Provider name (e.g., "anthropic-api", "claude-cli")
        tokens_used: Total tokens consumed (input + output), if available
        finish_reason: Why generation stopped ("stop", "length", etc.), if available
        raw_response: Original provider-specific response object (for debugging)
    """

    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[object] = None


class LLMProviderError(Exception):
    """
    Normalized exception for all provider errors.

    Enables generic error handling across different provider implementations.

    Error Types:
        - 'auth': Authentication failed (invalid API key, etc.)
        - 'rate_limit': Rate limit exceeded, quota exhausted
        - 'timeout': Request timeout (network or generation)
        - 'network': Network/connection errors
        - 'invalid_request': Malformed prompt, invalid parameters
        - 'unavailable': Provider not available (not installed, not running)
        - 'unknown': Other errors

    Attributes:
        message: Human-readable error message
        provider: Provider name that raised the error
        error_type: Normalized error type (see above)
    """

    def __init__(self, message: str, provider: str, error_type: str):
        super().__init__(message)
        self.provider = provider
        self.error_type = error_type


class LLMProvider(ABC):
    """
    Abstract base class for LLM provider implementations.

    All providers must implement this interface to ensure consistent behavior
    across different backends (APIs, local servers, CLI tools).

    Implementations:
        - ClaudeCliProvider: Claude Code CLI (subprocess)
        - AnthropicApiProvider: Anthropic API (HTTP, official SDK)
        - OpenAiApiProvider: OpenAI API (HTTP, official SDK)
        - GeminiProvider: Google Gemini (HTTP, official SDK)
        - OllamaProvider: Ollama (HTTP, local server)
        - LmStudioProvider: LM Studio (HTTP, local server)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Provider identifier.

        Returns:
            Provider name (e.g., "claude-cli", "anthropic-api")
        """
        ...

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """
        Privacy flag indicating if provider runs locally.

        Local providers (Ollama, LM Studio) process data on the local machine.
        Cloud providers (Anthropic, OpenAI, Gemini, Claude CLI) send data externally.

        Returns:
            True for local providers (privacy-conscious), False for cloud providers
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Fast check if provider is available and configured.

        This should be a quick validation without making actual LLM calls:
        - CLI providers: Check if executable is in PATH
        - API providers: Check if API key is set
        - Local servers: Optionally ping server (with timeout)

        Returns:
            True if provider can be used, False otherwise
        """
        ...

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout: int = 120,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate text completion from prompt.

        Args:
            prompt: User prompt (main content to process)
            model: Model identifier (provider-specific, uses default if None)
            timeout: Timeout in seconds (default 120s)
            temperature: Sampling temperature 0-1 (lower = more deterministic)
            max_tokens: Maximum tokens to generate (None = provider default)

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            LLMProviderError: If generation fails (any reason)

        Notes:
            - Provider implementations should normalize their native errors to
              LLMProviderError with appropriate error_type
            - Timeout enforcement is provider-specific (API vs subprocess)
        """
        ...

    def validate_config(self) -> list[str]:
        """
        Validate provider configuration.

        Returns:
            List of error messages (empty list = valid configuration)

        Notes:
            - Optional method (providers can override for custom validation)
            - Default implementation returns empty list (no validation)
        """
        return []
