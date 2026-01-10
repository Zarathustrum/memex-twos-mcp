"""
Anthropic API Provider

Uses official Anthropic Python SDK for Messages API calls.
This is the PRIORITY provider for Phase 1.

External Dependency: pip install anthropic
API Key: Set ANTHROPIC_API_KEY environment variable
Billing: Direct API billing (pay-per-token)

Advantages over CLI:
- Lower latency (no CLI overhead)
- Better rate limit handling
- Programmatic retry logic
- Token counting and cost tracking
"""

from __future__ import annotations

import os
from typing import Optional

from . import LLMProvider, LLMProviderError, LLMResponse


class AnthropicApiProvider(LLMProvider):
    """
    Anthropic API implementation using official SDK.

    Uses lazy initialization for the Anthropic client to improve startup time.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "claude-sonnet-4-5-20250929",
    ):
        """
        Initialize Anthropic API provider.

        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
            default_model: Default model ID

        Raises:
            ImportError: If anthropic package not installed
        """
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._default_model = default_model
        self._client = None  # Lazy-loaded

    @property
    def name(self) -> str:
        return "anthropic-api"

    @property
    def is_local(self) -> bool:
        # Anthropic API sends data to external service
        return False

    def _get_client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise ImportError(
                    "Anthropic SDK not installed. Install with: "
                    "pip install -e '.[llm-anthropic]'"
                ) from e

            if not self._api_key:
                raise LLMProviderError(
                    "ANTHROPIC_API_KEY not set. Export ANTHROPIC_API_KEY=sk-ant-...",
                    provider=self.name,
                    error_type="auth",
                )

            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return self._api_key is not None

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout: int = 120,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate text using Anthropic Messages API.

        Args:
            prompt: User prompt
            model: Model ID (e.g., claude-sonnet-4-5-20250929)
            timeout: Timeout in seconds (passed to API client)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate (default 2048)

        Returns:
            LLMResponse with generated content and token usage

        Raises:
            LLMProviderError: If API call fails
        """
        client = self._get_client()
        model_to_use = model or self._default_model
        max_tokens_to_use = max_tokens or 2048

        try:
            response = client.messages.create(
                model=model_to_use,
                max_tokens=max_tokens_to_use,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )

            # Extract text from response
            # Response format: Message(content=[ContentBlock(text="...", type="text")])
            if not response.content:
                raise LLMProviderError(
                    "Empty response from Anthropic API",
                    provider=self.name,
                    error_type="unknown",
                )

            # Get first text block
            first_block = response.content[0]
            text = getattr(first_block, "text", None)
            if not isinstance(text, str):
                raise LLMProviderError(
                    f"Unexpected response format: {type(first_block)}",
                    provider=self.name,
                    error_type="unknown",
                )

            # Extract token usage
            tokens_used = None
            if hasattr(response, "usage"):
                usage = response.usage
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
                tokens_used = input_tokens + output_tokens

            # Extract finish reason
            finish_reason = getattr(response, "stop_reason", None)

            return LLMResponse(
                content=text,
                model=model_to_use,
                provider=self.name,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                raw_response=response,
            )

        except ImportError:
            # Re-raise ImportError from _get_client()
            raise
        except Exception as e:
            # Normalize SDK exceptions to LLMProviderError
            error_str = str(e).lower()

            # Detect error type
            if "rate" in error_str or "429" in error_str:
                error_type = "rate_limit"
            elif (
                "auth" in error_str
                or "401" in error_str
                or "api key" in error_str
                or "invalid" in error_str and "key" in error_str
            ):
                error_type = "auth"
            elif "timeout" in error_str or "timed out" in error_str:
                error_type = "timeout"
            elif (
                "network" in error_str
                or "connection" in error_str
                or "503" in error_str
            ):
                error_type = "network"
            elif "invalid" in error_str or "400" in error_str:
                error_type = "invalid_request"
            else:
                error_type = "unknown"

            raise LLMProviderError(
                f"Anthropic API call failed: {e}",
                provider=self.name,
                error_type=error_type,
            ) from e

    def validate_config(self) -> list[str]:
        """Validate API key is configured."""
        if not self._api_key:
            return ["ANTHROPIC_API_KEY not set (export ANTHROPIC_API_KEY=sk-ant-...)"]
        return []
