"""
LM Studio Provider

Uses OpenAI-compatible HTTP API for local LM Studio server.
This is a REQUIRED provider for local/privacy-conscious usage.

External Dependency: LM Studio running locally
API Endpoint: http://localhost:1234 (default)
Billing: Free (local execution)

Privacy: All processing happens locally - no data sent to external services.

LM Studio Setup:
1. Download LM Studio from https://lmstudio.ai/
2. Load a model (e.g., Llama, Mistral, etc.)
3. Start the local server (default port 1234)
4. Server provides OpenAI-compatible API at http://localhost:1234/v1
"""

from __future__ import annotations

import json
import os
from typing import Optional

from . import LLMProvider, LLMProviderError, LLMResponse


class LmStudioProvider(LLMProvider):
    """
    LM Studio local server implementation.

    Uses OpenAI-compatible HTTP API. No SDK needed - pure HTTP calls.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        """
        Initialize LM Studio provider.

        Args:
            endpoint: Server endpoint URL (default: http://localhost:1234)
            default_model: Default model name (default: uses server's loaded model)

        Notes:
            - Endpoint can also be set via LMSTUDIO_ENDPOINT env var
            - Model name is optional - LM Studio uses whichever model is loaded
        """
        self._endpoint = endpoint or os.getenv(
            "LMSTUDIO_ENDPOINT", "http://localhost:1234"
        )
        # Ensure endpoint has /v1 suffix for OpenAI compatibility
        if not self._endpoint.endswith("/v1"):
            self._endpoint = f"{self._endpoint}/v1"

        self._default_model = default_model
        self._httpx_client = None  # Lazy-loaded

    @property
    def name(self) -> str:
        return "lmstudio"

    @property
    def is_local(self) -> bool:
        # LM Studio runs locally - privacy-conscious
        return True

    def _get_client(self):
        """Lazy-load httpx client."""
        if self._httpx_client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError(
                    "httpx not installed. Install with: pip install -e '.[llm-local]'"
                ) from e

            self._httpx_client = httpx.Client(timeout=None)  # Use custom timeout
        return self._httpx_client

    def is_available(self) -> bool:
        """
        Check if LM Studio server is running.

        Attempts a quick health check by listing models.
        """
        try:
            import httpx

            # Quick check: can we reach the server?
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self._endpoint}/models")
                return response.status_code == 200
        except (ImportError, Exception):
            return False

    def _get_loaded_model(self) -> Optional[str]:
        """
        Get the first loaded model from LM Studio.

        Returns:
            Model ID string, or None if no models available
        """
        try:
            import httpx

            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self._endpoint}/models")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    if models:
                        # Return first non-embedding model
                        for model in models:
                            model_id = model.get("id", "")
                            if "embedding" not in model_id.lower():
                                return model_id
                        # Fallback: return first model even if it's embedding
                        return models[0].get("id")
        except Exception:
            pass
        return None

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout: int = 120,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate text using LM Studio's OpenAI-compatible API.

        Args:
            prompt: User prompt
            model: Model name (optional - uses loaded model if None)
            timeout: Timeout in seconds
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate (default 2048)

        Returns:
            LLMResponse with generated content

        Raises:
            LLMProviderError: If API call fails

        Notes:
            - LM Studio uses OpenAI chat completion format
            - Model name is optional - uses whichever model is currently loaded
        """
        client = self._get_client()

        # Auto-detect loaded model if not specified
        model_to_use = model or self._default_model or self._get_loaded_model()

        if not model_to_use:
            raise LLMProviderError(
                "No model specified and no loaded model found in LM Studio. "
                "Load a model in LM Studio first.",
                provider=self.name,
                error_type="unavailable",
            )

        max_tokens_to_use = max_tokens or 2048

        # OpenAI-compatible chat completion format
        request_data = {
            "model": model_to_use,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens_to_use,
        }

        try:
            response = client.post(
                f"{self._endpoint}/chat/completions",
                json=request_data,
                timeout=timeout,
            )

            # Check for errors
            if response.status_code != 200:
                error_detail = response.text
                if response.status_code == 404:
                    error_type = "unavailable"
                    error_msg = (
                        f"LM Studio server not found at {self._endpoint}. "
                        "Is LM Studio running?"
                    )
                elif response.status_code == 500:
                    error_type = "unknown"
                    error_msg = f"LM Studio server error: {error_detail}"
                else:
                    error_type = "unknown"
                    error_msg = (
                        f"LM Studio HTTP {response.status_code}: {error_detail}"
                    )

                raise LLMProviderError(error_msg, provider=self.name, error_type=error_type)

            # Parse OpenAI-compatible response
            response_data = response.json()

            # Extract content from choices
            if "choices" not in response_data or not response_data["choices"]:
                raise LLMProviderError(
                    "Empty response from LM Studio",
                    provider=self.name,
                    error_type="unknown",
                )

            first_choice = response_data["choices"][0]
            message = first_choice.get("message", {})
            content = message.get("content", "")

            if not content:
                raise LLMProviderError(
                    "No content in LM Studio response",
                    provider=self.name,
                    error_type="unknown",
                )

            # Extract token usage (if available)
            tokens_used = None
            if "usage" in response_data:
                usage = response_data["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                tokens_used = prompt_tokens + completion_tokens

            # Extract finish reason
            finish_reason = first_choice.get("finish_reason")

            # Get actual model used (LM Studio returns this)
            actual_model = response_data.get("model", model_to_use)

            return LLMResponse(
                content=content,
                model=actual_model,
                provider=self.name,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                raw_response=response_data,
            )

        except ImportError:
            # Re-raise ImportError from _get_client()
            raise
        except Exception as e:
            # Normalize errors
            if isinstance(e, LLMProviderError):
                raise

            error_str = str(e).lower()

            # Detect error type
            if "timeout" in error_str or "timed out" in error_str:
                error_type = "timeout"
            elif (
                "connection" in error_str
                or "refused" in error_str
                or "unreachable" in error_str
            ):
                error_type = "network"
                error_str = (
                    f"{e}. Is LM Studio running on {self._endpoint.replace('/v1', '')}?"
                )
            else:
                error_type = "unknown"

            raise LLMProviderError(
                f"LM Studio API call failed: {error_str}",
                provider=self.name,
                error_type=error_type,
            ) from e

    def validate_config(self) -> list[str]:
        """Validate LM Studio server is reachable."""
        if not self.is_available():
            return [
                f"LM Studio server not reachable at {self._endpoint.replace('/v1', '')}. "
                "Start LM Studio and load a model."
            ]
        return []

    def __del__(self):
        """Cleanup httpx client on destruction."""
        if self._httpx_client is not None:
            try:
                self._httpx_client.close()
            except Exception:
                pass
