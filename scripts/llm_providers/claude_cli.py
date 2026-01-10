"""
Claude Code CLI Provider

Invokes Claude Code CLI via subprocess for LLM processing.
Migrated from original llm_utils.py implementation.

External Dependency: Requires `claude` CLI installed and authenticated.
Billing: Uses user's Claude subscription quota.

Security:
- Temp file used to avoid command-line injection
- Timeout enforced to prevent indefinite hangs
- Temp file cleaned up even on error (finally block)
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from . import LLMProvider, LLMProviderError, LLMResponse


class ClaudeCliProvider(LLMProvider):
    """
    Claude Code CLI provider implementation.

    Invokes the `claude` command-line tool via subprocess.
    """

    def __init__(self, default_model: str = "sonnet"):
        """
        Initialize Claude CLI provider.

        Args:
            default_model: Default model name (sonnet, opus, haiku)
        """
        self._default_model = default_model
        self._cli_path: Optional[str] = None

    @property
    def name(self) -> str:
        return "claude-cli"

    @property
    def is_local(self) -> bool:
        # Claude CLI sends data to external service
        return False

    def is_available(self) -> bool:
        """Check if Claude CLI is installed."""
        if self._cli_path is not None:
            return True

        result = subprocess.run(
            ["which", "claude"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            self._cli_path = result.stdout.strip()
            return True
        return False

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout: int = 120,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate text using Claude Code CLI.

        Args:
            prompt: User prompt
            model: Model name (sonnet, opus, haiku)
            timeout: Timeout in seconds
            temperature: Sampling temperature (Note: CLI may not support)
            max_tokens: Max tokens (Note: CLI may not support)

        Returns:
            LLMResponse with generated content

        Raises:
            LLMProviderError: If CLI invocation fails
        """
        if not self.is_available():
            raise LLMProviderError(
                "Claude CLI not installed. Install: https://code.claude.com/docs/en/quickstart",
                provider=self.name,
                error_type="unavailable",
            )

        model_to_use = model or self._default_model

        # Write prompt to temp file to avoid shell injection
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                temp_file = Path(f.name)
                f.write(prompt)

            # Invoke claude with prompt file via stdin
            # Note: Using cat | claude pattern for stdin compatibility
            # --print flag is REQUIRED for non-interactive (piped) usage
            command = f"cat {temp_file} | claude --model {model_to_use} --print"
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                # Normalize error type based on stderr
                stderr = result.stderr.lower()
                if "rate limit" in stderr or "quota" in stderr:
                    error_type = "rate_limit"
                elif "authentication" in stderr or "api key" in stderr:
                    error_type = "auth"
                elif "timeout" in stderr:
                    error_type = "timeout"
                else:
                    error_type = "unknown"

                raise LLMProviderError(
                    f"Claude CLI invocation failed: {result.stderr}",
                    provider=self.name,
                    error_type=error_type,
                )

            return LLMResponse(
                content=result.stdout,
                model=model_to_use,
                provider=self.name,
                tokens_used=None,  # CLI doesn't report token usage
                finish_reason="stop",
                raw_response=None,
            )

        except subprocess.TimeoutExpired as e:
            raise LLMProviderError(
                f"Claude CLI timeout after {timeout}s",
                provider=self.name,
                error_type="timeout",
            ) from e
        except Exception as e:
            # Catch-all for unexpected errors
            if isinstance(e, LLMProviderError):
                raise
            raise LLMProviderError(
                f"Claude CLI unexpected error: {e}",
                provider=self.name,
                error_type="unknown",
            ) from e
        finally:
            # Cleanup temp file
            if temp_file and temp_file.exists():
                temp_file.unlink()

    def validate_config(self) -> list[str]:
        """Validate CLI is installed."""
        if not self.is_available():
            return ["Claude CLI not installed (run: which claude)"]
        return []
