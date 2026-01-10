"""
LLM Configuration Management

Handles loading provider configuration from YAML config file or environment variables.

Configuration Priority:
1. YAML config file (.llm_config.yaml by default, or via --llm-config param)
2. Environment variables (fallback)
3. Auto-detection (if no explicit configuration)

YAML Config Format:
  default_provider: lmstudio  # or anthropic-api, claude-cli, etc.

  providers:
    lmstudio:
      endpoint: http://192.168.12.60:1234
    anthropic-api:
      api_key: sk-ant-...

Environment Variables (fallback):
- MEMEX_LLM_PROVIDER: Default provider name
- ANTHROPIC_API_KEY: Anthropic API key
- OPENAI_API_KEY: OpenAI API key
- GOOGLE_API_KEY: Google Gemini API key
- OLLAMA_ENDPOINT: Ollama server URL
- LMSTUDIO_ENDPOINT: LM Studio server URL
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ProviderConfig:
    """
    Configuration for a single LLM provider.

    Attributes:
        name: Provider identifier
        enabled: Whether provider is enabled
        api_key: API key (for cloud providers)
        endpoint: Server endpoint URL (for local providers)
        default_model: Default model to use
        timeout: Default timeout in seconds
        temperature: Default temperature (0-1)
        max_tokens: Default max tokens
    """

    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    default_model: Optional[str] = None
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: Optional[int] = None


@dataclass
class LLMConfig:
    """
    Global LLM configuration.

    Attributes:
        default_provider: Default provider name (None = auto-detect)
        providers: Dictionary of provider-specific configurations
    """

    default_provider: Optional[str] = None
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """
        Load configuration from environment variables.

        Returns:
            LLMConfig instance with env var settings

        Examples:
            >>> os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
            >>> config = LLMConfig.from_env()
            >>> print(config.providers["anthropic-api"].api_key)
        """
        config = cls()

        # Default provider
        if default_provider := os.getenv("MEMEX_LLM_PROVIDER"):
            config.default_provider = default_provider

        # Anthropic API
        if api_key := os.getenv("ANTHROPIC_API_KEY"):
            config.providers["anthropic-api"] = ProviderConfig(
                name="anthropic-api",
                api_key=api_key,
            )

        # OpenAI API (Phase 2)
        if api_key := os.getenv("OPENAI_API_KEY"):
            config.providers["openai-api"] = ProviderConfig(
                name="openai-api",
                api_key=api_key,
            )

        # Google Gemini (Phase 2)
        if api_key := os.getenv("GOOGLE_API_KEY"):
            config.providers["gemini"] = ProviderConfig(
                name="gemini",
                api_key=api_key,
            )

        # Ollama (Phase 2)
        if endpoint := os.getenv("OLLAMA_ENDPOINT"):
            config.providers["ollama"] = ProviderConfig(
                name="ollama",
                endpoint=endpoint,
            )

        # LM Studio (Phase 2)
        if endpoint := os.getenv("LMSTUDIO_ENDPOINT"):
            config.providers["lmstudio"] = ProviderConfig(
                name="lmstudio",
                endpoint=endpoint,
            )

        return config

    @classmethod
    def from_yaml(cls, config_path: Path) -> "LLMConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            LLMConfig instance with YAML settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is malformed

        Examples:
            >>> config = LLMConfig.from_yaml(Path(".llm_config.yaml"))
            >>> print(config.default_provider)
        """
        if not config_path.exists():
            raise FileNotFoundError(f"LLM config file not found: {config_path}")

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

        if not data:
            # Empty file
            return cls()

        config = cls()

        # Default provider
        if "default_provider" in data:
            config.default_provider = data["default_provider"]

        # Providers
        if "providers" in data and isinstance(data["providers"], dict):
            for provider_name, provider_data in data["providers"].items():
                if not isinstance(provider_data, dict):
                    continue

                config.providers[provider_name] = ProviderConfig(
                    name=provider_name,
                    enabled=provider_data.get("enabled", True),
                    api_key=provider_data.get("api_key"),
                    endpoint=provider_data.get("endpoint"),
                    default_model=provider_data.get("default_model"),
                    timeout=provider_data.get("timeout", 120),
                    temperature=provider_data.get("temperature", 0.7),
                    max_tokens=provider_data.get("max_tokens"),
                )

        return config

    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """
        Get configuration for a specific provider.

        Args:
            provider_name: Provider identifier

        Returns:
            ProviderConfig if configured, None otherwise

        Examples:
            >>> config = LLMConfig.from_env()
            >>> anthropic_config = config.get_provider_config("anthropic-api")
        """
        return self.providers.get(provider_name)


def load_config(config_path: Optional[Path] = None) -> LLMConfig:
    """
    Load LLM configuration.

    Priority order:
    1. YAML config file (if config_path provided or .llm_config.yaml exists)
    2. Environment variables (fallback)

    Args:
        config_path: Optional path to YAML config file. If None, checks for
                    .llm_config.yaml in current directory.

    Returns:
        LLMConfig instance

    Examples:
        >>> config = load_config()  # Auto-detect .llm_config.yaml or use env vars
        >>> config = load_config(Path("custom_config.yaml"))  # Explicit path
    """
    # Priority 1: Explicit config path
    if config_path and config_path.exists():
        return LLMConfig.from_yaml(config_path)

    # Priority 2: Default .llm_config.yaml in current directory
    default_config = Path.cwd() / ".llm_config.yaml"
    if default_config.exists():
        return LLMConfig.from_yaml(default_config)

    # Priority 3: Environment variables (fallback)
    return LLMConfig.from_env()
