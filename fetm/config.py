"""Configuration loader — single source of truth for all parameters."""

from pathlib import Path
from typing import Any

import yaml

_CONFIG_CACHE: dict[str, Any] | None = None
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration, caching on first call.

    Args:
        path: Path to settings.yaml. Defaults to config/settings.yaml
              relative to the project root.

    Returns:
        Configuration dictionary.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE

    config_path = Path(path) if path else _DEFAULT_PATH
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if path is None:
        _CONFIG_CACHE = config
    return config


def reset_config_cache() -> None:
    """Clear cached config (useful for testing)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
