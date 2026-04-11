"""Dot-path helpers for nested dictionaries.

Used by the backtest sensitivity analysis and the optimization package to
read/write deeply nested config values without hand-coding each path.
"""

from typing import Any


def set_nested(d: dict, key_path: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-separated key path.

    Example:
        set_nested(cfg, "volatility.exit_time.corridor_width", 0.015)
    """
    keys = key_path.split(".")
    cursor = d
    for k in keys[:-1]:
        cursor = cursor[k]
    cursor[keys[-1]] = value


def get_nested(d: dict, key_path: str, default: Any = None) -> Any:
    """Read a value from a nested dict using a dot-separated key path.

    Returns ``default`` if any segment of the path is missing.
    """
    cursor: Any = d
    for k in key_path.split("."):
        if not isinstance(cursor, dict) or k not in cursor:
            return default
        cursor = cursor[k]
    return cursor
