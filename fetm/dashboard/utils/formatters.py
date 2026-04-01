"""Number formatting helpers for dashboard display."""


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    """Format as percentage."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def fmt_ratio(value: float | None, decimals: int = 2) -> str:
    """Format as ratio (e.g., Sharpe)."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def fmt_days(value: int | float | None) -> str:
    """Format as integer days."""
    if value is None:
        return "N/A"
    return f"{int(value):,d}"


def fmt_number(value: float | None, decimals: int = 2) -> str:
    """Format generic number."""
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}"
