"""Styled metric tables for dashboard."""

import numpy as np
import pandas as pd

from fetm.dashboard.utils.theme import STRATEGY_LABELS


def metrics_table(metrics: dict) -> pd.DataFrame:
    """Create formatted strategy comparison table.

    Args:
        metrics: Dict from metrics.json['strategies'].

    Returns:
        Styled DataFrame with strategies as columns.
    """
    strategies = ["buyhold", "linear", "binary", "fetm"]
    display_metrics = [
        ("Annualized Return", "annualized_return", "{:.2%}"),
        ("Annualized Volatility", "annualized_vol", "{:.2%}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.3f}"),
        ("Sortino Ratio", "sortino_ratio", "{:.3f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("Max DD Duration (days)", "max_drawdown_duration_days", "{:.0f}"),
        ("Calmar Ratio", "calmar_ratio", "{:.3f}"),
        ("VaR (95%)", "var_95", "{:.4f}"),
        ("CVaR (95%)", "cvar_95", "{:.4f}"),
        ("Skewness", "skewness", "{:.3f}"),
        ("Kurtosis", "kurtosis", "{:.2f}"),
        ("Annual Turnover", "annual_turnover", "{:.2f}"),
        ("Avg Holding Period", "avg_holding_period", "{:.1f}"),
        ("% Time Long", "pct_time_long", "{:.1%}"),
        ("% Time Short", "pct_time_short", "{:.1%}"),
        ("% Time Flat", "pct_time_flat", "{:.1%}"),
        ("Win Rate (Monthly)", "win_rate_monthly", "{:.1%}"),
        ("Profit Factor", "profit_factor", "{:.3f}"),
    ]

    rows = []
    for label, key, fmt in display_metrics:
        row = {"Metric": label}
        for strat in strategies:
            v = metrics.get(strat, {}).get(key)
            if v is not None:
                try:
                    row[STRATEGY_LABELS[strat]] = fmt.format(v)
                except (ValueError, TypeError):
                    row[STRATEGY_LABELS[strat]] = str(v)
            else:
                row[STRATEGY_LABELS[strat]] = "N/A"
        rows.append(row)

    return pd.DataFrame(rows).set_index("Metric")


def crisis_table(crisis_performance: dict, crisis_periods: dict | None = None) -> pd.DataFrame:
    """Create crisis performance comparison table.

    Args:
        crisis_performance: Dict from metrics.json['crisis_performance'].
        crisis_periods: Optional config crisis periods for names.

    Returns:
        DataFrame with crisis periods as rows, strategies as columns.
    """
    rows = []
    for crisis_id, perf in crisis_performance.items():
        name = crisis_id
        if crisis_periods and crisis_id in crisis_periods:
            name = crisis_periods[crisis_id].get("name", crisis_id)

        row = {"Crisis Period": name}
        for strat in ["buyhold", "linear", "binary", "fetm"]:
            v = perf.get(strat)
            if v is not None:
                row[STRATEGY_LABELS.get(strat, strat)] = f"{v:.2%}"
            else:
                row[STRATEGY_LABELS.get(strat, strat)] = "N/A"
        rows.append(row)

    return pd.DataFrame(rows).set_index("Crisis Period")


def conditional_table(conditional_performance: dict) -> pd.DataFrame:
    """Create conditional performance table (by benchmark tercile).

    Args:
        conditional_performance: Dict from metrics.json['conditional_performance'].

    Returns:
        DataFrame with terciles as rows, strategy Sharpe as columns.
    """
    tercile_labels = {
        "bottom_tercile": "Bottom Tercile",
        "middle_tercile": "Middle Tercile",
        "top_tercile": "Top Tercile",
    }

    rows = []
    for tercile_key, label in tercile_labels.items():
        perf = conditional_performance.get(tercile_key, {})
        row = {"Benchmark Regime": label}
        for strat in ["buyhold", "linear", "binary", "fetm"]:
            v = perf.get(strat)
            if v is not None:
                row[STRATEGY_LABELS.get(strat, strat)] = f"{v:.3f}"
            else:
                row[STRATEGY_LABELS.get(strat, strat)] = "N/A"
        rows.append(row)

    return pd.DataFrame(rows).set_index("Benchmark Regime")
