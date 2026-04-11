"""Plotly chart builders for the dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fetm.dashboard.utils.theme import (
    STRATEGY_COLORS,
    STRATEGY_LABELS,
    apply_chart_defaults,
)


def equity_curve_chart(
    results: pd.DataFrame,
    crisis_periods: dict | None = None,
    in_sample_end: str | None = None,
) -> go.Figure:
    """Create interactive equity curve chart.

    Args:
        results: Backtest results DataFrame.
        crisis_periods: Dict of crisis periods for shading.
        in_sample_end: Date string for IS/OOS divider.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    for strat, color in STRATEGY_COLORS.items():
        col = f"cumreturn_{strat}"
        if col in results.columns:
            fig.add_trace(go.Scatter(
                x=results.index,
                y=results[col],
                name=STRATEGY_LABELS[strat],
                line=dict(color=color, width=2 if strat == "fetm" else 1.2),
                opacity=1.0 if strat == "fetm" else 0.7,
            ))

    # Add crisis period shading
    if crisis_periods:
        for crisis_id, period in crisis_periods.items():
            fig.add_vrect(
                x0=period["start"], x1=period["end"],
                fillcolor="rgba(239,68,68,0.08)", line_width=0,
                annotation_text=period.get("name", ""),
                annotation_position="top left",
                annotation_font_size=8,
                annotation_font_color="#EF4444",
            )

    # Add IS/OOS divider
    if in_sample_end:
        fig.add_vline(
            x=in_sample_end, line_dash="dash",
            line_color="#6B7280", line_width=1,
            annotation_text="IS/OOS",
            annotation_position="top",
        )

    fig.update_yaxes(type="log", title="Cumulative Return")
    return apply_chart_defaults(fig, "Equity Curves (Log Scale)")


def drawdown_chart(results: pd.DataFrame) -> go.Figure:
    """Create underwater drawdown chart."""
    fig = go.Figure()

    for strat, color in STRATEGY_COLORS.items():
        col = f"cumreturn_{strat}"
        if col in results.columns:
            cum = results[col]
            dd = cum / cum.cummax() - 1
            fig.add_trace(go.Scatter(
                x=results.index, y=dd,
                name=STRATEGY_LABELS[strat],
                fill="tozeroy",
                line=dict(color=color, width=0.5),
                fillcolor=color.replace(")", ",0.2)").replace("rgb", "rgba") if "rgb" in color else color + "33",
            ))

    fig.update_yaxes(title="Drawdown")
    return apply_chart_defaults(fig, "Drawdowns")


def rolling_sharpe_chart(results: pd.DataFrame, window: int = 252) -> go.Figure:
    """Create rolling Sharpe ratio chart."""
    fig = go.Figure()

    for strat, color in STRATEGY_COLORS.items():
        col = f"return_{strat}"
        if col in results.columns:
            ret = results[col]
            roll_mean = ret.rolling(window).mean() * 252
            roll_std = ret.rolling(window).std() * np.sqrt(252)
            roll_sharpe = (roll_mean - 0.02) / roll_std.clip(lower=1e-6)
            fig.add_trace(go.Scatter(
                x=results.index, y=roll_sharpe,
                name=STRATEGY_LABELS[strat],
                line=dict(color=color, width=1.5 if strat == "fetm" else 1),
            ))

    fig.add_hline(y=0, line_dash="dash", line_color="#6B7280", line_width=0.5)
    fig.add_hline(y=1, line_dash="dot", line_color="#10B981", line_width=0.5,
                  annotation_text="SR=1.0")
    fig.update_yaxes(title="Sharpe Ratio")
    return apply_chart_defaults(fig, f"Rolling {window // 21}M Sharpe Ratio")


def monthly_heatmap(results: pd.DataFrame, strategy: str = "fetm") -> go.Figure:
    """Create monthly returns heatmap.

    Args:
        results: Backtest results.
        strategy: Strategy to display.

    Returns:
        Plotly figure.
    """
    col = f"return_{strategy}"
    if col not in results.columns:
        return go.Figure()

    monthly = results[col].resample("ME").sum()
    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = df.pivot(index="year", columns="month", values="return")

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Create text annotations
    text = []
    for row in pivot.values:
        text.append([f"{v:.1%}" if not np.isnan(v) else "" for v in row])

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=month_labels,
        y=pivot.index.astype(str),
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorscale="RdYlGn",
        zmin=-0.10, zmax=0.10,
        colorbar=dict(title="Return", tickformat=".0%"),
    ))

    fig.update_yaxes(autorange="reversed")
    return apply_chart_defaults(fig, f"Monthly Returns — {STRATEGY_LABELS.get(strategy, strategy)}")


def annual_returns_bar(results: pd.DataFrame) -> go.Figure:
    """Create grouped annual returns bar chart."""
    fig = go.Figure()

    for strat, color in STRATEGY_COLORS.items():
        col = f"return_{strat}"
        if col in results.columns:
            annual = results[col].resample("YE").sum()
            fig.add_trace(go.Bar(
                x=annual.index.year,
                y=annual.values,
                name=STRATEGY_LABELS[strat],
                marker_color=color,
                opacity=0.8,
            ))

    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Annual Return", tickformat=".0%")
    fig.update_xaxes(title="Year")
    return apply_chart_defaults(fig, "Annual Returns by Strategy")


def return_distribution(results: pd.DataFrame) -> go.Figure:
    """Create overlaid histograms of daily returns."""
    fig = go.Figure()

    for strat, color in STRATEGY_COLORS.items():
        col = f"return_{strat}"
        if col in results.columns:
            ret = results[col].dropna()
            fig.add_trace(go.Histogram(
                x=ret,
                name=STRATEGY_LABELS[strat],
                marker_color=color,
                opacity=0.5,
                nbinsx=100,
            ))

    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Daily Return", tickformat=".1%")
    fig.update_yaxes(title="Frequency")
    return apply_chart_defaults(fig, "Return Distribution")


def parameter_drift_chart(
    per_window: pd.DataFrame,
    param_columns: list[str],
    title: str = "Optimal Parameter Drift",
) -> go.Figure:
    """Line chart of optimal parameter values across walk-forward windows.

    Args:
        per_window: DataFrame from ``walk_forward_results.parquet`` (one row
            per walk-forward window, columns prefixed ``param.``).
        param_columns: Fully-qualified column names (with ``param.`` prefix)
            to plot. Each column becomes one trace.
        title: Chart title.
    """
    fig = go.Figure()
    if per_window.empty or not param_columns:
        return apply_chart_defaults(fig, title)

    x_col = "oos_end" if "oos_end" in per_window.columns else "window"
    x = per_window[x_col]

    palette = [
        "#10B981", "#3B82F6", "#F59E0B", "#EF4444", "#8B5CF6",
        "#EC4899", "#14B8A6", "#F97316", "#6366F1", "#84CC16",
        "#06B6D4", "#A855F7", "#EAB308",
    ]
    for i, col in enumerate(param_columns):
        if col not in per_window.columns:
            continue
        label = col.removeprefix("param.")
        fig.add_trace(go.Scatter(
            x=x, y=per_window[col],
            mode="lines+markers",
            name=label,
            line=dict(color=palette[i % len(palette)], width=1.5),
            marker=dict(size=5),
            yaxis=f"y{i + 1}" if i == 0 else None,
        ))

    fig.update_xaxes(title="Walk-forward window (OOS end)")
    fig.update_yaxes(title="Value")
    return apply_chart_defaults(fig, title)


def qq_plot(results: pd.DataFrame, strategy: str = "fetm") -> go.Figure:
    """Create QQ-plot of strategy returns vs normal distribution."""
    from scipy import stats

    col = f"return_{strategy}"
    if col not in results.columns:
        return go.Figure()

    ret = results[col].dropna().values
    (theoretical_q, sample_q), (slope, intercept, _) = stats.probplot(ret, dist="norm")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical_q, y=sample_q,
        mode="markers", marker=dict(size=2, color=STRATEGY_COLORS.get(strategy, "#10B981")),
        name="Observed",
    ))
    # Reference line
    x_line = np.array([theoretical_q.min(), theoretical_q.max()])
    fig.add_trace(go.Scatter(
        x=x_line, y=slope * x_line + intercept,
        mode="lines", line=dict(color="#EF4444", dash="dash"),
        name="Normal Reference",
    ))

    fig.update_xaxes(title="Theoretical Quantiles")
    fig.update_yaxes(title="Sample Quantiles")
    return apply_chart_defaults(fig, f"QQ-Plot — {STRATEGY_LABELS.get(strategy, strategy)}")
