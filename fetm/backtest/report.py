"""Generate static reports and plots from backtest results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fetm.visualization.style import (
    STRATEGY_COLORS,
    STRATEGY_LABELS,
    VOL_ESTIMATOR_COLORS,
    HORIZON_COLORS,
    setup_style,
)

logger = logging.getLogger(__name__)


def generate_report(run_dir: str | Path) -> None:
    """Generate all plots and text summary for a backtest run.

    Args:
        run_dir: Path to the run output directory.
    """
    run_dir = Path(run_dir)
    setup_style()

    results = pd.read_parquet(run_dir / "results.parquet")
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    logger.info("Generating plots in %s", plots_dir)

    _plot_equity_curves(results, plots_dir)
    _plot_drawdowns(results, plots_dir)
    _plot_rolling_sharpe(results, plots_dir)
    _plot_volatility_comparison(results, plots_dir)
    _plot_signal_scatter(results, plots_dir)
    _plot_signal_timeseries(results, plots_dir)
    _plot_exit_time_distribution(results, plots_dir)
    _plot_monthly_heatmap(results, plots_dir)
    _plot_corridor_sensitivity(metrics, plots_dir)

    _write_text_summary(metrics, run_dir)
    logger.info("Report generation complete.")


def _plot_equity_curves(results: pd.DataFrame, plots_dir: Path) -> None:
    """Plot cumulative return equity curves for all strategies."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for strat, color in STRATEGY_COLORS.items():
        col = f"cumreturn_{strat}"
        if col in results.columns:
            ax.plot(results.index, results[col], color=color,
                    label=STRATEGY_LABELS[strat],
                    linewidth=1.5 if strat == "fetm" else 1.0,
                    alpha=1.0 if strat == "fetm" else 0.7)

    ax.set_yscale("log")
    ax.set_title("Equity Curves (Log Scale)")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "equity_curves.png")
    plt.close(fig)


def _plot_drawdowns(results: pd.DataFrame, plots_dir: Path) -> None:
    """Plot underwater (drawdown) chart."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for strat, color in STRATEGY_COLORS.items():
        col = f"cumreturn_{strat}"
        if col in results.columns:
            cum = results[col]
            dd = cum / cum.cummax() - 1
            ax.fill_between(results.index, dd, 0, alpha=0.3, color=color,
                            label=STRATEGY_LABELS[strat])

    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(plots_dir / "drawdowns.png")
    plt.close(fig)


def _plot_rolling_sharpe(results: pd.DataFrame, plots_dir: Path) -> None:
    """Plot rolling 12-month Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(14, 5))
    window = 252

    for strat, color in STRATEGY_COLORS.items():
        col = f"return_{strat}"
        if col in results.columns:
            ret = results[col].dropna()
            roll_mean = ret.rolling(window).mean() * 252
            roll_std = ret.rolling(window).std() * np.sqrt(252)
            roll_sharpe = (roll_mean - 0.02) / roll_std.clip(lower=1e-6)
            ax.plot(roll_sharpe.index, roll_sharpe, color=color,
                    label=STRATEGY_LABELS[strat], alpha=0.8)

    ax.axhline(0, color="#6B7280", linestyle="--", alpha=0.5)
    ax.axhline(1, color="#10B981", linestyle="--", alpha=0.3)
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "rolling_sharpe.png")
    plt.close(fig)


def _plot_volatility_comparison(results: pd.DataFrame, plots_dir: Path) -> None:
    """Plot all volatility estimators."""
    fig, ax = plt.subplots(figsize=(14, 5))

    vol_cols = {"ewma": "vol_ewma", "parkinson": "vol_parkinson",
                "fet": "vol_fet", "composite": "vol_composite"}
    for name, col in vol_cols.items():
        if col in results.columns:
            ax.plot(results.index, results[col], color=VOL_ESTIMATOR_COLORS[name],
                    label=name.upper(), alpha=0.8,
                    linewidth=1.5 if name == "composite" else 1.0)

    ax.set_title("Volatility Estimator Comparison")
    ax.set_ylabel("Annualized Volatility")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "volatility_comparison.png")
    plt.close(fig)


def _plot_signal_scatter(results: pd.DataFrame, plots_dir: Path) -> None:
    """Plot S-curve: raw signal vs transformed signal."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Theoretical S-curve
    s = np.linspace(-5, 5, 500)
    f_s = 0.394 * s / (s ** 2 + 1)
    ax.plot(s, f_s, color="#E0E0E0", linewidth=2, label="Theoretical S-curve", zorder=5)

    for h, color in HORIZON_COLORS.items():
        raw_col = f"signal_raw_{h}"
        nl_col = f"signal_nl_{h}"
        if raw_col in results.columns and nl_col in results.columns:
            raw = results[raw_col].dropna()
            nl = results[nl_col].dropna()
            common = raw.index.intersection(nl.index)
            ax.scatter(raw.loc[common], nl.loc[common], color=color, alpha=0.1,
                       s=2, label=f"{h} horizon")

    ax.set_xlabel("Raw Signal s")
    ax.set_ylabel("Transformed Signal f(s)")
    ax.set_title("S-Curve Transformation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "signal_scatter.png")
    plt.close(fig)


def _plot_signal_timeseries(results: pd.DataFrame, plots_dir: Path) -> None:
    """Plot signal and position time series."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    if "signal_combined" in results.columns:
        ax1.plot(results.index, results["signal_combined"], color="#10B981", alpha=0.8)
        ax1.set_title("Combined FETM Signal")
        ax1.set_ylabel("Signal")
        ax1.axhline(0, color="#6B7280", linestyle="--", alpha=0.5)

    if "position_fetm" in results.columns:
        ax2.plot(results.index, results["position_fetm"], color="#10B981", alpha=0.8)
        ax2.set_title("FETM Position")
        ax2.set_ylabel("Position")
        ax2.axhline(0, color="#6B7280", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(plots_dir / "signal_timeseries.png")
    plt.close(fig)


def _plot_exit_time_distribution(results: pd.DataFrame, plots_dir: Path) -> None:
    """Plot distribution of exit times."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if "exit_time_last" in results.columns:
        exit_times = results["exit_time_last"].dropna()
        if len(exit_times) > 10:
            ax.hist(exit_times * 252, bins=50, color="#F472B6", alpha=0.7,
                    edgecolor="none")
            ax.set_xlabel("Exit Time (trading days)")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Exit Time Distribution (n={len(exit_times)})")

    fig.tight_layout()
    fig.savefig(plots_dir / "exit_time_distribution.png")
    plt.close(fig)


def _plot_monthly_heatmap(results: pd.DataFrame, plots_dir: Path) -> None:
    """Plot monthly returns heatmap for FETM strategy."""
    fig, ax = plt.subplots(figsize=(12, 8))

    ret_col = "return_fetm"
    if ret_col not in results.columns:
        plt.close(fig)
        return

    monthly = results[ret_col].resample("ME").sum()
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = monthly_df.pivot(index="year", columns="month", values="return")

    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=7, color="black" if abs(val) < 0.05 else "white")

    ax.set_title("FETM Monthly Returns")
    fig.colorbar(im, ax=ax, label="Return", shrink=0.8)
    fig.tight_layout()
    fig.savefig(plots_dir / "monthly_heatmap.png")
    plt.close(fig)


def _plot_corridor_sensitivity(metrics: dict, plots_dir: Path) -> None:
    """Plot corridor width sensitivity (if data available)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sens = metrics.get("corridor_sensitivity", {})
    deltas = sens.get("delta_values", [])
    sharpes = sens.get("sharpe_ratios", [])

    if deltas and sharpes and len(deltas) == len(sharpes):
        ax.plot(deltas, sharpes, "o-", color="#10B981")
        ax.set_xlabel("Corridor Width (delta)")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Corridor Width Sensitivity")
    else:
        ax.text(0.5, 0.5, "No sensitivity data available\nRun sensitivity analysis first",
                transform=ax.transAxes, ha="center", va="center", fontsize=12,
                color="#9CA3AF")

    fig.tight_layout()
    fig.savefig(plots_dir / "corridor_sensitivity.png")
    plt.close(fig)


def _write_text_summary(metrics: dict, run_dir: Path) -> None:
    """Write a human-readable text summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("FETM Phase 1 Backtest Summary")
    lines.append("=" * 70)
    lines.append(f"Period: {metrics['period']['start']} to {metrics['period']['end']}")
    lines.append(f"Trading Days: {metrics['period']['trading_days']}")
    lines.append(f"In-Sample End: {metrics['period']['in_sample_end']}")
    lines.append("")

    lines.append(f"{'Metric':<30} {'Buy&Hold':>10} {'Linear':>10} {'Binary':>10} {'FETM':>10}")
    lines.append("-" * 70)

    display_metrics = [
        ("Ann. Return", "annualized_return"),
        ("Ann. Volatility", "annualized_vol"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Sortino Ratio", "sortino_ratio"),
        ("Max Drawdown", "max_drawdown"),
        ("Max DD Duration", "max_drawdown_duration_days"),
        ("Calmar Ratio", "calmar_ratio"),
        ("VaR 95%", "var_95"),
        ("CVaR 95%", "cvar_95"),
        ("Skewness", "skewness"),
        ("Kurtosis", "kurtosis"),
        ("Annual Turnover", "annual_turnover"),
        ("Win Rate (Monthly)", "win_rate_monthly"),
        ("Profit Factor", "profit_factor"),
    ]

    for label, key in display_metrics:
        vals = []
        for strat in ["buyhold", "linear", "binary", "fetm"]:
            v = metrics["strategies"].get(strat, {}).get(key, "N/A")
            if isinstance(v, (int, float)):
                vals.append(f"{v:>10.4f}")
            else:
                vals.append(f"{v!s:>10}")
        lines.append(f"{label:<30} {''.join(vals)}")

    lines.append("")
    lines.append("Crisis Performance:")
    lines.append("-" * 70)
    for crisis_id, perf in metrics.get("crisis_performance", {}).items():
        line = f"  {crisis_id:<20}"
        for strat in ["buyhold", "linear", "binary", "fetm"]:
            v = perf.get(strat)
            if v is not None:
                line += f" {v:>10.4f}"
            else:
                line += f" {'N/A':>10}"
        lines.append(line)

    text = "\n".join(lines)
    (run_dir / "metrics_summary.txt").write_text(text)
    logger.info("Wrote metrics_summary.txt")
