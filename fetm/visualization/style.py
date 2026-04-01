"""Consistent plot styling for static matplotlib charts and dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl

# Strategy colors (consistent across all visualizations)
STRATEGY_COLORS = {
    "buyhold": "#6B7280",
    "linear": "#3B82F6",
    "binary": "#F59E0B",
    "fetm": "#10B981",
}

STRATEGY_LABELS = {
    "buyhold": "Buy & Hold",
    "linear": "Linear TSMOM",
    "binary": "Binary TSMOM",
    "fetm": "FETM",
}

VOL_ESTIMATOR_COLORS = {
    "ewma": "#9CA3AF",
    "parkinson": "#A78BFA",
    "fet": "#F472B6",
    "composite": "#10B981",
}

HORIZON_COLORS = {
    "21d": "#60A5FA",
    "63d": "#FBBF24",
    "252d": "#34D399",
}

REGIME_COLORS = {
    "crisis": "rgba(239, 68, 68, 0.15)",
    "high_vol": "rgba(251, 191, 36, 0.10)",
    "normal": "rgba(0, 0, 0, 0)",
    "low_vol": "rgba(96, 165, 250, 0.08)",
}

# Matplotlib dark style
DARK_STYLE = {
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#0E1117",
    "axes.edgecolor": "#2A2D3E",
    "axes.labelcolor": "#E0E0E0",
    "text.color": "#E0E0E0",
    "xtick.color": "#9CA3AF",
    "ytick.color": "#9CA3AF",
    "grid.color": "#2A2D3E",
    "grid.alpha": 0.5,
    "legend.facecolor": "#1A1F2E",
    "legend.edgecolor": "#2A2D3E",
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.facecolor": "#0E1117",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
}


def setup_style() -> None:
    """Apply dark professional style to matplotlib."""
    for key, val in DARK_STYLE.items():
        mpl.rcParams[key] = val
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.linestyle"] = "--"
