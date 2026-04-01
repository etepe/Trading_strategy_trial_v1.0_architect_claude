"""Dashboard theme and chart styling constants."""

import plotly.graph_objects as go

CHART_DEFAULTS = {
    "template": "plotly_dark",
    "height": 450,
    "margin": dict(l=60, r=30, t=50, b=40),
    "font": dict(family="Inter, sans-serif", size=12),
    "hovermode": "x unified",
}

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

HORIZON_COLORS = {
    "21d": "#60A5FA",
    "63d": "#FBBF24",
    "252d": "#34D399",
}

VOL_ESTIMATOR_COLORS = {
    "ewma": "#9CA3AF",
    "parkinson": "#A78BFA",
    "fet": "#F472B6",
    "composite": "#10B981",
}

REGIME_COLORS = {
    "crisis": "rgba(239, 68, 68, 0.15)",
    "high_vol": "rgba(251, 191, 36, 0.10)",
    "normal": "rgba(0, 0, 0, 0)",
    "low_vol": "rgba(96, 165, 250, 0.08)",
}


def apply_chart_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent styling to a Plotly figure.

    Args:
        fig: Plotly figure to style.
        title: Chart title.

    Returns:
        Styled figure.
    """
    fig.update_layout(
        template=CHART_DEFAULTS["template"],
        height=CHART_DEFAULTS["height"],
        margin=CHART_DEFAULTS["margin"],
        font=CHART_DEFAULTS["font"],
        hovermode=CHART_DEFAULTS["hovermode"],
        title=dict(text=title, x=0.01, font=dict(size=16)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig
