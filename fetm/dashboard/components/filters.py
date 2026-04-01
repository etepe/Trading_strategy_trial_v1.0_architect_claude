"""Sidebar filter widgets for dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st


def render_sidebar(run_data: dict) -> dict:
    """Render sidebar filters and return filter state.

    Args:
        run_data: Loaded run data dict.

    Returns:
        Filter state dict with keys: period_option, date_range.
    """
    st.sidebar.title("FETM Strategy")
    st.sidebar.markdown("---")

    # In-sample / Out-of-sample toggle
    in_sample_end = run_data.get("metrics", {}).get("period", {}).get("in_sample_end", "2014-12-31")
    period_option = st.sidebar.radio(
        "Period",
        ["Full Period", "In-Sample Only", "Out-of-Sample Only"],
        index=2,  # Default to OOS
    )

    # Date range
    results = run_data["results"]
    min_date = results.index.min().date()
    max_date = results.index.max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Strategy heatmap selector
    strategy_for_heatmap = st.sidebar.selectbox(
        "Heatmap Strategy",
        ["fetm", "buyhold", "linear", "binary"],
        format_func=lambda x: {"fetm": "FETM", "buyhold": "Buy & Hold",
                                "linear": "Linear TSMOM", "binary": "Binary TSMOM"}.get(x, x),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Period:** {run_data['metrics']['period']['start']} to "
        f"{run_data['metrics']['period']['end']}"
    )
    st.sidebar.markdown(
        f"**Trading Days:** {run_data['metrics']['period']['trading_days']:,}"
    )

    return {
        "period_option": period_option,
        "in_sample_end": in_sample_end,
        "date_range": date_range,
        "strategy_for_heatmap": strategy_for_heatmap,
    }
