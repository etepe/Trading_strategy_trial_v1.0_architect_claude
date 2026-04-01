"""Page 1: Overview — Executive summary dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fetm.dashboard.components.kpi_cards import strategy_kpi_row
from fetm.dashboard.components.charts import (
    equity_curve_chart,
    drawdown_chart,
    rolling_sharpe_chart,
    monthly_heatmap,
)


def render_overview(
    results: pd.DataFrame,
    metrics: dict,
    config: dict,
    filters: dict,
) -> None:
    """Render the overview page.

    Args:
        results: Filtered results DataFrame.
        metrics: Metrics dict from metrics.json.
        config: Config dict.
        filters: Sidebar filter state.
    """
    st.title("FETM Strategy Overview")

    # Row 1: KPI Cards
    st.markdown("### Performance Summary")
    strategy_kpi_row(metrics.get("strategies", {}))

    st.markdown("---")

    # Row 2: Equity Curves
    crisis_periods = config.get("crisis_periods", {})
    in_sample_end = metrics.get("period", {}).get("in_sample_end")

    fig_equity = equity_curve_chart(results, crisis_periods, in_sample_end)
    st.plotly_chart(fig_equity, use_container_width=True)

    # Row 3: Two columns — Drawdown + Rolling Sharpe
    col1, col2 = st.columns(2)

    with col1:
        fig_dd = drawdown_chart(results)
        fig_dd.update_layout(height=350)
        st.plotly_chart(fig_dd, use_container_width=True)

    with col2:
        fig_rs = rolling_sharpe_chart(results)
        fig_rs.update_layout(height=350)
        st.plotly_chart(fig_rs, use_container_width=True)

    # Row 4: Monthly Returns Heatmap
    st.markdown("---")
    strategy = filters.get("strategy_for_heatmap", "fetm")
    fig_heatmap = monthly_heatmap(results, strategy)
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)
