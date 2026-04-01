"""Page 2: Strategy Comparison — Deep side-by-side analysis."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fetm.dashboard.components.charts import (
    annual_returns_bar,
    return_distribution,
    qq_plot,
)
from fetm.dashboard.components.tables import (
    metrics_table,
    crisis_table,
    conditional_table,
)


def render_strategies(
    results: pd.DataFrame,
    metrics: dict,
    config: dict,
) -> None:
    """Render the strategy comparison page.

    Args:
        results: Filtered results DataFrame.
        metrics: Metrics dict from metrics.json.
        config: Config dict.
    """
    st.title("Strategy Comparison")

    # Row 1: Full Metrics Table
    st.markdown("### Performance Metrics (Out-of-Sample)")
    strat_metrics = metrics.get("strategies", {})
    if strat_metrics:
        df = metrics_table(strat_metrics)
        st.dataframe(df, use_container_width=True, height=650)

    st.markdown("---")

    # Row 2: Annual Returns Bar Chart
    st.markdown("### Annual Returns")
    fig_annual = annual_returns_bar(results)
    st.plotly_chart(fig_annual, use_container_width=True)

    st.markdown("---")

    # Row 3: Return Distribution + QQ Plot
    st.markdown("### Return Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig_dist = return_distribution(results)
        fig_dist.update_layout(height=350)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        fig_qq = qq_plot(results, "fetm")
        fig_qq.update_layout(height=350)
        st.plotly_chart(fig_qq, use_container_width=True)

    st.markdown("---")

    # Row 4: Crisis Performance Table
    st.markdown("### Crisis Performance")
    crisis_perf = metrics.get("crisis_performance", {})
    crisis_periods = config.get("crisis_periods", {})
    if crisis_perf:
        df_crisis = crisis_table(crisis_perf, crisis_periods)
        st.dataframe(
            df_crisis.style.map(
                lambda v: "color: #10B981" if isinstance(v, str) and not v.startswith("-") and v != "N/A"
                else "color: #EF4444" if isinstance(v, str) and v.startswith("-")
                else ""
            ),
            use_container_width=True,
        )

    st.markdown("---")

    # Row 5: Conditional Performance
    st.markdown("### Conditional Performance (by Benchmark Return Tercile)")
    cond_perf = metrics.get("conditional_performance", {})
    if cond_perf:
        df_cond = conditional_table(cond_perf)
        st.dataframe(df_cond, use_container_width=True)
        st.caption("Sharpe ratio of each strategy conditional on benchmark return regime.")
