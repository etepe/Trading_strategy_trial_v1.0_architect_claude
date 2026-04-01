"""KPI card components for dashboard."""

import streamlit as st


def kpi_card(
    label: str,
    value: float,
    delta: float | None = None,
    format_str: str = "{:.2f}",
    color: str = "#10B981",
) -> None:
    """Render a styled KPI card using Streamlit markdown.

    Args:
        label: Card label/title.
        value: Primary metric value.
        delta: Delta vs benchmark (optional).
        format_str: Format string for the value.
        color: Accent color.
    """
    delta_html = ""
    if delta is not None:
        arrow = "&#9650;" if delta > 0 else "&#9660;"
        delta_color = "#10B981" if delta > 0 else "#EF4444"
        delta_html = (
            f'<span style="color:{delta_color};font-size:14px">'
            f'{arrow} {abs(delta):.2f}</span>'
        )

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}15, {color}05);
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 8px;
        ">
            <div style="color:#9CA3AF;font-size:13px;text-transform:uppercase;
                        letter-spacing:0.5px">{label}</div>
            <div style="font-size:32px;font-weight:700;color:#F3F4F6;
                        margin:4px 0">{format_str.format(value)}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def strategy_kpi_row(metrics: dict, benchmark_key: str = "buyhold") -> None:
    """Render a row of KPI cards for all strategies.

    Args:
        metrics: Dict from metrics.json['strategies'].
        benchmark_key: Key for the benchmark strategy.
    """
    from fetm.dashboard.utils.theme import STRATEGY_COLORS, STRATEGY_LABELS

    strategies = ["buyhold", "linear", "binary", "fetm"]
    cols = st.columns(4)

    bench_sharpe = metrics.get(benchmark_key, {}).get("sharpe_ratio", 0)

    for col, strat in zip(cols, strategies):
        m = metrics.get(strat, {})
        sharpe = m.get("sharpe_ratio", 0)
        ann_ret = m.get("annualized_return", 0)
        max_dd = m.get("max_drawdown", 0)
        delta = sharpe - bench_sharpe if strat != benchmark_key else None

        with col:
            kpi_card(
                label=f"{STRATEGY_LABELS.get(strat, strat)} | Sharpe",
                value=sharpe,
                delta=delta,
                format_str="{:.2f}",
                color=STRATEGY_COLORS.get(strat, "#6B7280"),
            )
            st.markdown(
                f"""<div style="padding:0 20px;color:#9CA3AF;font-size:12px">
                    Ret: {ann_ret:.1%} | MaxDD: {max_dd:.1%}
                </div>""",
                unsafe_allow_html=True,
            )
