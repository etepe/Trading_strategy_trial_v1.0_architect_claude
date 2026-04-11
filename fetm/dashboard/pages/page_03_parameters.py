"""Page 3: Strategy Parameters — view current config & optimization runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from fetm.dashboard.components.charts import parameter_drift_chart
from fetm.dashboard.components.kpi_cards import kpi_card
from fetm.dashboard.components.param_descriptions import PARAM_GROUPS
from fetm.dashboard.utils.data_loader import (
    list_optimization_runs,
    load_optimization,
)
from fetm.utils.nested import get_nested


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_MISSING = object()


def _format_value(v: Any) -> str:
    """Render a config value for a table cell."""
    if v is None or v is _MISSING:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        # Short-scale numbers (e.g. 0.012, 0.394) render cleanly with 4 sig figs.
        if abs(v) >= 1000 or (abs(v) < 0.001 and v != 0):
            return f"{v:.3e}"
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_format_value(x) for x in v)
    return str(v)


def _render_current_params(config: dict) -> None:
    """Render the grouped read-only view of the currently loaded config."""
    st.markdown("### Current Parameters")
    st.caption(
        "These are the values from `config_snapshot.yaml` for the run you're "
        "viewing. To change them, edit `config/settings.yaml` and rerun the "
        "backtest."
    )

    for section_title, params in PARAM_GROUPS:
        with st.expander(section_title, expanded=True):
            rows = []
            for dot_path, label, help_text in params:
                value = get_nested(config, dot_path, default=_MISSING)
                rows.append({
                    "Parameter": label,
                    "Value": _format_value(value),
                    "Path": dot_path,
                    "Description": help_text,
                })
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Parameter": st.column_config.TextColumn(width="medium"),
                    "Value": st.column_config.TextColumn(width="small"),
                    "Path": st.column_config.TextColumn(width="medium"),
                    "Description": st.column_config.TextColumn(width="large"),
                },
            )


def _render_optimization_section(current_config: dict) -> None:
    """Render the optimization results section with run selector."""
    st.markdown("---")
    st.markdown("### Walk-Forward Optimization Results")

    runs = list_optimization_runs()
    if not runs:
        st.info(
            "No optimization runs found under `output/optimization/`. "
            "Run the optimizer to populate this section:\n\n"
            "```\npython -m fetm.optimize --budget quick --lambda-dd 0.5\n```"
        )
        return

    default_label = "(none)"
    labels = [default_label] + [Path(r).name for r in runs]
    selected_label = st.selectbox(
        "Select optimization run",
        labels,
        index=1 if len(labels) > 1 else 0,
        help="Scans output/optimization/ for timestamped run directories.",
    )
    if selected_label == default_label:
        st.caption("Select a run above to view results.")
        return

    run_idx = labels.index(selected_label) - 1
    opt_dir = runs[run_idx]

    try:
        opt = load_optimization(opt_dir)
    except Exception as exc:
        st.error(f"Failed to load {opt_dir}: {exc}")
        return

    per_window: pd.DataFrame = opt["per_window"]
    summary: dict = opt["summary"]
    best_params: dict = opt["best_params"]
    run_config: dict = opt["run_config"]

    if per_window.empty:
        st.warning(f"Selected run at `{opt_dir}` has no walk-forward results.")
        return

    # --- KPI row ---------------------------------------------------------- #
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card(
            label="Avg OOS Sharpe",
            value=summary.get("avg_oos_sharpe") or 0.0,
            format_str="{:.2f}",
            color="#10B981",
        )
    with col2:
        dd = summary.get("avg_oos_max_drawdown") or 0.0
        kpi_card(
            label="Avg OOS Max DD",
            value=dd,
            format_str="{:.1%}",
            color="#EF4444",
        )
    with col3:
        kpi_card(
            label="# Windows",
            value=summary.get("n_windows") or 0,
            format_str="{:.0f}",
            color="#3B82F6",
        )
    with col4:
        kpi_card(
            label="Trials / window",
            value=summary.get("n_trials_per_window") or 0,
            format_str="{:.0f}",
            color="#8B5CF6",
        )

    st.caption(
        f"Objective: sharpe − λ·|max_drawdown|  (λ = "
        f"{run_config.get('lambda_dd', '?')})  ·  "
        f"Elapsed: {summary.get('elapsed_seconds', '?')}s"
    )

    # --- Parameter drift plot -------------------------------------------- #
    st.markdown("#### Optimal parameter evolution")
    param_cols = [c for c in per_window.columns if c.startswith("param.")]
    if param_cols:
        default_selection = [c for c in param_cols if any(
            k in c for k in (
                "corridor_width", "target_vol", "scurve_lambda",
                "rebalance_frequency", "max_leverage",
            )
        )] or param_cols[:4]
        chosen = st.multiselect(
            "Parameters to plot",
            options=param_cols,
            default=default_selection,
            format_func=lambda c: c.removeprefix("param."),
        )
        if chosen:
            fig = parameter_drift_chart(per_window, chosen)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    # --- Per-window table ------------------------------------------------- #
    st.markdown("#### Per-window results")
    display_cols = [
        c for c in (
            "window", "oos_start", "oos_end",
            "is_score", "oos_score",
            "oos_sharpe", "oos_max_drawdown", "oos_calmar", "oos_ann_return",
        ) if c in per_window.columns
    ]
    st.dataframe(
        per_window[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    # --- Baseline vs optimized ------------------------------------------- #
    st.markdown("#### Baseline vs Optimized (final window)")
    st.caption(
        "Last window's best parameters compared against the currently loaded "
        "`config_snapshot.yaml`. Copy these into `config/settings.yaml` and "
        "rerun the backtest to apply them."
    )

    diff_rows = []
    for dot_path, value in best_params.items():
        baseline = get_nested(current_config, dot_path, default=_MISSING)
        diff_rows.append({
            "Parameter": dot_path,
            "Baseline": _format_value(baseline),
            "Optimized": _format_value(value),
            "Changed": (
                "—" if baseline is _MISSING
                else ("yes" if baseline != value else "no")
            ),
        })

    if diff_rows:
        diff_df = pd.DataFrame(diff_rows)
        st.dataframe(diff_df, use_container_width=True, hide_index=True)

    with st.expander("best_params.yaml (copy into settings.yaml)"):
        import yaml as _yaml

        def _nest(flat: dict) -> dict:
            out: dict = {}
            for path, value in flat.items():
                keys = path.split(".")
                cursor = out
                for k in keys[:-1]:
                    cursor = cursor.setdefault(k, {})
                cursor[keys[-1]] = value
            return out

        st.code(_yaml.dump(_nest(best_params), default_flow_style=False,
                           sort_keys=False), language="yaml")


# --------------------------------------------------------------------------- #
# Page entry point
# --------------------------------------------------------------------------- #

def render_parameters(
    run_data: dict,
    filters: dict | None = None,  # noqa: ARG001 — unused, kept for parity with other pages
) -> None:
    """Render the Strategy Parameters page."""
    st.title("Strategy Parameters")

    config = run_data.get("config", {}) or {}
    if not config:
        st.warning(
            "No `config_snapshot.yaml` found for this run. Rerun the backtest "
            "to regenerate it."
        )
        return

    _render_current_params(config)
    _render_optimization_section(config)
