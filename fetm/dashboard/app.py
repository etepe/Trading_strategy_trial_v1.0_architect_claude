"""FETM Dashboard — Interactive Backtest Viewer.

Launch:
    python -m fetm.dashboard --run-dir output/runs/latest
    streamlit run fetm/dashboard/app.py -- --run-dir output/runs/latest
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="FETM Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from fetm.dashboard.utils.data_loader import load_run, filter_by_period
from fetm.dashboard.components.filters import render_sidebar


def _resolve_latest(runs_dir: Path) -> str | None:
    """Find the latest run directory, handling both symlinks and Windows fallback."""
    # 1. Try symlink (Unix)
    latest = runs_dir / "latest"
    if latest.exists():
        return str(latest.resolve())

    # 2. Try latest.txt pointer (Windows fallback)
    latest_txt = runs_dir / "latest.txt"
    if latest_txt.is_file():
        target = latest_txt.read_text().strip()
        if Path(target).exists():
            return target

    # 3. Pick the most recent timestamped directory
    if runs_dir.is_dir():
        subdirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
            key=lambda d: d.name,
            reverse=True,
        )
        if subdirs:
            return str(subdirs[0].resolve())

    return None


def get_run_dir() -> str:
    """Parse run directory from CLI args or use default."""
    # Check for --run-dir in sys.argv (after --)
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--run-dir" and i + 1 < len(args):
            given = args[i + 1]
            # If "latest" was given, resolve it
            p = Path(given)
            if p.name == "latest" and not p.exists():
                resolved = _resolve_latest(p.parent)
                if resolved:
                    return resolved
            return given

    # Default: try to find latest run
    resolved = _resolve_latest(Path("output/runs"))
    if resolved:
        return resolved

    st.error("No run directory found. Run `python -m fetm.backtest` first.")
    st.stop()
    return ""


def main() -> None:
    """Main dashboard entry point."""
    run_dir = get_run_dir()

    try:
        run_data = load_run(run_dir)
    except Exception as e:
        st.error(f"Failed to load run data from {run_dir}: {e}")
        st.stop()
        return

    # Store in session state for pages
    st.session_state["run_data"] = run_data

    # Render sidebar and get filter state
    filters = render_sidebar(run_data)
    st.session_state["filters"] = filters

    # Page navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Strategy Comparison"],
        index=0,
    )

    # Apply period filter
    filtered = filter_by_period(
        run_data["results"],
        filters["period_option"],
        filters["in_sample_end"],
    )
    st.session_state["filtered_results"] = filtered

    # Render selected page
    if page == "Overview":
        from fetm.dashboard.pages.page_01_overview import render_overview
        render_overview(filtered, run_data["metrics"], run_data.get("config", {}), filters)
    elif page == "Strategy Comparison":
        from fetm.dashboard.pages.page_02_strategies import render_strategies
        render_strategies(filtered, run_data["metrics"], run_data.get("config", {}))


if __name__ == "__main__":
    main()
