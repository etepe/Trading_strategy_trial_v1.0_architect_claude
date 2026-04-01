"""Cached data loading for dashboard."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300)
def load_run(run_dir: str) -> dict:
    """Load all backtest outputs into memory.

    Args:
        run_dir: Path to the run output directory.

    Returns:
        Dict with keys: results, metrics, config, diagnostics.
    """
    run_path = Path(run_dir)

    results = pd.read_parquet(run_path / "results.parquet")

    with open(run_path / "metrics.json") as f:
        metrics = json.load(f)

    config_path = run_path / "config_snapshot.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    diagnostics = {}
    diag_dir = run_path / "diagnostics"
    if diag_dir.exists():
        for f in diag_dir.glob("*.parquet"):
            diagnostics[f.stem] = pd.read_parquet(f)

    return {
        "results": results,
        "metrics": metrics,
        "config": config,
        "diagnostics": diagnostics,
    }


def filter_by_period(
    results: pd.DataFrame,
    period_option: str,
    in_sample_end: str,
) -> pd.DataFrame:
    """Filter results by in-sample/out-of-sample period.

    Args:
        results: Full results DataFrame.
        period_option: One of 'Full Period', 'In-Sample Only', 'Out-of-Sample Only'.
        in_sample_end: End date of in-sample period.

    Returns:
        Filtered DataFrame.
    """
    if period_option == "In-Sample Only":
        return results[results.index <= pd.Timestamp(in_sample_end)]
    elif period_option == "Out-of-Sample Only":
        return results[results.index > pd.Timestamp(in_sample_end)]
    return results
