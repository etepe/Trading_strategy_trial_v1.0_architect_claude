"""Cached data loading for dashboard."""

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


@st.cache_data(ttl=300)
def load_optimization(opt_dir: str) -> dict:
    """Load walk-forward optimization artifacts from ``opt_dir``.

    Expects the directory produced by ``python -m fetm.optimize``:

        walk_forward_results.parquet
        trials.parquet
        best_params.yaml
        run_config.yaml
        summary.json

    Missing files are tolerated and represented as empty frames / dicts so
    the dashboard can render a partial view.
    """
    opt_path = Path(opt_dir)

    per_window = pd.DataFrame()
    wf_path = opt_path / "walk_forward_results.parquet"
    if wf_path.exists():
        per_window = pd.read_parquet(wf_path)

    trials = pd.DataFrame()
    trials_path = opt_path / "trials.parquet"
    if trials_path.exists():
        trials = pd.read_parquet(trials_path)

    best_params: dict = {}
    best_path = opt_path / "best_params.yaml"
    if best_path.exists():
        with open(best_path) as f:
            best_params = yaml.safe_load(f) or {}

    run_config: dict = {}
    rc_path = opt_path / "run_config.yaml"
    if rc_path.exists():
        with open(rc_path) as f:
            run_config = yaml.safe_load(f) or {}

    summary: dict = {}
    summary_path = opt_path / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    return {
        "per_window": per_window,
        "trials": trials,
        "best_params": best_params,
        "run_config": run_config,
        "summary": summary,
        "path": str(opt_path),
    }


def list_optimization_runs(root: str = "output/optimization") -> list[str]:
    """Return sorted list of optimization run directories (most recent first)."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    dirs = [p for p in root_path.iterdir() if p.is_dir() and p.name != "latest"]
    dirs.sort(key=lambda p: p.name, reverse=True)
    return [str(p) for p in dirs]


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
