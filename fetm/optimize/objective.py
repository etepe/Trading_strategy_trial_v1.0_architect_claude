"""Single-trial evaluator used by the walk-forward optimizer.

Given a base config, a slice of OHLCV data and a parameter override dict,
this runs the full backtest engine on that slice and returns a scalar
score:

    score = sharpe_ratio - lambda_dd * |max_drawdown|

Any numerical failure (empty returns, NaN Sharpe, engine exception) is
converted to a very negative sentinel so Optuna treats the trial as bad
without crashing the whole study.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any

import pandas as pd

from fetm.backtest.engine import BacktestEngine
from fetm.backtest.metrics import PerformanceMetrics
from fetm.utils.nested import set_nested

logger = logging.getLogger(__name__)

BAD_SCORE = -1e9


def score_from_metrics(metrics: dict, lambda_dd: float) -> float:
    """Combine sharpe and drawdown into a single maximization target."""
    sharpe = metrics.get("sharpe_ratio")
    max_dd = metrics.get("max_drawdown")
    if sharpe is None or max_dd is None:
        return BAD_SCORE
    if not math.isfinite(sharpe) or not math.isfinite(max_dd):
        return BAD_SCORE
    return float(sharpe) - float(lambda_dd) * abs(float(max_dd))


def evaluate_params(
    base_config: dict[str, Any],
    data: pd.DataFrame,
    trial_params: dict[str, Any],
    lambda_dd: float = 0.5,
    eval_slice: tuple[pd.Timestamp | None, pd.Timestamp | None] | None = None,
) -> tuple[float, dict]:
    """Run one backtest with ``trial_params`` applied and return (score, metrics).

    Args:
        base_config: Baseline config (deep-copied internally).
        data: Raw OHLCV dataframe; the engine handles cleaning and warmup.
        trial_params: Dict keyed by dot-path (e.g. ``"strategy.target_vol"``).
        lambda_dd: Drawdown penalty weight in the scalar objective.
        eval_slice: ``(start, end)`` timestamps restricting which rows of the
            engine result are used to compute metrics. ``None`` on either side
            means "open".

    Returns:
        (score, metrics_dict). ``metrics_dict`` is the full dict from
        :class:`PerformanceMetrics.compute` — useful for logging trials.
    """
    config = copy.deepcopy(base_config)
    for path, value in trial_params.items():
        set_nested(config, path, value)

    try:
        engine = BacktestEngine(config)
        results = engine.run(data)
    except Exception as exc:  # pragma: no cover — backtest is deterministic
        logger.warning("Trial backtest failed: %s", exc)
        return BAD_SCORE, {}

    returns = results["return_fetm"]
    positions = results["position_fetm"]

    if eval_slice is not None:
        start, end = eval_slice
        mask = pd.Series(True, index=results.index)
        if start is not None:
            mask &= results.index >= start
        if end is not None:
            mask &= results.index <= end
        returns = returns[mask]
        positions = positions[mask]

    returns = returns.dropna()
    if len(returns) < 60:
        return BAD_SCORE, {}

    pm = PerformanceMetrics(risk_free_rate=config["backtest"]["risk_free_rate"])
    metrics = pm.compute(returns, positions)
    return score_from_metrics(metrics, lambda_dd), metrics
