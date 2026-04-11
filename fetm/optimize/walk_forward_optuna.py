"""Walk-forward optimization driven by Optuna TPE.

For each walk-forward window the optimizer:

1. Draws ``n_trials`` candidate parameter sets via an Optuna TPE sampler,
   scoring each on the *in-sample* slice of that window.
2. Takes the best in-sample trial and re-evaluates it on the window's
   out-of-sample slice to produce an honest OOS score.
3. Records the per-window best parameters plus both IS and OOS metrics.

The per-window results, the flat list of every trial, and a ``best_params``
snapshot are what the dashboard reads later.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fetm.optimize.objective import BAD_SCORE, evaluate_params, score_from_metrics
from fetm.optimize.search_space import SEARCH_SPACE, Param, suggest_params
from fetm.utils.nested import set_nested

logger = logging.getLogger(__name__)


@dataclass
class OptimizerResult:
    """Container for optimizer outputs."""

    per_window: pd.DataFrame = field(default_factory=pd.DataFrame)
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_params: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


class WalkForwardOptunaOptimizer:
    """Rolling walk-forward parameter search using Optuna.

    Args:
        base_config: Starting config dict (typically ``load_config()``).
        initial_window_years: Length of the first in-sample window.
        refit_freq_years: How often a new window starts (also the OOS length).
        n_trials: Optuna trials per window.
        lambda_dd: Drawdown penalty in the scalar objective.
        seed: TPE sampler seed for reproducibility.
        search_space: Parameter list to search over (defaults to SEARCH_SPACE).
    """

    def __init__(
        self,
        base_config: dict[str, Any],
        initial_window_years: int = 5,
        refit_freq_years: int = 1,
        n_trials: int = 50,
        lambda_dd: float = 0.5,
        seed: int = 42,
        search_space: tuple[Param, ...] = SEARCH_SPACE,
    ) -> None:
        self.base_config = base_config
        self.initial_window_years = initial_window_years
        self.refit_freq_years = refit_freq_years
        self.n_trials = n_trials
        self.lambda_dd = lambda_dd
        self.seed = seed
        self.search_space = search_space

    # ------------------------------------------------------------------ #

    def run(self, data: pd.DataFrame) -> OptimizerResult:
        """Run the full walk-forward optimization.

        ``data`` should be the raw OHLCV frame for the whole history — each
        window passes a progressively larger slice to
        :func:`fetm.optimize.objective.evaluate_params` and restricts metrics
        via ``eval_slice``.
        """
        # Lazy import so the rest of the codebase doesn't need optuna.
        import optuna
        from optuna.samplers import TPESampler

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        initial_days = self.initial_window_years * 252
        refit_days = self.refit_freq_years * 252

        n = len(data)
        if n < initial_days + refit_days:
            raise ValueError(
                f"Not enough data: have {n} rows, need at least "
                f"{initial_days + refit_days} for one window."
            )

        per_window_rows: list[dict[str, Any]] = []
        trial_rows: list[dict[str, Any]] = []
        window_idx = 0
        t0 = time.time()

        current_end = initial_days
        while current_end + refit_days <= n:
            train_end_idx = current_end
            test_end_idx = min(current_end + refit_days, n)

            train_slice = data.iloc[:train_end_idx]
            full_slice = data.iloc[:test_end_idx]

            is_start = train_slice.index[0]
            is_end = train_slice.index[-1]
            oos_start = data.index[train_end_idx]
            oos_end = data.index[test_end_idx - 1]

            logger.info(
                "Window %d: IS=[%s..%s] OOS=[%s..%s] trials=%d",
                window_idx, is_start.date(), is_end.date(),
                oos_start.date(), oos_end.date(), self.n_trials,
            )

            sampler = TPESampler(seed=self.seed + window_idx)
            study = optuna.create_study(direction="maximize", sampler=sampler)

            def _objective(trial: "optuna.Trial") -> float:
                params = suggest_params(trial, self.search_space)
                score, metrics = evaluate_params(
                    self.base_config,
                    train_slice,
                    params,
                    lambda_dd=self.lambda_dd,
                    eval_slice=(is_start, is_end),
                )
                trial_rows.append({
                    "window": window_idx,
                    "trial": trial.number,
                    "score": score,
                    "sharpe": metrics.get("sharpe_ratio"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "ann_return": metrics.get("annualized_return"),
                    **{f"param.{k}": v for k, v in params.items()},
                })
                return score

            study.optimize(_objective, n_trials=self.n_trials, show_progress_bar=False)

            best_params = dict(study.best_params)
            is_score = study.best_value

            # Re-evaluate best params on the OOS slice of this window.
            oos_score, oos_metrics = evaluate_params(
                self.base_config,
                full_slice,
                best_params,
                lambda_dd=self.lambda_dd,
                eval_slice=(oos_start, oos_end),
            )

            row: dict[str, Any] = {
                "window": window_idx,
                "is_start": is_start,
                "is_end": is_end,
                "oos_start": oos_start,
                "oos_end": oos_end,
                "is_score": is_score,
                "oos_score": oos_score,
                "oos_sharpe": oos_metrics.get("sharpe_ratio"),
                "oos_max_drawdown": oos_metrics.get("max_drawdown"),
                "oos_calmar": oos_metrics.get("calmar_ratio"),
                "oos_ann_return": oos_metrics.get("annualized_return"),
                "n_trials": self.n_trials,
            }
            for k, v in best_params.items():
                row[f"param.{k}"] = v
            per_window_rows.append(row)

            logger.info(
                "Window %d best: IS=%.3f OOS=%.3f OOS_Sharpe=%.3f",
                window_idx, is_score, oos_score,
                oos_metrics.get("sharpe_ratio", float("nan")),
            )

            window_idx += 1
            current_end += refit_days

        elapsed = time.time() - t0
        per_window_df = pd.DataFrame(per_window_rows)
        trials_df = pd.DataFrame(trial_rows)

        # Build a config-shaped dict for the final window's best params so it
        # can be dropped into settings.yaml directly.
        final_best: dict[str, Any] = {}
        if len(per_window_df) > 0:
            last = per_window_rows[-1]
            final_best = {k.removeprefix("param."): v
                          for k, v in last.items() if k.startswith("param.")}

        valid_oos = per_window_df["oos_sharpe"].dropna() if len(per_window_df) else pd.Series(dtype=float)
        valid_dd = per_window_df["oos_max_drawdown"].dropna() if len(per_window_df) else pd.Series(dtype=float)
        summary = {
            "n_windows": int(len(per_window_df)),
            "n_trials_per_window": self.n_trials,
            "lambda_dd": self.lambda_dd,
            "avg_oos_sharpe": float(valid_oos.mean()) if len(valid_oos) else None,
            "median_oos_sharpe": float(valid_oos.median()) if len(valid_oos) else None,
            "avg_oos_max_drawdown": float(valid_dd.mean()) if len(valid_dd) else None,
            "elapsed_seconds": round(elapsed, 1),
        }

        return OptimizerResult(
            per_window=per_window_df,
            trials=trials_df,
            best_params=final_best,
            summary=summary,
        )

    # ------------------------------------------------------------------ #

    def apply_best(self, best_params: dict[str, Any]) -> dict[str, Any]:
        """Return a deep copy of the base config with ``best_params`` applied.

        Useful when you want to call :class:`BacktestEngine` directly with the
        optimizer's recommendation.
        """
        cfg = copy.deepcopy(self.base_config)
        for path, value in best_params.items():
            set_nested(cfg, path, value)
        return cfg
