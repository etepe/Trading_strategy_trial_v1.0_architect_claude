"""Walk-forward validation framework."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from fetm.backtest.engine import BacktestEngine
from fetm.backtest.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Walk-forward validation: optimize parameters in-sample, evaluate out-of-sample.

    Args:
        initial_window_years: Size of initial training window.
        refit_freq_years: How often to refit parameters.
        config: Base configuration dictionary.
    """

    def __init__(
        self,
        initial_window_years: int = 5,
        refit_freq_years: int = 1,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.initial_window_years = initial_window_years
        self.refit_freq_years = refit_freq_years
        self.config = config

    def run(
        self,
        data: pd.DataFrame,
        delta_grid: list[float] | None = None,
    ) -> pd.DataFrame:
        """Run walk-forward validation over corridor width parameter.

        Args:
            data: Raw OHLCV data.
            delta_grid: Corridor width values to test.

        Returns:
            DataFrame with columns: refit_date, optimal_delta,
            sharpe_in_sample, sharpe_out_of_sample.
        """
        if delta_grid is None:
            delta_grid = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03]

        initial_days = self.initial_window_years * 252
        refit_days = self.refit_freq_years * 252

        results = []
        n = len(data)

        current_start = initial_days
        while current_start + refit_days <= n:
            train_end = current_start
            test_end = min(current_start + refit_days, n)

            train_data = data.iloc[:train_end]
            test_data = data.iloc[:test_end]

            # Find optimal delta on in-sample
            best_delta = delta_grid[0]
            best_sharpe_is = -np.inf

            for delta in delta_grid:
                config = self._make_config(delta)
                engine = BacktestEngine(config)
                res = engine.run(train_data)

                # Compute IS sharpe on the training portion
                ret = res["return_fetm"].dropna()
                if len(ret) > 60:
                    pm = PerformanceMetrics(risk_free_rate=config["backtest"]["risk_free_rate"])
                    m = pm.compute(ret)
                    sr = m["sharpe_ratio"]
                    if sr > best_sharpe_is:
                        best_sharpe_is = sr
                        best_delta = delta

            # Run with optimal delta on full data up to test_end
            config_opt = self._make_config(best_delta)
            engine_opt = BacktestEngine(config_opt)
            res_full = engine_opt.run(test_data)

            # Compute OOS sharpe on the test portion only
            oos_ret = res_full["return_fetm"].iloc[train_end:test_end].dropna()
            pm = PerformanceMetrics(risk_free_rate=config_opt["backtest"]["risk_free_rate"])
            if len(oos_ret) > 10:
                m_oos = pm.compute(oos_ret)
                sharpe_oos = m_oos["sharpe_ratio"]
            else:
                sharpe_oos = np.nan

            refit_date = data.index[train_end] if train_end < len(data.index) else None
            results.append({
                "refit_date": refit_date,
                "optimal_delta": best_delta,
                "sharpe_in_sample": best_sharpe_is,
                "sharpe_out_of_sample": sharpe_oos,
            })

            logger.info(
                "Walk-forward: refit=%s, delta=%.4f, IS_SR=%.3f, OOS_SR=%.3f",
                refit_date, best_delta, best_sharpe_is, sharpe_oos,
            )

            current_start += refit_days

        return pd.DataFrame(results)

    def _make_config(self, delta: float) -> dict:
        """Create config with specified corridor width."""
        import copy
        config = copy.deepcopy(self.config) if self.config else {}
        if not config:
            from fetm.config import load_config
            config = copy.deepcopy(load_config())
        config["volatility"]["exit_time"]["corridor_width"] = delta
        return config
