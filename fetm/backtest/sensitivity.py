"""Parameter sensitivity analysis."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd

from fetm.backtest.engine import BacktestEngine
from fetm.backtest.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class SensitivityAnalysis:
    """Analyze strategy sensitivity to parameter changes.

    Args:
        config: Base configuration dictionary.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            from fetm.config import load_config
            config = load_config()
        self.base_config = config

    def run_corridor_sensitivity(
        self,
        data: pd.DataFrame,
        delta_values: list[float] | None = None,
    ) -> dict:
        """Run backtest for each corridor width value.

        Args:
            data: Raw OHLCV data.
            delta_values: Corridor widths to test.

        Returns:
            Dict with delta_values and sharpe_ratios lists.
        """
        if delta_values is None:
            delta_values = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03]

        sharpes = []
        turnovers = []

        for delta in delta_values:
            config = copy.deepcopy(self.base_config)
            config["volatility"]["exit_time"]["corridor_width"] = delta

            engine = BacktestEngine(config)
            results = engine.run(data)

            # OOS metrics
            in_sample_end = pd.Timestamp(config["backtest"]["in_sample_end"])
            oos_mask = results.index > in_sample_end
            oos_returns = results.loc[oos_mask, "return_fetm"].dropna()

            pm = PerformanceMetrics(risk_free_rate=config["backtest"]["risk_free_rate"])
            m = pm.compute(oos_returns, results.loc[oos_mask, "position_fetm"])
            sharpes.append(m["sharpe_ratio"])
            turnovers.append(m["annual_turnover"])

            logger.info("Delta=%.4f: Sharpe=%.3f, Turnover=%.1f", delta, m["sharpe_ratio"], m["annual_turnover"])

        return {
            "delta_values": delta_values,
            "sharpe_ratios": sharpes,
            "annual_turnovers": turnovers,
        }

    def run_2d_grid(
        self,
        data: pd.DataFrame,
        param1_name: str,
        param1_values: list[float],
        param2_name: str,
        param2_values: list[float],
    ) -> pd.DataFrame:
        """Run 2D parameter grid sensitivity.

        Args:
            data: Raw OHLCV data.
            param1_name: First parameter path (dot-separated, e.g. 'volatility.exit_time.corridor_width').
            param1_values: Values for first parameter.
            param2_name: Second parameter path.
            param2_values: Values for second parameter.

        Returns:
            DataFrame with columns: param1, param2, sharpe_ratio.
        """
        results_list = []

        for v1 in param1_values:
            for v2 in param2_values:
                config = copy.deepcopy(self.base_config)
                self._set_nested(config, param1_name, v1)
                self._set_nested(config, param2_name, v2)

                engine = BacktestEngine(config)
                res = engine.run(data)

                in_sample_end = pd.Timestamp(config["backtest"]["in_sample_end"])
                oos_mask = res.index > in_sample_end
                oos_returns = res.loc[oos_mask, "return_fetm"].dropna()

                pm = PerformanceMetrics(risk_free_rate=config["backtest"]["risk_free_rate"])
                m = pm.compute(oos_returns)

                results_list.append({
                    param1_name: v1,
                    param2_name: v2,
                    "sharpe_ratio": m["sharpe_ratio"],
                })

        return pd.DataFrame(results_list)

    @staticmethod
    def _set_nested(d: dict, key_path: str, value: Any) -> None:
        """Set a value in a nested dict using dot-separated path."""
        keys = key_path.split(".")
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
