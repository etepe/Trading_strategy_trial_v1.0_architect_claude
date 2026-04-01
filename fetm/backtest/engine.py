"""Core backtest engine — runs all four strategies in a single pass."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fetm.config import load_config
from fetm.data.clean import clean_ohlcv
from fetm.data.features import compute_features
from fetm.signals.volatility import (
    CompositeVolatility,
    EWMAVolatility,
    FETVolatility,
    ParkinsonVolatility,
)
from fetm.signals.momentum import BinaryMomentum, LinearMomentum, NonlinearMomentum
from fetm.portfolio.construction import PositionSizer
from fetm.portfolio.costs import TransactionCostModel
from fetm.portfolio.rebalance import RebalanceScheduler
from fetm.backtest.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Event-driven backtest engine for FETM strategy.

    Runs all four strategies (buy-and-hold, linear TSMOM, binary TSMOM, FETM)
    in a single pass over the data.

    Args:
        config: Configuration dictionary (from settings.yaml).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize all strategy components from config."""
        c = self.config

        # Volatility estimators
        self.ewma = EWMAVolatility(halflife=c["volatility"]["ewma"]["halflife"])
        self.parkinson = ParkinsonVolatility(window=c["volatility"]["parkinson"]["window"])
        self.fet = FETVolatility(
            corridor_width=c["volatility"]["exit_time"]["corridor_width"],
            buffer_size=c["volatility"]["exit_time"]["buffer_size"],
            max_no_exit_days=c["volatility"]["exit_time"]["max_no_exit_days"],
            bias_correction=c["volatility"]["exit_time"]["bias_correction"],
        )
        self.composite = CompositeVolatility(
            sigmoid_k=c["volatility"]["composite"]["blend_steepness"],
            max_weight=c["volatility"]["composite"]["max_fet_weight"],
        )

        # Signal generators
        lookbacks = c["momentum"]["lookback_days"]
        weights = c["momentum"]["horizon_weights"]
        norm_window = c["momentum"]["signal_normalization_window"]
        signal_cap = c["momentum"]["signal_cap"]
        scurve_lam = c["momentum"]["scurve_lambda"]

        self.linear_mom = LinearMomentum(lookbacks, weights, norm_window, signal_cap)
        self.nonlinear_mom = NonlinearMomentum(lookbacks, weights, scurve_lam, norm_window, signal_cap)
        self.binary_mom = BinaryMomentum(lookbacks, weights, norm_window)

        # Portfolio components
        target_vol = c["strategy"]["target_vol"]
        max_leverage = c["strategy"]["max_leverage"]
        self.sizer = PositionSizer(target_vol, max_leverage)
        self.cost_model = TransactionCostModel(cost_bps=c["costs"]["equity_etf_bps"] / 10000)
        self.rebalancer = RebalanceScheduler(
            frequency=c["strategy"]["rebalance_frequency"],
            vol_threshold=c["strategy"]["rebalance_vol_threshold"],
            signal_threshold=c["strategy"]["rebalance_signal_threshold"],
        )

        # Metrics
        self.metrics = PerformanceMetrics(risk_free_rate=c["backtest"]["risk_free_rate"])

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the full backtest on OHLCV data.

        Args:
            data: Raw or clean OHLCV DataFrame.

        Returns:
            Results DataFrame with all columns from the spec.
        """
        logger.info("Starting backtest engine...")

        # Clean and add features
        df = clean_ohlcv(data)
        df = compute_features(df)

        warmup = self.config["backtest"]["warmup_days"]
        logger.info("Data: %d rows, warmup: %d days", len(df), warmup)

        # === Volatility Estimation ===
        logger.info("Computing volatility estimates...")
        vol_ewma = self.ewma.estimate_series(df["log_return"])
        vol_parkinson = self.parkinson.estimate_series(df["high"], df["low"])
        vol_fet, fet_diag = self.fet.estimate_series(
            df["open"], df["high"], df["low"], df["close"], vol_ewma
        )
        vol_composite, fet_weight = self.composite.blend(vol_fet, vol_ewma)

        # === Signal Computation ===
        logger.info("Computing momentum signals...")
        linear_signals = self.linear_mom.compute(df["close"], vol_ewma)
        nonlinear_signals = self.nonlinear_mom.compute(df["close"], vol_composite)
        binary_signals = self.binary_mom.compute(df["close"], vol_ewma)

        # === Position Sizing ===
        logger.info("Computing positions...")

        # Strategy 1: Buy-and-hold (always position = 1.0)
        pos_buyhold = pd.Series(1.0, index=df.index, name="position_buyhold")

        # Strategy 2: Linear TSMOM (linear signal + EWMA vol)
        pos_linear_raw = self.sizer.size(linear_signals["signal_combined"], vol_ewma)
        rebal_mask_linear = self.rebalancer.get_rebalance_mask(vol_ewma, linear_signals["signal_combined"])
        pos_linear = self.rebalancer.apply_rebalance(pos_linear_raw, rebal_mask_linear)
        pos_linear.name = "position_linear"

        # Strategy 3: Binary TSMOM (binary signal + EWMA vol)
        pos_binary_raw = self.sizer.size(binary_signals["signal_combined"], vol_ewma)
        rebal_mask_binary = self.rebalancer.get_rebalance_mask(vol_ewma, binary_signals["signal_combined"])
        pos_binary = self.rebalancer.apply_rebalance(pos_binary_raw, rebal_mask_binary)
        pos_binary.name = "position_binary"

        # Strategy 4: FETM (nonlinear signal + composite vol)
        pos_fetm_raw = self.sizer.size(nonlinear_signals["signal_combined"], vol_composite)
        rebal_mask_fetm = self.rebalancer.get_rebalance_mask(vol_composite, nonlinear_signals["signal_combined"])
        pos_fetm = self.rebalancer.apply_rebalance(pos_fetm_raw, rebal_mask_fetm)
        pos_fetm.name = "position_fetm"

        # Zero out positions during warmup
        warmup_mask = np.arange(len(df)) < warmup
        pos_linear[warmup_mask] = 0.0
        pos_binary[warmup_mask] = 0.0
        pos_fetm[warmup_mask] = 0.0

        # === Returns Computation (CRITICAL: use position.shift(1) for no look-ahead) ===
        logger.info("Computing returns...")
        log_ret = df["log_return"]

        # Buy-and-hold returns (no costs)
        ret_buyhold = log_ret.copy()
        ret_buyhold.name = "return_buyhold"

        # Linear TSMOM returns
        cost_linear = self.cost_model.compute(pos_linear)
        ret_linear = pos_linear.shift(1).fillna(0) * log_ret - cost_linear
        ret_linear.name = "return_linear"

        # Binary TSMOM returns
        cost_binary = self.cost_model.compute(pos_binary)
        ret_binary = pos_binary.shift(1).fillna(0) * log_ret - cost_binary
        ret_binary.name = "return_binary"

        # FETM returns
        cost_fetm = self.cost_model.compute(pos_fetm)
        ret_fetm = pos_fetm.shift(1).fillna(0) * log_ret - cost_fetm
        ret_fetm.name = "return_fetm"

        # Cumulative returns (simple compounding of log returns approximation)
        cum_buyhold = (1 + ret_buyhold.fillna(0)).cumprod()
        cum_linear = (1 + ret_linear.fillna(0)).cumprod()
        cum_binary = (1 + ret_binary.fillna(0)).cumprod()
        cum_fetm = (1 + ret_fetm.fillna(0)).cumprod()

        # === Vol Regime Classification ===
        expanding_median = vol_fet.expanding(min_periods=60).median()
        vol_ratio = vol_fet / expanding_median.clip(lower=1e-6)
        vol_change = vol_fet.diff() / vol_fet.shift(1).clip(lower=1e-6)

        regime = pd.Series("normal", index=df.index)
        regime[vol_ratio > 1.5] = "high_vol"
        regime[(vol_ratio > 2.0) & (vol_change > 0)] = "crisis"
        regime[vol_ratio < 0.7] = "low_vol"

        # === Rebalance day flags ===
        is_rebalance = rebal_mask_fetm & (~warmup_mask)

        # Turnover
        turnover_linear = pos_linear.diff().abs()
        turnover_binary = pos_binary.diff().abs()
        turnover_fetm = pos_fetm.diff().abs()

        # === Assemble Results DataFrame ===
        logger.info("Assembling results...")
        results = pd.DataFrame({
            # Price data
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df["volume"],
            "log_return": df["log_return"],

            # Volatility estimates
            "vol_ewma": vol_ewma,
            "vol_parkinson": vol_parkinson,
            "vol_fet": vol_fet,
            "vol_composite": vol_composite,
            "fet_weight": fet_weight,

            # Signals (nonlinear for FETM)
            "signal_raw_21d": nonlinear_signals.get("signal_raw_21d"),
            "signal_raw_63d": nonlinear_signals.get("signal_raw_63d"),
            "signal_raw_252d": nonlinear_signals.get("signal_raw_252d"),
            "signal_nl_21d": nonlinear_signals.get("signal_nl_21d"),
            "signal_nl_63d": nonlinear_signals.get("signal_nl_63d"),
            "signal_nl_252d": nonlinear_signals.get("signal_nl_252d"),
            "signal_combined": nonlinear_signals.get("signal_combined"),

            # Positions
            "position_buyhold": pos_buyhold,
            "position_linear": pos_linear,
            "position_binary": pos_binary,
            "position_fetm": pos_fetm,

            # Returns
            "return_buyhold": ret_buyhold,
            "return_linear": ret_linear,
            "return_binary": ret_binary,
            "return_fetm": ret_fetm,

            # Cumulative returns
            "cumreturn_buyhold": cum_buyhold,
            "cumreturn_linear": cum_linear,
            "cumreturn_binary": cum_binary,
            "cumreturn_fetm": cum_fetm,

            # Exit-time diagnostics
            "exit_time_last": fet_diag["exit_time_last"],
            "exit_times_mean": fet_diag["exit_times_mean"],
            "corridor_upper": fet_diag["corridor_upper"],
            "corridor_lower": fet_diag["corridor_lower"],
            "days_since_exit": fet_diag["days_since_exit"],

            # Regime
            "vol_regime": regime,

            # Meta
            "is_rebalance_day": is_rebalance,
            "turnover": turnover_fetm,
            "transaction_cost": cost_fetm,
        })
        results.index.name = "date"

        # Add asset column
        results["asset"] = self.config["data"]["ticker"]

        logger.info("Backtest complete. %d rows.", len(results))
        return results

    def save_results(
        self, results: pd.DataFrame, run_dir: Path | None = None
    ) -> Path:
        """Save backtest results to disk.

        Args:
            results: Results DataFrame from run().
            run_dir: Output directory. Auto-generated if None.

        Returns:
            Path to the run directory.
        """
        if run_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            runs_dir = Path(self.config["output"]["runs_dir"])
            run_dir = runs_dir / ts
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save results parquet
        results.to_parquet(run_dir / "results.parquet")
        logger.info("Saved results.parquet (%d rows)", len(results))

        # Save config snapshot
        with open(run_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # Compute and save metrics
        metrics = self._compute_all_metrics(results)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("Saved metrics.json")

        # Save diagnostics
        diag_dir = run_dir / "diagnostics"
        diag_dir.mkdir(exist_ok=True)

        vol_comp = results[["vol_ewma", "vol_parkinson", "vol_fet", "vol_composite", "fet_weight"]].copy()
        vol_comp.to_parquet(diag_dir / "vol_estimator_comparison.parquet")

        exit_diag = results[["exit_time_last", "exit_times_mean", "corridor_upper", "corridor_lower", "days_since_exit"]].copy()
        exit_diag.to_parquet(diag_dir / "exit_times_raw.parquet")

        signal_cols = [c for c in results.columns if c.startswith("signal_")]
        if signal_cols:
            results[signal_cols].to_parquet(diag_dir / "signal_components.parquet")

        # Update latest symlink
        latest = run_dir.parent / "latest"
        if latest.is_symlink():
            latest.unlink()
        elif latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)

        logger.info("Results saved to %s", run_dir)
        return run_dir

    def _compute_all_metrics(self, results: pd.DataFrame) -> dict:
        """Compute metrics for all strategies and save as JSON-compatible dict."""
        c = self.config
        in_sample_end = pd.Timestamp(c["backtest"]["in_sample_end"])

        strategies = {
            "buyhold": ("return_buyhold", "position_buyhold"),
            "linear": ("return_linear", "position_linear"),
            "binary": ("return_binary", "position_binary"),
            "fetm": ("return_fetm", "position_fetm"),
        }

        # Compute metrics for full, in-sample, and out-of-sample
        strat_metrics = {}
        for name, (ret_col, pos_col) in strategies.items():
            oos_mask = results.index > in_sample_end
            oos_returns = results.loc[oos_mask, ret_col].dropna()
            oos_positions = results.loc[oos_mask, pos_col]
            strat_metrics[name] = self.metrics.compute(oos_returns, oos_positions)

        # Crisis performance
        crisis_periods = c.get("crisis_periods", {})
        returns_dict = {
            name: results[ret_col]
            for name, (ret_col, _) in strategies.items()
        }
        crisis_perf = self.metrics.compute_crisis_performance(returns_dict, crisis_periods)

        # Conditional performance
        cond_perf = self.metrics.compute_conditional_performance(
            returns_dict, results["return_buyhold"]
        )

        # Corridor sensitivity placeholder
        corridor_sensitivity = {
            "delta_values": [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03],
            "sharpe_ratios": [],  # filled by sensitivity analysis
        }

        # Run info
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        period_start = results.index[0].strftime("%Y-%m-%d")
        period_end = results.index[-1].strftime("%Y-%m-%d")

        return {
            "run_id": run_id,
            "phase": 1,
            "config": c,
            "period": {
                "start": period_start,
                "end": period_end,
                "trading_days": len(results),
                "in_sample_end": str(in_sample_end.date()),
            },
            "strategies": strat_metrics,
            "crisis_performance": crisis_perf,
            "conditional_performance": cond_perf,
            "corridor_sensitivity": corridor_sensitivity,
        }
