"""Tests for backtest engine."""

import numpy as np
import pandas as pd
import pytest

from fetm.portfolio.construction import PositionSizer
from fetm.portfolio.costs import TransactionCostModel
from fetm.portfolio.rebalance import RebalanceScheduler
from fetm.backtest.metrics import PerformanceMetrics


class TestPositionSizer:
    def test_basic_sizing(self):
        idx = pd.date_range("2020-01-01", periods=5)
        signal = pd.Series([1.0, 0.5, -1.0, 0.0, 2.0], index=idx)
        vol = pd.Series([0.15, 0.15, 0.15, 0.15, 0.15], index=idx)

        sizer = PositionSizer(target_vol=0.15, max_leverage=3.0)
        pos = sizer.size(signal, vol)

        # pos = signal * (0.15 / 0.15) = signal
        np.testing.assert_allclose(pos.values, [1.0, 0.5, -1.0, 0.0, 2.0])

    def test_leverage_cap(self):
        idx = pd.date_range("2020-01-01", periods=3)
        signal = pd.Series([10.0, -10.0, 0.5], index=idx)
        vol = pd.Series([0.15, 0.15, 0.15], index=idx)

        sizer = PositionSizer(target_vol=0.15, max_leverage=3.0)
        pos = sizer.size(signal, vol)

        assert pos.iloc[0] == 3.0
        assert pos.iloc[1] == -3.0
        assert pos.iloc[2] == 0.5

    def test_vol_scaling(self):
        idx = pd.date_range("2020-01-01", periods=2)
        signal = pd.Series([1.0, 1.0], index=idx)
        vol = pd.Series([0.30, 0.15], index=idx)  # double vol -> half position

        sizer = PositionSizer(target_vol=0.15)
        pos = sizer.size(signal, vol)

        assert pos.iloc[0] == pytest.approx(0.5)
        assert pos.iloc[1] == pytest.approx(1.0)


class TestTransactionCosts:
    def test_cost_on_position_change(self):
        idx = pd.date_range("2020-01-01", periods=4)
        positions = pd.Series([0.0, 1.0, 1.0, 0.5], index=idx)

        model = TransactionCostModel(cost_bps=0.001)  # 10 bps
        costs = model.compute(positions)

        assert costs.iloc[0] == 0.0  # initial position 0
        assert costs.iloc[1] == pytest.approx(0.001)  # |1.0 - 0.0| * 0.001
        assert costs.iloc[2] == pytest.approx(0.0)    # no change
        assert costs.iloc[3] == pytest.approx(0.0005)  # |0.5 - 1.0| * 0.001


class TestRebalanceScheduler:
    def test_weekly_schedule(self):
        n = 25
        idx = pd.bdate_range("2020-01-01", periods=n)
        vol = pd.Series(0.15, index=idx)
        signal = pd.Series(0.5, index=idx)

        sched = RebalanceScheduler(frequency=5, vol_threshold=0.5, signal_threshold=10)
        mask = sched.get_rebalance_mask(vol, signal)

        # Should rebalance every 5 days
        rebal_days = mask[mask].index
        assert len(rebal_days) >= 4

    def test_forced_rebalance_on_vol_spike(self):
        n = 20
        idx = pd.bdate_range("2020-01-01", periods=n)
        vol = pd.Series(0.15, index=idx)
        vol.iloc[10] = 0.30  # 100% increase -> should force rebalance
        signal = pd.Series(0.5, index=idx)

        sched = RebalanceScheduler(frequency=100, vol_threshold=0.20)
        mask = sched.get_rebalance_mask(vol, signal)

        assert mask.iloc[10]  # forced rebalance on vol spike day


class TestPerformanceMetrics:
    def test_basic_metrics(self):
        rng = np.random.default_rng(123)
        n = 2000
        dates = pd.bdate_range("2020-01-01", periods=n)
        # Strong positive drift strategy
        returns = pd.Series(rng.normal(0.001, 0.01, n), index=dates)

        pm = PerformanceMetrics(risk_free_rate=0.02)
        m = pm.compute(returns)

        assert m["annualized_vol"] > 0
        assert m["max_drawdown"] < 0
        assert 0 < m["win_rate_daily"] < 1
        # All expected keys present
        assert "sharpe_ratio" in m
        assert "sortino_ratio" in m
        assert "profit_factor" in m

    def test_buy_and_hold_sharpe(self):
        # Known market-like returns: ~10% return, ~16% vol, long series
        rng = np.random.default_rng(99)
        n = 10000
        dates = pd.bdate_range("2000-01-01", periods=n)
        returns = pd.Series(rng.normal(0.10 / 252, 0.16 / np.sqrt(252), n), index=dates)

        pm = PerformanceMetrics(risk_free_rate=0.02)
        m = pm.compute(returns)

        # With 10000 samples the Sharpe should converge closer to theoretical
        assert 0.0 < m["sharpe_ratio"] < 1.0
        assert 0.10 < m["annualized_vol"] < 0.25
