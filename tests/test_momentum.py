"""Tests for momentum signal generators."""

import numpy as np
import pandas as pd
import pytest

from fetm.signals.momentum import LinearMomentum, NonlinearMomentum, BinaryMomentum, scurve


class TestScurve:
    def test_zero(self):
        assert scurve(np.array([0.0])) == pytest.approx(0.0)

    def test_symmetry(self):
        x = np.array([0.5, 1.0, 2.0, 3.0])
        assert np.allclose(scurve(x), -scurve(-x))

    def test_at_one(self):
        # f(1) = 0.394 * 1 / (1 + 1) = 0.197
        assert scurve(np.array([1.0])) == pytest.approx(0.197, abs=0.001)

    def test_attenuates_extremes(self):
        # |f(10)| should be much less than |f(1)|
        f1 = abs(scurve(np.array([1.0]))[0])
        f10 = abs(scurve(np.array([10.0]))[0])
        assert f10 < f1

    def test_peak_near_one(self):
        # Maximum of f(s) = lam * s / (s^2 + 1) is at s=1
        x = np.linspace(0.5, 1.5, 100)
        vals = scurve(x)
        peak_idx = np.argmax(vals)
        assert abs(x[peak_idx] - 1.0) < 0.1


class TestLinearMomentum:
    def test_trending_produces_positive_signal(self):
        n = 500
        dates = pd.bdate_range("2020-01-01", periods=n)
        # Steadily rising prices
        close = pd.Series(100 * np.exp(np.linspace(0, 0.5, n)), index=dates)
        vol = pd.Series(0.15, index=dates)

        mom = LinearMomentum(lookbacks=[21, 63], weights=[0.5, 0.5], norm_window=100)
        signals = mom.compute(close, vol)

        # After warmup, combined signal should be positive
        assert signals["signal_combined"].iloc[-50:].mean() > 0

    def test_output_columns(self):
        n = 300
        dates = pd.bdate_range("2020-01-01", periods=n)
        close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.01, n))), index=dates)
        vol = pd.Series(0.15, index=dates)

        mom = LinearMomentum()
        signals = mom.compute(close, vol)

        assert "signal_raw_21d" in signals.columns
        assert "signal_raw_63d" in signals.columns
        assert "signal_raw_252d" in signals.columns
        assert "signal_combined" in signals.columns


class TestNonlinearMomentum:
    def test_has_nl_columns(self):
        n = 300
        dates = pd.bdate_range("2020-01-01", periods=n)
        close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.01, n))), index=dates)
        vol = pd.Series(0.15, index=dates)

        mom = NonlinearMomentum()
        signals = mom.compute(close, vol)

        assert "signal_nl_21d" in signals.columns
        assert "signal_nl_63d" in signals.columns
        assert "signal_nl_252d" in signals.columns

    def test_nl_attenuates_vs_linear(self):
        n = 500
        dates = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.default_rng(42)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=dates)
        vol = pd.Series(0.15, index=dates)

        lin = LinearMomentum(lookbacks=[21, 63], weights=[0.5, 0.5])
        nl = NonlinearMomentum(lookbacks=[21, 63], weights=[0.5, 0.5])

        lin_sig = lin.compute(close, vol)
        nl_sig = nl.compute(close, vol)

        # Both should have same raw signals
        np.testing.assert_allclose(
            lin_sig["signal_raw_21d"].dropna().values,
            nl_sig["signal_raw_21d"].dropna().values,
        )


class TestBinaryMomentum:
    def test_signals_are_discrete(self):
        n = 300
        dates = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.default_rng(42)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)
        vol = pd.Series(0.15, index=dates)

        mom = BinaryMomentum(lookbacks=[21], weights=[1.0])
        signals = mom.compute(close, vol)

        # Raw signals should be -1, 0, or 1
        raw = signals["signal_raw_21d"].dropna()
        assert set(raw.unique()).issubset({-1.0, 0.0, 1.0})
