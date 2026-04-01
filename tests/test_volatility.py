"""Tests for volatility estimators."""

import numpy as np
import pandas as pd
import pytest

from fetm.signals.volatility import (
    CompositeVolatility,
    EWMAVolatility,
    FETVolatility,
    ParkinsonVolatility,
)


@pytest.fixture
def constant_returns():
    """Returns series with constant daily vol ~1%."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    returns = rng.normal(0, 0.01, n)  # daily vol = 1%
    return pd.Series(returns, index=dates)


@pytest.fixture
def step_vol_returns():
    """Returns with a vol regime change: 1% for 250 days then 3%."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    vol = np.where(np.arange(n) < 250, 0.01, 0.03)
    returns = rng.normal(0, 1, n) * vol
    return pd.Series(returns, index=dates), vol


class TestEWMA:
    def test_constant_vol_converges(self, constant_returns):
        ewma = EWMAVolatility(halflife=60)
        vol = ewma.estimate_series(constant_returns)
        # Should converge near 0.01 * sqrt(252) ~ 0.159
        last_100_mean = vol.iloc[-100:].mean()
        assert 0.10 < last_100_mean < 0.25

    def test_annualization(self, constant_returns):
        ewma = EWMAVolatility(halflife=60)
        vol = ewma.estimate_series(constant_returns)
        # Daily vol = 0.01, annualized ~ 0.159
        expected = 0.01 * np.sqrt(252)
        last_val = vol.iloc[-1]
        assert abs(last_val - expected) / expected < 0.3  # within 30%

    def test_responds_to_vol_increase(self, step_vol_returns):
        returns, _ = step_vol_returns
        ewma = EWMAVolatility(halflife=60)
        vol = ewma.estimate_series(returns)
        # Vol at end should be higher than at day 200
        assert vol.iloc[-1] > vol.iloc[200]

    def test_handles_leading_nan(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        returns = pd.Series(np.nan, index=dates)
        returns.iloc[10:] = np.random.default_rng(42).normal(0, 0.01, 90)
        ewma = EWMAVolatility(halflife=20)
        vol = ewma.estimate_series(returns)
        assert vol.iloc[:10].isna().all()
        assert not vol.iloc[-1:].isna().any()


class TestParkinson:
    def test_basic(self):
        n = 100
        dates = pd.bdate_range("2020-01-01", periods=n)
        # Create prices with known range
        high = pd.Series(np.full(n, 102.0), index=dates)
        low = pd.Series(np.full(n, 98.0), index=dates)
        park = ParkinsonVolatility(window=20)
        vol = park.estimate_series(high, low)
        # With constant H/L ratio, vol should be constant after window
        assert not vol.iloc[-1:].isna().any()
        assert vol.iloc[-1] > 0

    def test_higher_range_means_higher_vol(self):
        n = 100
        dates = pd.bdate_range("2020-01-01", periods=n)
        high_narrow = pd.Series(np.full(n, 101.0), index=dates)
        low_narrow = pd.Series(np.full(n, 99.0), index=dates)
        high_wide = pd.Series(np.full(n, 105.0), index=dates)
        low_wide = pd.Series(np.full(n, 95.0), index=dates)

        park = ParkinsonVolatility(window=20)
        vol_narrow = park.estimate_series(high_narrow, low_narrow)
        vol_wide = park.estimate_series(high_wide, low_wide)
        assert vol_wide.iloc[-1] > vol_narrow.iloc[-1]


class TestFET:
    def test_basic_exit_detection(self):
        """Test that FET detects barrier crossings."""
        n = 200
        dates = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.default_rng(42)

        # Trending prices that will cross corridors
        returns = rng.normal(0.001, 0.015, n)
        close = 100 * np.exp(np.cumsum(returns))
        open_ = close * np.exp(rng.normal(0, 0.002, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))

        close = pd.Series(close, index=dates)
        open_ = pd.Series(open_, index=dates)
        high = pd.Series(high, index=dates)
        low = pd.Series(low, index=dates)
        ewma_fallback = pd.Series(np.full(n, 0.15), index=dates)

        fet = FETVolatility(corridor_width=0.012, buffer_size=5)
        vol, diag = fet.estimate_series(open_, high, low, close, ewma_fallback)

        # Should have some non-NaN vol estimates
        assert vol.notna().sum() > 0
        # Should have recorded some exit times
        assert diag["exit_time_last"].notna().sum() > 0

    def test_jensen_correction(self):
        """Verify Jensen correction reduces estimate."""
        n = 200
        dates = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, n)
        close = 100 * np.exp(np.cumsum(returns))
        open_ = close * np.exp(rng.normal(0, 0.003, n))
        high = np.maximum(open_, close) * 1.005
        low = np.minimum(open_, close) * 0.995

        kw = dict(
            open_=pd.Series(open_, index=dates),
            high=pd.Series(high, index=dates),
            low=pd.Series(low, index=dates),
            close=pd.Series(close, index=dates),
            ewma_fallback=pd.Series(np.full(n, 0.15), index=dates),
        )

        fet_corr = FETVolatility(corridor_width=0.015, buffer_size=10, bias_correction=True)
        fet_nocorr = FETVolatility(corridor_width=0.015, buffer_size=10, bias_correction=False)

        vol_corr, _ = fet_corr.estimate_series(**kw)
        vol_nocorr, _ = fet_nocorr.estimate_series(**kw)

        # Corrected should be slightly lower (n/(n+0.25) < 1)
        valid = vol_corr.notna() & vol_nocorr.notna()
        if valid.sum() > 10:
            assert vol_corr[valid].mean() < vol_nocorr[valid].mean()


class TestComposite:
    def test_low_divergence_low_weight(self):
        """When FET and EWMA agree, FET weight should be low."""
        idx = pd.date_range("2020-01-01", periods=100)
        ewma = pd.Series(np.full(100, 0.15), index=idx)
        fet = pd.Series(np.full(100, 0.16), index=idx)  # close to EWMA

        comp = CompositeVolatility()
        composite, w = comp.blend(fet, ewma)

        assert w.mean() < 0.3  # low weight when they agree

    def test_high_divergence_high_weight(self):
        """When FET and EWMA diverge, FET weight should be high."""
        idx = pd.date_range("2020-01-01", periods=100)
        ewma = pd.Series(np.full(100, 0.15), index=idx)
        fet = pd.Series(np.full(100, 0.45), index=idx)  # 3x EWMA

        comp = CompositeVolatility()
        composite, w = comp.blend(fet, ewma)

        assert w.mean() > 0.5  # high weight when they diverge

    def test_nan_fallback(self):
        """When FET is NaN, composite should equal EWMA."""
        idx = pd.date_range("2020-01-01", periods=10)
        ewma = pd.Series(0.15, index=idx)
        fet = pd.Series(np.nan, index=idx)

        comp = CompositeVolatility()
        composite, w = comp.blend(fet, ewma)

        assert np.allclose(composite.values, 0.15)
        assert np.allclose(w.values, 0.0)
