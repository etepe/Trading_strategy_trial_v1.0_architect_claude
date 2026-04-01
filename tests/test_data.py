"""Tests for data pipeline."""

import numpy as np
import pandas as pd
import pytest

from fetm.data.clean import clean_ohlcv
from fetm.data.features import compute_features


@pytest.fixture
def sample_ohlcv():
    """Create a small synthetic OHLCV DataFrame."""
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6],
        },
        index=dates,
    )


def test_clean_removes_negative_close(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[2, df.columns.get_loc("close")] = -1.0
    result = clean_ohlcv(df)
    assert (result["close"] > 0).all()
    assert len(result) == 4


def test_clean_fixes_high_low_swap(sample_ohlcv):
    df = sample_ohlcv.copy()
    # Swap high and low for row 1
    df.iloc[1, df.columns.get_loc("high")] = 99.0
    df.iloc[1, df.columns.get_loc("low")] = 103.0
    result = clean_ohlcv(df)
    assert (result["high"] >= result["low"]).all()


def test_clean_forward_fills_nan(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[2, df.columns.get_loc("close")] = np.nan
    result = clean_ohlcv(df)
    assert not result["close"].isna().any()


def test_clean_removes_duplicate_dates(sample_ohlcv):
    df = pd.concat([sample_ohlcv, sample_ohlcv.iloc[[0]]])
    result = clean_ohlcv(df)
    assert not result.index.duplicated().any()


def test_compute_features(sample_ohlcv):
    result = compute_features(sample_ohlcv)

    # First log_return should be NaN
    assert np.isnan(result["log_return"].iloc[0])

    # Check log_return calculation for row 1
    expected = np.log(101.5 / 100.5)
    assert abs(result["log_return"].iloc[1] - expected) < 1e-10

    # Check log_hl is positive
    assert (result["log_hl"] > 0).all()

    # Check simple_return
    expected_sr = (101.5 - 100.5) / 100.5
    assert abs(result["simple_return"].iloc[1] - expected_sr) < 1e-10
