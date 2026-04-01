"""Derived features from OHLCV data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to OHLCV DataFrame.

    Adds: log_return, simple_return, log_hl (for Parkinson vol).

    Args:
        df: Clean OHLCV DataFrame with datetime index.

    Returns:
        DataFrame with additional feature columns.
    """
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["simple_return"] = df["close"].pct_change()
    df["log_hl"] = np.log(df["high"] / df["low"])
    return df
