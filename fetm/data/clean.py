"""Data cleaning and validation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate OHLCV data.

    Args:
        df: Raw OHLCV DataFrame with datetime index.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Ensure datetime index sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Remove duplicate dates
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        logger.warning("Removing %d duplicate dates", n_dupes)
        df = df[~df.index.duplicated(keep="last")]

    # Drop rows with non-positive close
    bad_close = df["close"] <= 0
    if bad_close.any():
        logger.warning("Dropping %d rows with close <= 0", bad_close.sum())
        df = df[~bad_close]

    # Forward-fill NaN values (rare, but handles holidays/gaps)
    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        logger.warning("Forward-filling %d NaN values", n_nan)
        df = df.ffill()

    # Validate OHLC consistency: high >= low
    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        logger.warning(
            "Fixing %d rows where high < low (swapping)", bad_hl.sum()
        )
        swap_mask = bad_hl
        df.loc[swap_mask, ["high", "low"]] = df.loc[
            swap_mask, ["low", "high"]
        ].values

    # Ensure high >= open and high >= close
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    # Ensure low <= open and low <= close
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    # Drop any remaining NaN rows
    df = df.dropna()

    logger.info(
        "Clean data: %d rows, %s to %s",
        len(df),
        df.index[0].strftime("%Y-%m-%d"),
        df.index[-1].strftime("%Y-%m-%d"),
    )
    return df
