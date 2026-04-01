"""Data download module — fetch OHLCV data from yfinance."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from fetm.config import load_config

logger = logging.getLogger(__name__)


def download_ticker(
    ticker: str = "SPY",
    start_date: str = "1993-01-01",
    end_date: str | None = None,
    output_dir: str | Path = "data/raw",
    max_retries: int = 3,
) -> Path:
    """Download OHLCV data from yfinance and save as parquet.

    Args:
        ticker: Stock/ETF ticker symbol.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date (defaults to today).
        output_dir: Directory to save parquet files.
        max_retries: Number of download retries.

    Returns:
        Path to the saved parquet file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}.parquet"

    for attempt in range(max_retries):
        try:
            logger.info(
                "Downloading %s from %s (attempt %d/%d)",
                ticker, start_date, attempt + 1, max_retries,
            )
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )

            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Standardize column names to lowercase
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df.index.name = "date"

            # Ensure we have the required columns
            required = {"open", "high", "low", "close", "volume"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            df.to_parquet(output_path)
            logger.info(
                "Saved %d rows to %s (%s to %s)",
                len(df), output_path,
                df.index[0].strftime("%Y-%m-%d"),
                df.index[-1].strftime("%Y-%m-%d"),
            )
            return output_path

        except Exception as e:
            logger.warning("Download attempt %d failed: %s", attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

    raise RuntimeError("Unreachable")


def load_raw_data(ticker: str = "SPY", data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Load previously downloaded raw data from parquet.

    Args:
        ticker: Ticker symbol.
        data_dir: Directory containing parquet files.

    Returns:
        DataFrame with OHLCV data, datetime index.
    """
    path = Path(data_dir) / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No data file for {ticker} at {path}. Run download first."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows for %s", len(df), ticker)
    return df
