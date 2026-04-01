"""Generate synthetic SPY-like data for development when yfinance is unavailable."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_spy_like_data(
    start_date: str = "1993-01-29",
    end_date: str = "2026-03-28",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic SPY-like daily OHLCV data.

    Uses a regime-switching GBM with parameters calibrated to historical SPY:
    - Average annual return ~10%
    - Average annual vol ~16%, with regime shifts
    - Crisis periods with higher vol and negative drift
    - Realistic OHLC relationships

    Args:
        start_date: Start date.
        end_date: End date.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with OHLCV data matching real SPY structure.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, end_date)
    n = len(dates)

    # Regime parameters: (annual_drift, annual_vol, transition_prob_to_other)
    # Normal regime: ~10% return, ~14% vol
    # High vol regime: -5% return, ~30% vol
    # Crisis regime: -40% return, ~60% vol
    regimes = {
        "normal": {"drift": 0.10, "vol": 0.14, "p_to_high": 0.003, "p_to_crisis": 0.0005},
        "high_vol": {"drift": -0.05, "vol": 0.30, "p_to_normal": 0.02, "p_to_crisis": 0.005},
        "crisis": {"drift": -0.40, "vol": 0.60, "p_to_normal": 0.001, "p_to_high": 0.05},
    }

    # Simulate regime path
    regime_state = np.zeros(n, dtype=int)  # 0=normal, 1=high_vol, 2=crisis
    state = 0
    for i in range(1, n):
        u = rng.random()
        if state == 0:
            if u < regimes["normal"]["p_to_crisis"]:
                state = 2
            elif u < regimes["normal"]["p_to_crisis"] + regimes["normal"]["p_to_high"]:
                state = 1
        elif state == 1:
            if u < regimes["high_vol"]["p_to_crisis"]:
                state = 2
            elif u < regimes["high_vol"]["p_to_crisis"] + regimes["high_vol"]["p_to_normal"]:
                state = 0
        else:  # crisis
            if u < regimes["crisis"]["p_to_high"]:
                state = 1
            elif u < regimes["crisis"]["p_to_high"] + regimes["crisis"]["p_to_normal"]:
                state = 0
        regime_state[i] = state

    # Map regimes to daily parameters
    regime_names = ["normal", "high_vol", "crisis"]
    daily_drift = np.zeros(n)
    daily_vol = np.zeros(n)
    for i, s in enumerate(regime_state):
        r = regimes[regime_names[s]]
        daily_drift[i] = r["drift"] / 252
        daily_vol[i] = r["vol"] / np.sqrt(252)

    # Generate log returns with GARCH-like vol clustering
    log_returns = daily_drift + daily_vol * rng.standard_normal(n)

    # Generate price path starting at SPY ~$44 (Jan 1993 level)
    log_price = np.cumsum(log_returns)
    log_price = log_price - log_price[0] + np.log(44.0)
    close = np.exp(log_price)

    # Generate realistic OHLC from close
    # Intraday range is proportional to daily vol
    intraday_range = daily_vol * close * 1.5  # slightly wider than vol
    open_prices = close * np.exp(rng.normal(0, 0.002, n))

    # High is max of open, close, plus some range above
    high = np.maximum(open_prices, close) + np.abs(rng.normal(0, 1, n)) * intraday_range * 0.5
    # Low is min of open, close, minus some range below
    low = np.minimum(open_prices, close) - np.abs(rng.normal(0, 1, n)) * intraday_range * 0.5

    # Ensure consistency
    high = np.maximum(high, np.maximum(open_prices, close))
    low = np.minimum(low, np.minimum(open_prices, close))
    low = np.maximum(low, 0.01)  # no negative prices

    # Volume: higher in crisis, correlated with vol
    base_volume = 50e6
    volume = base_volume * (1 + 2 * daily_vol / daily_vol.mean()) * rng.lognormal(0, 0.3, n)

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"

    logger.info(
        "Generated synthetic SPY data: %d rows, price range %.1f to %.1f",
        len(df), close.min(), close.max(),
    )
    return df
