"""Volatility estimators: EWMA, Parkinson, First Exit-Time, and Composite."""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EWMAVolatility:
    """Exponentially Weighted Moving Average volatility estimator.

    Args:
        halflife: Halflife in days for exponential decay (default 60).
    """

    def __init__(self, halflife: int = 60) -> None:
        self.halflife = halflife
        self.lambda_ = np.exp(-1.0 / halflife)

    def estimate_series(self, log_returns: pd.Series) -> pd.Series:
        """Compute annualized EWMA volatility for a full return series.

        Args:
            log_returns: Series of log returns (may contain leading NaN).

        Returns:
            Annualized EWMA volatility series (same index as input).
        """
        returns = log_returns.values.copy()
        n = len(returns)
        vol = np.full(n, np.nan)

        # Find first valid (non-NaN) index
        valid_mask = ~np.isnan(returns)
        if not valid_mask.any():
            return pd.Series(vol, index=log_returns.index, name="vol_ewma")

        first_valid = np.argmax(valid_mask)
        init_end = min(first_valid + self.halflife, n)

        # Initialize with sample variance of first halflife returns
        init_returns = returns[first_valid:init_end]
        init_returns = init_returns[~np.isnan(init_returns)]
        if len(init_returns) < 2:
            return pd.Series(vol, index=log_returns.index, name="vol_ewma")

        variance = np.var(init_returns, ddof=1)

        # Fill initialization period with sample vol
        ann_vol = np.sqrt(variance) * np.sqrt(252)
        for i in range(first_valid, init_end):
            vol[i] = ann_vol

        # Recursive EWMA from init_end onward
        lam = self.lambda_
        for i in range(init_end, n):
            r = returns[i]
            if np.isnan(r):
                vol[i] = vol[i - 1] if i > 0 else np.nan
                continue
            variance = lam * variance + (1 - lam) * r * r
            vol[i] = np.sqrt(variance) * np.sqrt(252)

        return pd.Series(vol, index=log_returns.index, name="vol_ewma")


class ParkinsonVolatility:
    """Parkinson range-based volatility estimator.

    Args:
        window: Rolling window in days (default 20).
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def estimate_series(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Compute annualized Parkinson volatility.

        Args:
            high: Daily high prices.
            low: Daily low prices.

        Returns:
            Annualized Parkinson volatility series.
        """
        log_hl_sq = np.log(high / low) ** 2
        # Parkinson formula: sqrt( (1/n) * sum(log(H/L)^2) / (4*log(2)) )
        park_var = log_hl_sq.rolling(window=self.window).mean() / (4 * np.log(2))
        vol = np.sqrt(park_var) * np.sqrt(252)
        vol.name = "vol_parkinson"
        return vol


class FETVolatility:
    """First Exit-Time volatility estimator.

    Estimates volatility from the time it takes price to exit a corridor,
    rather than from fixed-interval returns. Converges faster during
    regime transitions.

    Args:
        corridor_width: Half-width delta of log-price corridor.
        buffer_size: Number of recent exit times to average.
        max_no_exit_days: Force corridor reset after this many days.
        bias_correction: Apply Jensen bias correction.
    """

    def __init__(
        self,
        corridor_width: float = 0.012,
        buffer_size: int = 30,
        max_no_exit_days: int = 60,
        bias_correction: bool = True,
    ) -> None:
        self.delta = corridor_width
        self.buffer_size = buffer_size
        self.max_no_exit_days = max_no_exit_days
        self.bias_correction = bias_correction

    def estimate_series(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        ewma_fallback: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Compute annualized FET volatility from daily OHLC.

        Args:
            open_: Daily open prices.
            high: Daily high prices.
            low: Daily low prices.
            close: Daily close prices.
            ewma_fallback: EWMA vol series to use before buffer is full.

        Returns:
            Tuple of (vol_series, diagnostics_dataframe).
        """
        n = len(close)
        o = open_.values
        h = high.values
        l = low.values  # noqa: E741
        c = close.values
        ewma = ewma_fallback.values

        vol = np.full(n, np.nan)
        exit_time_last = np.full(n, np.nan)
        exit_times_mean = np.full(n, np.nan)
        corridor_upper = np.full(n, np.nan)
        corridor_lower = np.full(n, np.nan)
        days_since_exit = np.zeros(n, dtype=int)
        fet_weight = np.full(n, np.nan)

        exit_buffer: deque[float] = deque(maxlen=self.buffer_size)

        # Find first valid day (no NaN in prices)
        start = 0
        while start < n and (np.isnan(c[start]) or np.isnan(h[start]) or np.isnan(l[start])):
            start += 1
        if start >= n:
            idx = close.index
            return (
                pd.Series(vol, index=idx, name="vol_fet"),
                pd.DataFrame(
                    {"exit_time_last": exit_time_last, "exit_times_mean": exit_times_mean,
                     "corridor_upper": corridor_upper, "corridor_lower": corridor_lower,
                     "days_since_exit": days_since_exit},
                    index=idx,
                ),
            )

        ref_price = c[start]
        t_last_reset = start
        delta = self.delta

        for i in range(start, n):
            current_delta = delta
            upper = ref_price * np.exp(current_delta)
            lower = ref_price * np.exp(-current_delta)

            corridor_upper[i] = upper
            corridor_lower[i] = lower
            days_since = i - t_last_reset
            days_since_exit[i] = days_since

            # Check for forced reset (stale corridor)
            if days_since >= self.max_no_exit_days:
                logger.debug("Forced corridor reset at day %d (stale for %d days)", i, days_since)
                ref_price = c[i]
                t_last_reset = i
                # Widen delta temporarily for next iteration
                current_delta = delta * 1.5
                upper = ref_price * np.exp(current_delta)
                lower = ref_price * np.exp(-current_delta)
                corridor_upper[i] = upper
                corridor_lower[i] = lower
                days_since_exit[i] = 0

            # Check barrier breach
            hit_upper = h[i] >= upper
            hit_lower = l[i] <= lower

            if hit_upper or hit_lower:
                # Compute exit time in years
                tau_days = days_since + 1  # +1 because current day counts
                tau = tau_days / 252.0

                # Intra-day interpolation
                if hit_upper and hit_lower:
                    # Both barriers hit same day
                    tau_intraday = 0.5 / 252.0
                elif hit_upper:
                    denom = h[i] - o[i]
                    if denom > 0:
                        penetration = (h[i] - upper) / denom
                        penetration = min(max(penetration, 0.0), 1.0)
                        tau_intraday = (1.0 - penetration * 0.5) / 252.0
                    else:
                        tau_intraday = 0.5 / 252.0
                else:  # hit_lower
                    denom = o[i] - l[i]
                    if denom > 0:
                        penetration = (lower - l[i]) / denom
                        penetration = min(max(penetration, 0.0), 1.0)
                        tau_intraday = (1.0 - penetration * 0.5) / 252.0
                    else:
                        tau_intraday = 0.5 / 252.0

                # Adjust: replace full last day with intra-day estimate
                tau = tau - (1.0 / 252.0) + tau_intraday
                tau = max(tau, 1e-6)  # floor at tiny positive

                exit_buffer.append(tau)
                exit_time_last[i] = tau

                # Reset corridor
                ref_price = c[i]
                t_last_reset = i
                days_since_exit[i] = 0

            # Compute volatility estimate
            if len(exit_buffer) >= 1:
                mean_tau = np.mean(list(exit_buffer))
                exit_times_mean[i] = mean_tau
                sigma = current_delta / np.sqrt(mean_tau)

                # Jensen bias correction
                if self.bias_correction:
                    buf_n = len(exit_buffer)
                    sigma *= buf_n / (buf_n + 0.25)

                vol[i] = sigma
            else:
                # Fallback to EWMA before we have exit times
                vol[i] = ewma[i] if not np.isnan(ewma[i]) else np.nan

        idx = close.index
        diagnostics = pd.DataFrame(
            {
                "exit_time_last": exit_time_last,
                "exit_times_mean": exit_times_mean,
                "corridor_upper": corridor_upper,
                "corridor_lower": corridor_lower,
                "days_since_exit": days_since_exit,
            },
            index=idx,
        )
        return pd.Series(vol, index=idx, name="vol_fet"), diagnostics


class CompositeVolatility:
    """Blended FET + EWMA volatility with adaptive weighting.

    When FET and EWMA agree, lean on EWMA (smoother).
    When they diverge (regime transition), lean on FET (faster).

    Args:
        sigmoid_k: Steepness of blending sigmoid (default 5.0).
        max_weight: Maximum FET weight (default 0.8).
    """

    def __init__(self, sigmoid_k: float = 5.0, max_weight: float = 0.8) -> None:
        self.sigmoid_k = sigmoid_k
        self.max_weight = max_weight

    def blend(
        self, vol_fet: pd.Series, vol_ewma: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Compute composite volatility and FET weight.

        Args:
            vol_fet: FET volatility series.
            vol_ewma: EWMA volatility series.

        Returns:
            Tuple of (composite_vol, fet_weight).
        """
        # Compute divergence
        divergence = np.abs(vol_fet - vol_ewma) / vol_ewma.clip(lower=1e-6)

        # Sigmoid blending weight
        w = self.max_weight / (1.0 + np.exp(-(self.sigmoid_k * divergence) + 2.0))

        composite = w * vol_fet + (1.0 - w) * vol_ewma

        # Where FET is NaN, fall back to EWMA
        mask = vol_fet.isna()
        composite[mask] = vol_ewma[mask]
        w[mask] = 0.0

        composite.name = "vol_composite"
        w.name = "fet_weight"
        return composite, w
