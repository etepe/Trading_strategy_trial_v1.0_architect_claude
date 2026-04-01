"""Momentum signal generators: Linear, Binary, and Nonlinear (S-curve)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def scurve(s: np.ndarray | pd.Series, lam: float = 0.394) -> np.ndarray:
    """Apply S-curve nonlinear transformation.

    f(s) = lam * s / (s^2 + 1)

    This attenuates extreme signals (|s| > 1) while maintaining
    near-linear response for moderate signals.

    Args:
        s: Raw normalized signal values.
        lam: Scaling parameter (default 0.394 per Moskowitz et al. 2025).

    Returns:
        Transformed signal values.
    """
    s = np.asarray(s, dtype=float)
    return lam * s / (s ** 2 + 1.0)


class LinearMomentum:
    """Linear time-series momentum signal (Moskowitz 2012 baseline).

    Computes volatility-normalized returns at multiple horizons
    and combines them with fixed weights.

    Args:
        lookbacks: Lookback periods in days (default [21, 63, 252]).
        weights: Horizon weights (default [0.25, 0.40, 0.35]).
        norm_window: Rolling window for signal normalization (default 252).
        signal_cap: Maximum absolute raw signal value (default 5.0).
    """

    def __init__(
        self,
        lookbacks: list[int] | None = None,
        weights: list[float] | None = None,
        norm_window: int = 252,
        signal_cap: float = 5.0,
    ) -> None:
        self.lookbacks = lookbacks or [21, 63, 252]
        self.weights = weights or [0.25, 0.40, 0.35]
        self.norm_window = norm_window
        self.signal_cap = signal_cap

    def compute(self, close: pd.Series, vol: pd.Series) -> pd.DataFrame:
        """Compute momentum signals at all horizons.

        Args:
            close: Daily close prices.
            vol: Annualized volatility series (for normalization).

        Returns:
            DataFrame with signal columns for each horizon and combined.
        """
        result = pd.DataFrame(index=close.index)

        combined = pd.Series(0.0, index=close.index)
        for h, w in zip(self.lookbacks, self.weights):
            # Log return over h days
            r_h = np.log(close / close.shift(h))
            # Normalize by annualized vol (convert vol back to h-day scale)
            # vol is annualized, so h-day vol = vol * sqrt(h/252)
            vol_h = vol * np.sqrt(h / 252.0)
            s_h = r_h / vol_h.clip(lower=1e-6)
            # Cap extreme values
            s_h = s_h.clip(-self.signal_cap, self.signal_cap)

            result[f"signal_raw_{h}d"] = s_h
            combined += w * s_h

        # Normalize combined signal by rolling std
        roll_std = combined.rolling(window=self.norm_window, min_periods=60).std()
        combined_norm = combined / roll_std.clip(lower=1e-6)
        combined_norm = combined_norm.clip(-self.signal_cap, self.signal_cap)
        result["signal_combined"] = combined_norm

        return result


class NonlinearMomentum:
    """Nonlinear (S-curve) momentum signal (Moskowitz et al. 2025).

    Same as LinearMomentum but applies S-curve transformation to each
    horizon signal before combining.

    Args:
        lookbacks: Lookback periods in days.
        weights: Horizon weights.
        scurve_lambda: S-curve scaling parameter.
        norm_window: Rolling window for signal normalization.
        signal_cap: Maximum absolute raw signal value.
    """

    def __init__(
        self,
        lookbacks: list[int] | None = None,
        weights: list[float] | None = None,
        scurve_lambda: float = 0.394,
        norm_window: int = 252,
        signal_cap: float = 5.0,
    ) -> None:
        self.lookbacks = lookbacks or [21, 63, 252]
        self.weights = weights or [0.25, 0.40, 0.35]
        self.scurve_lambda = scurve_lambda
        self.norm_window = norm_window
        self.signal_cap = signal_cap

    def compute(self, close: pd.Series, vol: pd.Series) -> pd.DataFrame:
        """Compute nonlinear momentum signals.

        Args:
            close: Daily close prices.
            vol: Annualized volatility series.

        Returns:
            DataFrame with raw and S-curve transformed signals.
        """
        result = pd.DataFrame(index=close.index)

        combined = pd.Series(0.0, index=close.index)
        for h, w in zip(self.lookbacks, self.weights):
            r_h = np.log(close / close.shift(h))
            vol_h = vol * np.sqrt(h / 252.0)
            s_h = r_h / vol_h.clip(lower=1e-6)
            s_h = s_h.clip(-self.signal_cap, self.signal_cap)

            result[f"signal_raw_{h}d"] = s_h

            # Apply S-curve transformation
            nl_h = scurve(s_h, self.scurve_lambda)
            result[f"signal_nl_{h}d"] = nl_h

            combined += w * nl_h

        # Normalize combined signal
        roll_std = combined.rolling(window=self.norm_window, min_periods=60).std()
        combined_norm = combined / roll_std.clip(lower=1e-6)
        combined_norm = combined_norm.clip(-self.signal_cap, self.signal_cap)
        result["signal_combined"] = combined_norm

        return result


class BinaryMomentum:
    """Binary momentum signal — sign of past returns.

    Uses sign(return) rather than magnitude, combined with S-curve
    transformation per Moskowitz 2012.

    Args:
        lookbacks: Lookback periods in days.
        weights: Horizon weights.
        norm_window: Rolling window for signal normalization.
    """

    def __init__(
        self,
        lookbacks: list[int] | None = None,
        weights: list[float] | None = None,
        norm_window: int = 252,
    ) -> None:
        self.lookbacks = lookbacks or [21, 63, 252]
        self.weights = weights or [0.25, 0.40, 0.35]
        self.norm_window = norm_window

    def compute(self, close: pd.Series, vol: pd.Series) -> pd.DataFrame:
        """Compute binary momentum signals.

        Args:
            close: Daily close prices.
            vol: Annualized volatility (used for position sizing, not signal).

        Returns:
            DataFrame with binary signal columns.
        """
        result = pd.DataFrame(index=close.index)

        combined = pd.Series(0.0, index=close.index)
        for h, w in zip(self.lookbacks, self.weights):
            r_h = np.log(close / close.shift(h))
            sign_h = np.sign(r_h)
            result[f"signal_raw_{h}d"] = sign_h
            combined += w * sign_h

        # Normalize
        roll_std = combined.rolling(window=self.norm_window, min_periods=60).std()
        combined_norm = combined / roll_std.clip(lower=1e-6)
        combined_norm = combined_norm.clip(-5.0, 5.0)
        result["signal_combined"] = combined_norm

        return result
