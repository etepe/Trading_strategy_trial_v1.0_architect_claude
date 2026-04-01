"""Adaptive rebalancing logic."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RebalanceScheduler:
    """Adaptive rebalancing: weekly baseline with forced rebalances on vol/signal jumps.

    Args:
        frequency: Base rebalance frequency in trading days (default 5 = weekly).
        vol_threshold: Force rebalance if vol changes by this fraction (default 0.20).
        signal_threshold: Force rebalance if signal jumps by this many std (default 1.5).
    """

    def __init__(
        self,
        frequency: int = 5,
        vol_threshold: float = 0.20,
        signal_threshold: float = 1.5,
    ) -> None:
        self.frequency = frequency
        self.vol_threshold = vol_threshold
        self.signal_threshold = signal_threshold

    def get_rebalance_mask(
        self, vol: pd.Series, signal: pd.Series
    ) -> pd.Series:
        """Determine which days to rebalance.

        Args:
            vol: Volatility series.
            signal: Combined signal series.

        Returns:
            Boolean series (True = rebalance on this day).
        """
        n = len(vol)
        mask = pd.Series(False, index=vol.index)

        # Always rebalance on first valid day
        first_valid = vol.first_valid_index()
        if first_valid is not None:
            mask.loc[first_valid] = True

        # Weekly schedule
        for i in range(n):
            if i % self.frequency == 0:
                mask.iloc[i] = True

        # Forced rebalance on vol spike
        vol_change = (vol / vol.shift(1) - 1).abs()
        vol_force = vol_change > self.vol_threshold
        mask = mask | vol_force

        # Forced rebalance on signal jump
        signal_diff = signal.diff().abs()
        signal_std = signal.diff().rolling(window=63, min_periods=10).std()
        signal_force = signal_diff > (self.signal_threshold * signal_std)
        mask = mask | signal_force.fillna(False)

        return mask

    def apply_rebalance(
        self, target_positions: pd.Series, mask: pd.Series
    ) -> pd.Series:
        """Apply rebalance mask: carry forward positions on non-rebalance days.

        Args:
            target_positions: Desired position on each day.
            mask: Boolean rebalance mask.

        Returns:
            Actual positions (carried forward between rebalances).
        """
        actual = target_positions.copy()
        actual[~mask] = np.nan
        actual = actual.ffill().fillna(0.0)
        actual.name = "position"
        return actual
