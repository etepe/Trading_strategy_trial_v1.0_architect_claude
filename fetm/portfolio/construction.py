"""Position sizing and portfolio construction."""

import numpy as np
import pandas as pd


class PositionSizer:
    """Volatility-targeted position sizing.

    Position = Signal * (TargetVol / EstimatedVol), capped at MaxLeverage.

    Args:
        target_vol: Target annualized volatility (default 0.15).
        max_leverage: Maximum absolute position size (default 3.0).
    """

    def __init__(self, target_vol: float = 0.15, max_leverage: float = 3.0) -> None:
        self.target_vol = target_vol
        self.max_leverage = max_leverage

    def size(self, signal: pd.Series, vol: pd.Series) -> pd.Series:
        """Compute position sizes from signal and volatility.

        Args:
            signal: Combined momentum signal.
            vol: Annualized volatility estimate.

        Returns:
            Position series, clipped to [-max_leverage, max_leverage].
        """
        position = signal * (self.target_vol / vol.clip(lower=1e-6))
        position = position.clip(-self.max_leverage, self.max_leverage)
        position = position.fillna(0.0)
        position.name = "position"
        return position
