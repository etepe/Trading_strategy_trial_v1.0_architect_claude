"""Transaction cost model."""

from __future__ import annotations

import pandas as pd


class TransactionCostModel:
    """Simple proportional transaction cost model.

    Cost = |delta_position| * cost_bps

    Args:
        cost_bps: Cost per unit of position change (default 0.0005 = 5 bps).
    """

    def __init__(self, cost_bps: float = 0.0005) -> None:
        self.cost_bps = cost_bps

    def compute(self, positions: pd.Series) -> pd.Series:
        """Compute daily transaction costs.

        Args:
            positions: Position series.

        Returns:
            Cost series (positive values represent costs incurred).
        """
        turnover = positions.diff().abs()
        turnover.iloc[0] = positions.iloc[0].item() if len(positions) > 0 else 0.0
        costs = turnover * self.cost_bps
        costs.name = "transaction_cost"
        return costs
