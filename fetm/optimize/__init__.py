"""Walk-forward parameter optimization for the FETM strategy.

Submodules:
    search_space        — declarative list of tunable parameters.
    objective           — single-trial evaluator (backtest + scalar score).
    walk_forward_optuna — rolling Optuna TPE search over walk-forward windows.
"""

from fetm.optimize.search_space import Param, SEARCH_SPACE
from fetm.optimize.objective import evaluate_params, score_from_metrics
from fetm.optimize.walk_forward_optuna import WalkForwardOptunaOptimizer

__all__ = [
    "Param",
    "SEARCH_SPACE",
    "evaluate_params",
    "score_from_metrics",
    "WalkForwardOptunaOptimizer",
]
