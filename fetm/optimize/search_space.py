"""Declarative search space for FETM strategy parameter optimization.

Each entry describes a single tunable parameter by its dot-path inside
``config/settings.yaml``, its numeric type, and the range to explore. The
Optuna walk-forward optimizer iterates over this list to build a trial.

Adding a new parameter is a one-line change here — no other file needs to
be touched unless the parameter has a non-trivial encoding (e.g. list types
like ``momentum.lookback_days``, which are deliberately excluded from the
first iteration).
"""

from dataclasses import dataclass
from typing import Literal


ParamType = Literal["int", "float", "categorical"]


@dataclass(frozen=True)
class Param:
    """One tunable parameter in the FETM search space."""

    name: str                       # dot-path into the config dict
    type: ParamType
    low: float | int | None = None  # used for int/float
    high: float | int | None = None
    log: bool = False               # sample on log scale (float only)
    choices: tuple | None = None    # used for categorical
    description: str = ""


SEARCH_SPACE: tuple[Param, ...] = (
    # --- Volatility estimators ---
    Param(
        "volatility.ewma.halflife", "int", low=20, high=180,
        description="EWMA half-life in days (shorter = more reactive).",
    ),
    Param(
        "volatility.parkinson.window", "int", low=5, high=60,
        description="Rolling window (days) for Parkinson range estimator.",
    ),
    Param(
        "volatility.exit_time.corridor_width", "float",
        low=0.005, high=0.03, log=True,
        description="Log-price corridor width for the first-exit-time estimator.",
    ),
    Param(
        "volatility.exit_time.buffer_size", "int", low=10, high=60,
        description="Number of recent exit times averaged into the FET estimate.",
    ),
    Param(
        "volatility.composite.blend_steepness", "float", low=1.0, high=10.0,
        description="Sigmoid steepness blending FET vs EWMA vol.",
    ),
    Param(
        "volatility.composite.max_fet_weight", "float", low=0.3, high=0.95,
        description="Maximum weight the composite can put on the FET estimator.",
    ),

    # --- Momentum signals ---
    Param(
        "momentum.scurve_lambda", "float", low=0.1, high=1.0,
        description="S-curve non-linearity for the non-linear momentum signal.",
    ),
    Param(
        "momentum.signal_cap", "float", low=2.0, high=8.0,
        description="Hard cap on the absolute normalized signal value.",
    ),

    # --- Portfolio / rebalance ---
    Param(
        "strategy.target_vol", "float", low=0.08, high=0.25,
        description="Target annualized portfolio volatility.",
    ),
    Param(
        "strategy.max_leverage", "float", low=1.0, high=4.0,
        description="Maximum absolute leverage per position.",
    ),
    Param(
        "strategy.rebalance_frequency", "int", low=1, high=21,
        description="Scheduled rebalance cadence in trading days.",
    ),
    Param(
        "strategy.rebalance_vol_threshold", "float", low=0.10, high=0.50,
        description="Force rebalance when vol jumps more than this fraction.",
    ),
    Param(
        "strategy.rebalance_signal_threshold", "float", low=0.5, high=3.0,
        description="Force rebalance when signal moves more than N std-devs.",
    ),
)


def suggest_params(trial, space: tuple[Param, ...] = SEARCH_SPACE) -> dict:
    """Draw one candidate parameter set from an Optuna trial.

    Returns a dict keyed by the parameter dot-path so it can be fed directly
    into :func:`fetm.utils.nested.set_nested`.
    """
    out: dict = {}
    for p in space:
        if p.type == "int":
            out[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high))
        elif p.type == "float":
            out[p.name] = trial.suggest_float(
                p.name, float(p.low), float(p.high), log=p.log
            )
        elif p.type == "categorical":
            out[p.name] = trial.suggest_categorical(p.name, list(p.choices or ()))
        else:  # pragma: no cover — defensive, dataclass restricts this
            raise ValueError(f"Unknown param type: {p.type}")
    return out
