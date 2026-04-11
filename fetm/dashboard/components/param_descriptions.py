"""Human-readable descriptions and grouping for FETM strategy parameters.

Used by the ``Strategy Parameters`` dashboard page to render grouped tables
with tooltips explaining what each value controls.
"""

from __future__ import annotations


# Ordered list of category (section title, [(dot_path, short_label, help), ...])
PARAM_GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "Volatility Estimators",
        [
            ("volatility.ewma.halflife", "EWMA half-life (days)",
             "Exponential decay half-life. Shorter = more reactive to recent vol."),
            ("volatility.ewma.min_history", "EWMA min history",
             "Minimum warm-up observations before EWMA is trusted."),
            ("volatility.parkinson.window", "Parkinson window",
             "Rolling window for the Parkinson high/low range estimator."),
            ("volatility.exit_time.corridor_width", "FET corridor width",
             "Log-price corridor (fraction). Smaller = more exits / noisier."),
            ("volatility.exit_time.buffer_size", "FET buffer size",
             "Recent exit times averaged into the FET estimate."),
            ("volatility.exit_time.max_no_exit_days", "FET max no-exit days",
             "Cap on how long we can go without a corridor exit."),
            ("volatility.exit_time.bias_correction", "FET bias correction",
             "Enable the finite-sample bias correction."),
            ("volatility.composite.blend_steepness", "Composite steepness",
             "Sigmoid steepness blending FET vs EWMA vol."),
            ("volatility.composite.max_fet_weight", "Composite max FET weight",
             "Upper bound on the weight the composite puts on FET vol."),
        ],
    ),
    (
        "Momentum Signals",
        [
            ("momentum.lookback_days", "Lookback horizons (days)",
             "Momentum horizons blended into the combined signal."),
            ("momentum.horizon_weights", "Horizon weights",
             "Weights applied to each lookback horizon."),
            ("momentum.scurve_lambda", "S-curve lambda",
             "Non-linearity knob for the non-linear momentum signal."),
            ("momentum.signal_normalization_window", "Norm. window (days)",
             "Rolling window used to z-score the raw signal."),
            ("momentum.signal_cap", "Signal cap",
             "Hard cap on the absolute normalized signal value."),
        ],
    ),
    (
        "Portfolio Construction",
        [
            ("strategy.name", "Strategy name", "Internal strategy identifier."),
            ("strategy.target_vol", "Target vol",
             "Per-position target annualized volatility."),
            ("strategy.portfolio_target_vol", "Portfolio target vol",
             "Overall portfolio-level volatility target."),
            ("strategy.max_leverage", "Max leverage",
             "Maximum absolute leverage per position."),
        ],
    ),
    (
        "Rebalancing",
        [
            ("strategy.rebalance_frequency", "Rebalance frequency (days)",
             "Scheduled rebalance cadence in trading days."),
            ("strategy.rebalance_vol_threshold", "Vol trigger",
             "Force rebalance when vol jumps more than this fraction."),
            ("strategy.rebalance_signal_threshold", "Signal trigger",
             "Force rebalance when signal moves more than N std-devs."),
        ],
    ),
    (
        "Transaction Costs (bps)",
        [
            ("costs.equity_etf_bps", "Equity ETF", "One-way cost in basis points."),
            ("costs.commodity_etf_bps", "Commodity ETF", "One-way cost in basis points."),
            ("costs.bond_etf_bps", "Bond ETF", "One-way cost in basis points."),
            ("costs.currency_etf_bps", "Currency ETF", "One-way cost in basis points."),
        ],
    ),
    (
        "Backtest",
        [
            ("backtest.start_date", "Start date", "Backtest start date."),
            ("backtest.warmup_days", "Warm-up (days)", "Warm-up before positions are taken."),
            ("backtest.in_sample_end", "In-sample end", "IS/OOS split date."),
            ("backtest.risk_free_rate", "Risk-free rate", "Annual risk-free rate used in Sharpe/Sortino."),
            ("backtest.walk_forward.initial_window_years", "WF initial window (yr)",
             "Length of the first walk-forward training window."),
            ("backtest.walk_forward.refit_frequency_years", "WF refit freq (yr)",
             "How often walk-forward refits parameters."),
        ],
    ),
]
