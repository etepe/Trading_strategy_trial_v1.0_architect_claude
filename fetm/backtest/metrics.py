"""Performance metrics calculation."""

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """Compute comprehensive performance metrics for a return series.

    Args:
        risk_free_rate: Annual risk-free rate (default 0.02).
    """

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        self.rf = risk_free_rate

    def compute(
        self, returns: pd.Series, positions: pd.Series | None = None
    ) -> dict:
        """Compute all performance metrics.

        Args:
            returns: Daily strategy returns (after costs).
            positions: Daily position series (optional, for turnover metrics).

        Returns:
            Dictionary of performance metrics.
        """
        returns = returns.dropna()
        if len(returns) < 2:
            return self._empty_metrics()

        n_years = len(returns) / 252.0
        daily_rf = self.rf / 252.0

        # Returns metrics
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        ann_vol = returns.std() * np.sqrt(252)
        excess_return = ann_return - self.rf
        sharpe = excess_return / ann_vol if ann_vol > 1e-8 else 0.0

        # Sortino ratio
        downside = returns[returns < daily_rf]
        downside_vol = np.sqrt((downside ** 2).mean()) * np.sqrt(252) if len(downside) > 0 else 1e-8
        sortino = excess_return / downside_vol if downside_vol > 1e-8 else 0.0

        # Drawdown analysis
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        drawdown = cum / peak - 1
        max_dd = drawdown.min()

        # Max drawdown duration
        dd_duration = self._max_dd_duration(drawdown)

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0

        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95

        # Distribution metrics
        skew = float(returns.skew())
        kurt = float(returns.kurtosis())

        # Monthly aggregation
        monthly = returns.resample("ME").sum()
        worst_month = float(monthly.min()) if len(monthly) > 0 else 0.0
        best_month = float(monthly.max()) if len(monthly) > 0 else 0.0

        quarterly = returns.resample("QE").sum()
        worst_quarter = float(quarterly.min()) if len(quarterly) > 0 else 0.0
        best_quarter = float(quarterly.max()) if len(quarterly) > 0 else 0.0

        # Win rate
        win_rate_daily = (returns > 0).mean()
        win_rate_monthly = (monthly > 0).mean() if len(monthly) > 0 else 0.0

        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 1e-8 else float("inf")

        # Turnover and holding metrics
        if positions is not None:
            positions = positions.dropna()
            annual_turnover = positions.diff().abs().sum() / (2 * n_years) if n_years > 0 else 0.0
            pct_long = (positions > 0.01).mean()
            pct_short = (positions < -0.01).mean()
            pct_flat = 1 - pct_long - pct_short

            # Average holding period (approx)
            position_changes = (positions.diff().abs() > 0.01).sum()
            avg_holding = len(positions) / max(position_changes, 1)
        else:
            annual_turnover = 0.0
            pct_long = 0.0
            pct_short = 0.0
            pct_flat = 1.0
            avg_holding = None

        return {
            "annualized_return": round(ann_return, 6),
            "annualized_vol": round(ann_vol, 6),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd, 6),
            "max_drawdown_duration_days": dd_duration,
            "calmar_ratio": round(calmar, 4),
            "var_95": round(var_95, 6),
            "var_99": round(var_99, 6),
            "cvar_95": round(cvar_95, 6),
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "worst_month": round(worst_month, 6),
            "best_month": round(best_month, 6),
            "worst_quarter": round(worst_quarter, 6),
            "best_quarter": round(best_quarter, 6),
            "annual_turnover": round(annual_turnover, 4),
            "avg_holding_period": round(avg_holding, 1) if avg_holding is not None else None,
            "pct_time_long": round(pct_long, 4),
            "pct_time_short": round(pct_short, 4),
            "pct_time_flat": round(pct_flat, 4),
            "win_rate_daily": round(win_rate_daily, 4),
            "win_rate_monthly": round(win_rate_monthly, 4),
            "profit_factor": round(profit_factor, 4),
        }

    def _max_dd_duration(self, drawdown: pd.Series) -> int:
        """Compute maximum drawdown duration in trading days."""
        in_dd = drawdown < -1e-6
        if not in_dd.any():
            return 0
        groups = (~in_dd).cumsum()
        dd_groups = groups[in_dd]
        if len(dd_groups) == 0:
            return 0
        return int(dd_groups.value_counts().max())

    def _empty_metrics(self) -> dict:
        return {k: 0.0 for k in [
            "annualized_return", "annualized_vol", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "max_drawdown_duration_days",
            "calmar_ratio", "var_95", "var_99", "cvar_95",
            "skewness", "kurtosis", "worst_month", "best_month",
            "worst_quarter", "best_quarter", "annual_turnover",
            "avg_holding_period", "pct_time_long", "pct_time_short",
            "pct_time_flat", "win_rate_daily", "win_rate_monthly",
            "profit_factor",
        ]}

    def compute_crisis_performance(
        self,
        returns_dict: dict[str, pd.Series],
        crisis_periods: dict[str, dict],
    ) -> dict:
        """Compute returns during each crisis period for each strategy.

        Args:
            returns_dict: {strategy_name: returns_series}.
            crisis_periods: {crisis_id: {name, start, end}}.

        Returns:
            Nested dict: {crisis_id: {strategy: total_return}}.
        """
        result = {}
        for crisis_id, period in crisis_periods.items():
            start = pd.Timestamp(period["start"])
            end = pd.Timestamp(period["end"])
            crisis_returns = {}
            for strat, rets in returns_dict.items():
                mask = (rets.index >= start) & (rets.index <= end)
                period_rets = rets[mask]
                if len(period_rets) > 0:
                    total = (1 + period_rets).prod() - 1
                    crisis_returns[strat] = round(float(total), 6)
                else:
                    crisis_returns[strat] = None
            result[crisis_id] = crisis_returns
        return result

    def compute_conditional_performance(
        self,
        returns_dict: dict[str, pd.Series],
        benchmark_returns: pd.Series,
    ) -> dict:
        """Compute Sharpe ratios conditional on benchmark return terciles.

        Args:
            returns_dict: {strategy_name: returns_series}.
            benchmark_returns: Benchmark (buy-and-hold) returns.

        Returns:
            Dict: {tercile: {strategy: sharpe}}.
        """
        # Monthly returns
        monthly_bench = benchmark_returns.resample("ME").sum()
        terciles = pd.qcut(monthly_bench, 3, labels=["bottom_tercile", "middle_tercile", "top_tercile"])

        result = {}
        for label in ["bottom_tercile", "middle_tercile", "top_tercile"]:
            months = terciles[terciles == label].index
            strat_sharpes = {}
            for strat, rets in returns_dict.items():
                monthly_strat = rets.resample("ME").sum()
                matched = monthly_strat.reindex(months).dropna()
                if len(matched) > 1:
                    ann_ret = matched.mean() * 12
                    ann_vol = matched.std() * np.sqrt(12)
                    sr = (ann_ret - self.rf) / ann_vol if ann_vol > 1e-8 else 0.0
                    strat_sharpes[strat] = round(sr, 4)
                else:
                    strat_sharpes[strat] = None
            result[label] = strat_sharpes
        return result
