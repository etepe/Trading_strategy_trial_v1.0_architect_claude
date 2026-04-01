# FETM Strategy: First Exit-Time Momentum
## Complete Implementation Specification for Claude Code

**Version:** 1.0
**Date:** March 2026
**Purpose:** This document is a complete, self-contained specification for building and backtesting a systematic multi-asset trading strategy. Implement it phase by phase. Each phase should be a working, testable system before proceeding to the next.

---

## Executive Summary

FETM combines two academic innovations into a unified trading strategy:

1. **First Exit-Time Volatility Estimation** (Merrill & Sinclair 2014): estimate volatility from the time it takes price to exit a corridor, rather than from fixed-interval returns. This converges faster during regime transitions.

2. **Nonlinear Time Series Momentum** (Moskowitz, Sabbatucci, Tamoni & Uhl 2025): apply an S-curve transformation to momentum signals, attenuating extreme signals. This improves Sharpe ratios by ~20% and dramatically improves tail-hedging during market crashes.

The novel contribution is their combination: using exit-time volatility to normalize the nonlinear momentum signal, creating a strategy that adapts faster to regime changes and generates superior risk-adjusted returns, particularly during extreme market environments.

---

## PHASE 1: Foundation — Single Asset MVP
**Goal:** Prove the core signal works on SPY with daily data.
**Deliverables:** Working backtest, performance report, signal diagnostics.

### 1.1 Data Pipeline

```
Data source: yfinance (for Phase 1 only — free, no API key needed)
Ticker: SPY
Fields: Date, Open, High, Low, Close, Adj Close, Volume
History: Maximum available (SPY inception: January 1993)
Frequency: Daily
Storage: Parquet files in data/raw/ directory
```

**Task:** Create a `data/` module with:
- `download.py`: fetch and store raw OHLCV data
- `clean.py`: handle splits, dividends, missing days, ensure Adj Close is used for all return calculations
- `features.py`: compute derived features (returns, log returns, rolling stats)

### 1.2 Volatility Estimators — Build Three, Compare

Implement all three estimators so we can compare throughout the backtest:

**Estimator A: EWMA Volatility (Baseline)**
```
σ_EWMA(t) = sqrt( λ * σ²_EWMA(t-1) + (1-λ) * r²(t) )
where λ = exp(-1/60) ≈ 0.9835 (60-day half-life)
r(t) = log(Close(t) / Close(t-1))
Initialization: first 60 days use sample std of log returns
Annualization: multiply by sqrt(252)
```

**Estimator B: Parkinson Range-Based Volatility**
```
σ_Park(t) = sqrt( (1/n) * Σ [log(H(i)/L(i))]² / (4*log(2)) )
Use rolling window n = 20 trading days
Annualization: multiply by sqrt(252)
```

**Estimator C: First Exit-Time Volatility (Core Innovation)**

Since we only have daily OHLC for the full history, we simulate exit times from daily bars:

```
Parameters:
  Δ = corridor half-width (start with 0.01 = 1% log-price corridor)
  n_buffer = 30 (number of recent exit times to average)

Algorithm:
  1. Set reference_price = Close(0)
  2. Set upper_barrier = reference_price * exp(+Δ)
  3. Set lower_barrier = reference_price * exp(-Δ)
  4. For each day t:
     a. Check if High(t) >= upper_barrier OR Low(t) <= lower_barrier
     b. If YES (barrier breached):
        - Record exit_time τ = (t - t_last_reset) / 252  [in years]
        - Estimate intra-day exit time:
          If High(t) >= upper_barrier AND Low(t) <= lower_barrier:
            # Both barriers hit same day — use midpoint approximation
            τ_intraday = 0.5 / 252
          Else:
            # Linear interpolation within the day
            If High(t) >= upper_barrier:
              penetration = (High(t) - upper_barrier) / (High(t) - Open(t))
              τ_intraday = (1 - penetration * 0.5) / 252
            Else:
              penetration = (lower_barrier - Low(t)) / (Open(t) - Low(t))
              τ_intraday = (1 - penetration * 0.5) / 252
        - Adjust: τ = τ - (1/252) + τ_intraday  [replace full last day with intra-day estimate]
        - Append τ to exit_times buffer
        - Reset reference_price = Close(t)
        - Reset barriers around new reference_price
     c. If NO (still inside corridor):
        - Continue accumulating time
  5. σ_FET(t) = Δ / sqrt(mean(last n_buffer exit_times))
  6. Apply Jensen bias correction: σ_FET(t) *= n_buffer / (n_buffer + 0.25)
  7. Annualization: already in annual terms if τ is in years

Edge cases:
  - Before first n_buffer exits: use EWMA as fallback
  - If no exit for 60+ days: force corridor reset, widen Δ by 50% temporarily
  - Corridor width Δ should be calibrated (see 1.5)
```

**Estimator D: Composite Volatility (Blended)**
```
σ_composite(t) = w(t) * σ_FET(t) + (1 - w(t)) * σ_EWMA(t)

where w(t) = sigmoid(k * |σ_FET(t) - σ_EWMA(t)| / σ_EWMA(t))
  k = 5 (steepness parameter)
  sigmoid(x) = 0.8 / (1 + exp(-x + 2))

Intuition: when FET and EWMA agree, lean on EWMA (smoother).
When they diverge (regime transition), lean on FET (faster).
Maximum FET weight capped at 0.8 (never fully abandon EWMA).
```

**Task:** Create `signals/volatility.py` with classes for each estimator. Each must expose:
- `update(ohlcv_row)` method for streaming updates
- `estimate(t)` method returning annualized vol at time t
- `history` property returning full time series
- Unit tests comparing estimators on known volatility paths

### 1.3 Momentum Signal Construction

**Step 1: Compute normalized returns at three horizons**
```
For lookback h in {21, 63, 252} trading days:
  r_h(t) = log(Close(t) / Close(t-h))
  s_h(t) = r_h(t) / σ_composite(t)   [volatility-normalized signal]
```

**Step 2: Apply S-curve nonlinear transformation**
```
f(s) = λ_h * s / (s² + 1)

where λ_h is calibrated per horizon to normalize variance:
  λ_h = 1 / std(f(s_h)) computed over a 5-year expanding window
  
  Alternatively, use the Moskowitz et al. (2025) constant: λ ≈ 0.394
  (calibrated so var(f(s)) ≈ 1 for standard normal s)
  
  Start with λ = 0.394 for all horizons. Optimize in Phase 3.
```

**Step 3: Combine horizons**
```
Signal(t) = α₁ * f(s_21(t)) + α₂ * f(s_63(t)) + α₃ * f(s_252(t))

Baseline weights: α₁ = 0.25, α₂ = 0.40, α₃ = 0.35
(Heavier on 3M based on NLTSMOM paper's strongest results)

Normalize: Signal(t) = Signal(t) / rolling_std(Signal, 252)
```

**Task:** Create `signals/momentum.py` with:
- `LinearMomentum`: raw normalized returns (baseline)
- `BinaryMomentum`: sign of past returns (Moskowitz 2012 baseline)
- `NonlinearMomentum`: S-curve transformed (our strategy)
- Each class computes signals at all three horizons and combines them
- Expose both individual horizon signals and combined signal

### 1.4 Position Sizing and Portfolio Construction

```
Position(t) = Signal(t) * (TargetVol / σ_composite(t))

Parameters:
  TargetVol = 0.15 (15% annualized target volatility)
  MaxLeverage = 3.0 (cap position at 3x)
  MinPosition = 0.0 (no minimum — can be flat)
  
Position is continuous: ranges from -MaxLeverage to +MaxLeverage
Negative = short, Positive = long
```

**Transaction costs:**
```
Cost per trade = |ΔPosition(t)| * cost_bps
cost_bps = 5 basis points (0.0005) for SPY
  (conservative: includes half-spread + market impact for $50M+ portfolio)

Net return(t) = Position(t-1) * r(t) - |Position(t) - Position(t-1)| * cost_bps
```

**Rebalancing rule:**
```
Rebalance weekly (every 5 trading days) UNLESS:
  |σ_FET(t) / σ_FET(t-1) - 1| > 0.20  →  force daily rebalance (vol spike)
  |Signal(t) - Signal(t-5)| > 1.5 * std(ΔSignal)  →  force rebalance (signal jump)
```

**Task:** Create `portfolio/construction.py` with:
- `PositionSizer` class that takes signal + vol estimate → position
- `TransactionCostModel` class
- `RebalanceScheduler` class implementing the adaptive rule

### 1.5 Backtest Engine

```
Backtest period: 1994-01-01 to present
  (start after 252 days of warmup for 12M lookback)
In-sample: 1994-2014 (parameter calibration)
Out-of-sample: 2015-present (performance evaluation)

Walk-forward validation:
  Initial training window: 5 years
  Refit every: 1 year
  Parameters refitted: corridor width Δ, horizon weights α, λ scaling
  Parameters FIXED: S-curve functional form, rebalance logic, cost model
```

**Performance metrics to compute:**
```
Returns:
  - Annualized return
  - Annualized volatility (realized)
  - Sharpe ratio (excess return over risk-free rate)
  - Sortino ratio
  - Maximum drawdown (peak-to-trough)
  - Maximum drawdown duration (days)
  - Calmar ratio (return / max drawdown)

Risk:
  - 95% and 99% Value-at-Risk (historical)
  - Expected Shortfall (CVaR) at 95%
  - Skewness of returns
  - Kurtosis of returns
  - Worst month, worst quarter

Efficiency:
  - Annual turnover (sum of |ΔPosition| / 2)
  - Average holding period
  - Percentage of time long / short / flat
  - Win rate (% of positive return periods)
  - Profit factor (gross gains / gross losses)

Comparison:
  - All metrics computed for FOUR strategies simultaneously:
    1. Buy-and-hold SPY (benchmark)
    2. Linear TSMOM with EWMA vol (Moskowitz 2012 replication)
    3. Nonlinear TSMOM with EWMA vol (S-curve only improvement)
    4. Nonlinear TSMOM with FET composite vol (FETM — our strategy)
```

**Corridor width calibration (Δ):**
```
Test Δ ∈ {0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03}
For each Δ, run full backtest on in-sample period.
Select Δ that maximizes Sharpe ratio.
Report sensitivity: Sharpe ratio vs Δ plot.
Expected optimal range: 0.01–0.02 for SPY (1–2% corridor).
```

**Task:** Create `backtest/` module with:
- `engine.py`: event-driven backtest loop
- `metrics.py`: all performance calculations
- `report.py`: generate summary tables and plots
- `walk_forward.py`: walk-forward validation framework

### 1.6 Visualization and Diagnostics

Generate the following plots:

```
1. Equity curves: all four strategies on same chart (log scale)
2. Drawdown chart: underwater curves for all four strategies
3. Rolling 1-year Sharpe ratio: all four strategies
4. Volatility comparison: σ_EWMA vs σ_FET vs σ_composite over time
   - Highlight periods where they diverge (regime transitions)
5. Signal diagnostics:
   a. Scatter plot: raw signal s vs transformed signal f(s), colored by horizon
   b. Time series of combined Signal(t) with position overlay
   c. Histogram of signal values with S-curve overlay
6. Exit-time diagnostics:
   a. Time series of average exit time τ̄
   b. Distribution of exit times (histogram + fitted inverse Gaussian)
   c. Corridor width sensitivity: Sharpe vs Δ
7. Monthly returns heatmap (year × month)
8. Crisis performance table:
   - 2000-2002 dot-com bust
   - 2008-2009 GFC
   - 2015 Aug flash crash
   - 2018 Q4 selloff
   - 2020 Feb-Mar COVID
   - 2022 rate hiking cycle
   - 2025 tariff/Iran crisis
9. Conditional performance:
   - Sharpe ratio in top/middle/bottom tercile of SPY returns
   - Matches Table 8 from NLTSMOM paper
```

**Task:** Create `visualization/` module with functions for each plot. Use matplotlib with a clean, professional style (dark background optional). All plots must be saved as PNG files.

### 1.7 Phase 1 Success Criteria

Before moving to Phase 2, verify:
```
□ FETM Sharpe ratio > Linear TSMOM Sharpe ratio (out-of-sample)
□ FETM max drawdown < Buy-and-hold max drawdown
□ FET volatility detects regime changes faster than EWMA
  (measure: average lag in days to detect a >50% vol increase)
□ S-curve reduces turnover vs binary signal by >10%
□ Crisis alpha: FETM returns positive in at least 4 of 7 crisis periods
□ All four strategies produce plausible results (no bugs)
```

---

## PHASE 2: Multi-Asset Expansion
**Goal:** Extend to a diversified futures proxy portfolio using liquid ETFs.
**Prerequisite:** Phase 1 complete and passing success criteria.

### 2.1 Asset Universe

Use ETFs as liquid proxies for the futures universe:

```
Equity Indices:
  SPY   — S&P 500
  EFA   — MSCI EAFE (developed ex-US)
  EEM   — MSCI Emerging Markets
  EWJ   — MSCI Japan

Commodities:
  GLD   — Gold
  SLV   — Silver
  USO   — Crude Oil (WTI)
  DBA   — Agriculture
  DBB   — Base Metals

Fixed Income / Rates:
  TLT   — US 20+ Year Treasury
  IEF   — US 7-10 Year Treasury
  TIP   — TIPS (inflation-linked)
  BWX   — International Treasury

Currencies (via ETFs):
  UUP   — US Dollar Index (long USD)
  FXE   — Euro
  FXY   — Japanese Yen
  FXA   — Australian Dollar

Total: 17 assets
Data start: use earliest common date (likely ~2006 for most)
```

**Transaction costs by asset class:**
```
Equity ETFs:    5 bps
Commodity ETFs: 10 bps (wider spreads, tracking error)
Bond ETFs:      3 bps
Currency ETFs:  8 bps
```

### 2.2 Per-Asset Signal Construction

Apply the Phase 1 signal pipeline independently to each asset:

```
For each asset i:
  1. Compute σ_composite,i(t) using asset-specific EWMA and FET estimators
  2. Compute signals s_h,i(t) for h ∈ {21, 63, 252}
  3. Apply S-curve: f(s_h,i(t))
  4. Combine horizons: Signal_i(t) = Σ α_h * f(s_h,i(t))
  
  Corridor width Δ_i calibrated per asset class:
    Equity ETFs:    Δ = 0.010 – 0.020
    Commodity ETFs: Δ = 0.015 – 0.030 (higher vol, wider corridors)
    Bond ETFs:      Δ = 0.005 – 0.010 (lower vol, tighter corridors)
    Currency ETFs:  Δ = 0.008 – 0.015
  
  Optimize per-class (not per-asset) to reduce overfitting.
```

### 2.3 Portfolio Construction

**Equal risk contribution (volatility parity):**
```
Position_i(t) = Signal_i(t) * (TargetVol_i / σ_composite,i(t))

where TargetVol_i = PortfolioTargetVol / sqrt(N_assets)
  PortfolioTargetVol = 0.12 (12% annualized)
  N_assets = 17

This ensures each asset contributes roughly equal risk.
```

**Portfolio-level risk overlay:**
```
Realized portfolio vol = std(portfolio returns, 20-day rolling)
If realized_vol > 1.5 * PortfolioTargetVol:
  Scale ALL positions by (PortfolioTargetVol / realized_vol)
  
This is a circuit breaker for correlated drawdowns.
```

**Correlation-aware adjustment (optional enhancement):**
```
Compute rolling 60-day correlation matrix C(t)
Effective diversification = 1 / sqrt(avg pairwise correlation)
Adjust TargetVol_i by diversification factor

This prevents over-leveraging when correlations spike (crisis periods).
```

### 2.4 Performance Analysis

Compute all Phase 1 metrics PLUS:
```
Portfolio-level:
  - Asset class contribution to return (equity, commodity, rates, FX)
  - Correlation of FETM with each asset class buy-and-hold
  - Rolling correlation between FETM and SPY (tail hedge test)

Benchmarks (add to comparison):
  5. Equal-weight buy-and-hold of all 17 ETFs
  6. SG Trend Index (if available) or CTA proxy
  7. 60/40 portfolio (SPY/TLT)
  
Cross-asset signal analysis:
  - Do exit-time vol estimates diverge across asset classes before crises?
  - Which asset classes benefit most from FET vs EWMA?
  - Signal correlation matrix: are momentum signals diversifying?
```

### 2.5 Phase 2 Success Criteria
```
□ Portfolio Sharpe > 0.8 (out-of-sample, after costs)
□ Maximum drawdown < 20%
□ Positive returns in at least 3 of 4 asset classes individually
□ Portfolio correlation with SPY < 0.3
□ FET volatility adds value in at least 3 of 4 asset classes
□ Performance improvement over equal-weight benchmark > 30% (Sharpe ratio)
```

---

## PHASE 3: Bloomberg Futures Universe
**Goal:** Move from ETF proxies to actual futures data via Bloomberg.
**Prerequisite:** Phase 2 complete. Bloomberg API access configured.

### 3.1 Bloomberg Data Pipeline

```
Futures contracts (continuous, ratio-adjusted):
  Use Bloomberg generic tickers (e.g., ES1 Index, CL1 Comdty)
  
Fields to pull:
  PX_OPEN, PX_HIGH, PX_LOW, PX_LAST (daily OHLC)
  VOLUME, OPEN_INT (for liquidity filtering)
  FUT_CUR_GEN_TICKER (for roll tracking)

Instrument universe (from NLTSMOM paper Table 1):

Equity Index Futures (8):
  ES1 Index   — S&P 500 E-mini
  CF1 Index   — CAC 40
  GX1 Index   — DAX
  Z 1 Index   — FTSE 100
  TP1 Index   — TOPIX
  XP1 Index   — S&P/ASX 200
  ST1 Index   — FTSE MIB
  IB1 Index   — IBEX 35

Commodity Futures (24):
  CL1 Comdty  — WTI Crude Oil
  CO1 Comdty  — Brent Crude
  NG1 Comdty  — Natural Gas
  HO1 Comdty  — Heating Oil (ULSD)
  XB1 Comdty  — RBOB Gasoline
  GC1 Comdty  — Gold
  SI1 Comdty  — Silver
  LA1 Comdty  — Aluminium (LME)
  LP1 Comdty  — Copper (LME)
  LN1 Comdty  — Nickel (LME)
  LX1 Comdty  — Zinc (LME)
  C 1 Comdty  — Corn
  S 1 Comdty  — Soybeans
  W 1 Comdty  — Wheat
  SM1 Comdty  — Soybean Meal
  BO1 Comdty  — Soybean Oil
  KC1 Comdty  — Coffee
  CC1 Comdty  — Cocoa
  CT1 Comdty  — Cotton
  SB1 Comdty  — Sugar
  LH1 Comdty  — Lean Hogs
  LC1 Comdty  — Live Cattle
  JA1 Comdty  — Platinum
  QS1 Comdty  — Gasoil

Rates Futures (13):
  TU1 Comdty  — US 2-Year
  FV1 Comdty  — US 5-Year
  TY1 Comdty  — US 10-Year
  US1 Comdty  — US Long Bond
  RX1 Comdty  — Bund
  OE1 Comdty  — Bobl
  DU1 Comdty  — Schatz
  UB1 Comdty  — Buxl
  G 1 Comdty  — UK Gilt
  JB1 Comdty  — Japan 10-Year
  CN1 Comdty  — Canada 10-Year
  YM1 Comdty  — Australia 3-Year
  XM1 Comdty  — Australia 10-Year

Currency Futures (9):
  EC1 Curncy  — EUR/USD
  JY1 Curncy  — JPY/USD
  BP1 Curncy  — GBP/USD
  SF1 Curncy  — CHF/USD
  AD1 Curncy  — AUD/USD
  CD1 Curncy  — CAD/USD
  NO1 Curncy  — NOK/USD
  SE1 Curncy  — SEK/USD
  NV1 Curncy  — NZD/USD

Total: 54 instruments
History: maximum available (some back to 1980, most to 1990+)
```

**Bloomberg API data pull script structure:**
```python
# Pseudocode for Bloomberg data pipeline
# Use blpapi Python SDK

tickers = [list above]
fields = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "VOLUME", "OPEN_INT"]
start_date = "19800101"
end_date = "today"

# Pull via bdh (historical data)
# Store as parquet files: data/futures/{ticker}.parquet
# Include metadata: contract size, tick size, currency, sector

# Roll adjustment:
# Bloomberg generic tickers already provide continuous series
# Verify: check for jumps > 3σ on roll dates and flag them
```

### 3.2 Futures-Specific Adjustments

**Roll return handling:**
```
Bloomberg generic tickers use ratio adjustment (same as NLTSMOM paper eq. 2).
Log returns on roll dates: r(t) = log(P_near(t)) - log(P_near(t-1))
No price jump from rolls. Carry/roll return is implicitly earned.
Verify this is the case for each pulled series.
```

**Position sizing in dollar terms:**
```
For futures, position sizing is in contracts:

N_contracts_i(t) = Signal_i(t) * (PortfolioNAV * TargetVol_i) / (σ_composite,i(t) * ContractValue_i(t))

where ContractValue_i = Price * Multiplier
  e.g., ES: $50 * index level ≈ $250,000 per contract
  
Round to integer contracts. Minimum: 1 contract (or skip asset).
```

**Margin and capital efficiency:**
```
Track notional exposure vs margin requirement.
Typical initial margins: 5-15% of notional.
Cap total notional at 5x portfolio NAV.
```

### 3.3 Sector-Specific Corridor Calibration

```
Optimize Δ per sector (not per instrument) on in-sample data:

Equity Indices:   Δ ∈ {0.008, 0.010, 0.012, 0.015}
Energies:         Δ ∈ {0.015, 0.020, 0.025, 0.030}
Metals:           Δ ∈ {0.010, 0.015, 0.020, 0.025}
Agriculturals:    Δ ∈ {0.012, 0.015, 0.020, 0.025}
Rates:            Δ ∈ {0.003, 0.005, 0.007, 0.010}
Currencies:       Δ ∈ {0.005, 0.008, 0.010, 0.012}

Selection criterion: maximize in-sample Sharpe ratio of sector portfolio.
Robustness check: performance should be stable across adjacent Δ values.
```

### 3.4 NLTSMOM Paper Replication

**Critical validation step:** Before trusting our framework, replicate key results from the NLTSMOM paper.

```
Replicate Table 3 (daily strategy results):
  - Linear TSMOM: target SR ≈ 0.70 (12M lookback)
  - Nonlinear (F&S): target SR ≈ 0.83
  - Empirical nonlinear: target SR ≈ 0.84
  - Binary: target SR ≈ 0.88
  
  Our results should be within ±0.10 of these benchmarks.
  
Replicate Table 8 (conditional performance):
  - Sort months by SPY return into terciles
  - Report Sharpe by tercile for each strategy
  - Key result: nonlinear strategies outperform most in bottom tercile

Replicate Figure 2 (estimated nonlinear function):
  - Plot f(s) vs s for equity futures
  - Should show S-curve shape with rollback at extremes
```

### 3.5 Phase 3 Success Criteria
```
□ Successfully replicate NLTSMOM paper results within ±0.10 SR
□ FET volatility improves Sharpe over EWMA in at least 3 of 6 sectors
□ Portfolio Sharpe > 0.9 (out-of-sample, after futures transaction costs)
□ Maximum drawdown < 25%
□ Positive alpha (CAPM) in unconditional and down-market states
□ Signal works across daily, weekly, monthly frequencies
```

---

## PHASE 4: Intraday Validation & Exit-Time Refinement
**Goal:** Validate exit-time estimator with actual intraday data (6-month window).
**Prerequisite:** Phase 3 complete.

### 4.1 Intraday Data Pull

```
From Bloomberg, pull hourly bars for all 54 instruments:
  Fields: PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, VOLUME
  Frequency: 60-minute bars
  History: maximum available (~6 months)
  
Store in: data/intraday/{ticker}_hourly.parquet
```

### 4.2 True Exit-Time Computation

```
With hourly data, compute actual barrier crossing times:
  - Use same corridor widths Δ as Phase 3
  - Record exact crossing time (to the hour)
  - Compare: σ_FET from hourly data vs σ_FET approximated from daily OHLC

Validation metrics:
  - Correlation between daily-approximated and hourly-actual σ_FET
  - Mean absolute error
  - Does daily approximation capture regime transitions with similar lag?
  - Distribution of exit times: does it match the theoretical inverse Gaussian?
```

### 4.3 Adaptive Corridor Width

```
Innovation: make Δ adaptive based on recent volatility.

Δ(t) = k * σ_EWMA(t) * sqrt(target_exit_frequency / 252)

where:
  k = scaling constant (calibrate)
  target_exit_frequency = desired exits per day (e.g., 1-3 for hourly data)

This ensures roughly constant information flow regardless of volatility regime.
When vol doubles, corridor widens to maintain signal quality.
When vol halves, corridor narrows to capture more granular information.
```

### 4.4 Phase 4 Success Criteria
```
□ Daily OHLC approximation correlates > 0.85 with hourly true exit-time vol
□ Adaptive corridor produces more stable exit frequency across regimes
□ Intraday validation confirms no systematic bias in daily approximation
□ Document which asset classes benefit most from intraday data
```

---

## PHASE 5: Advanced Enhancements
**Goal:** Layer in ML-driven improvements and cross-asset features.
**Prerequisite:** Phase 4 complete. Core strategy validated.

### 5.1 Volatility Regime Indicator

```
Construct a composite regime indicator from exit-time statistics:

Regime(t) = {
  "crisis":     if σ_FET(t) > 2 * median(σ_FET) AND dσ_FET/dt > 0
  "high_vol":   if σ_FET(t) > 1.5 * median(σ_FET)
  "normal":     if 0.7 * median(σ_FET) < σ_FET(t) < 1.5 * median(σ_FET)
  "low_vol":    if σ_FET(t) < 0.7 * median(σ_FET)
}

In each regime, adapt signal weights:
  crisis:   α = (0.50, 0.35, 0.15)  — heavily favor fast signal
  high_vol: α = (0.35, 0.40, 0.25)  — moderate fast tilt
  normal:   α = (0.25, 0.40, 0.35)  — balanced (baseline)
  low_vol:  α = (0.15, 0.35, 0.50)  — favor slow signal
```

### 5.2 Cross-Asset Lead-Lag Signals

```
Compute rolling 20-day cross-correlation between exit-time vol
estimates across asset classes. When equity vol leads commodity vol
(or vice versa), this provides a few-day head start for
portfolio adjustments.

Lead-lag matrix L(t):
  L_ij(t) = argmax_τ corr(σ_FET,i(t), σ_FET,j(t-τ)) for τ ∈ {-5,...,+5}

If equity vol is leading (L_equity,commodity > 0), pre-emptively
adjust commodity positions based on equity regime.
```

### 5.3 S-Curve Parameter Learning

```
Instead of fixed λ = 0.394, learn the S-curve shape per sector and horizon
using a lightweight neural network:

Input:  s_h(t) — the normalized momentum signal
Output: f(s_h(t)) — optimal position weight

Architecture:
  - 1 hidden layer, 8 nodes, tanh activation
  - Enforce symmetry: f(-s) = -f(s) via data augmentation
  - Enforce f(0) = 0 via zero bias in output layer
  - Loss function: negative Sharpe ratio of the strategy

Train on expanding window (first 60% of data).
Validate on next 20%.
Test on final 20%.

Compare learned function to theoretical S-curve.
If they match closely → validates theory, use parametric.
If they diverge → potential improvement, use learned function with regularization.
```

### 5.4 Individual Equity Extension

```
Apply FETM to a universe of 50-100 liquid US equities:
  - Russell 1000 constituents with >$5B market cap
  - Minimum 5 years of history
  - Minimum average daily volume > $50M

Additional filters (from Nagel 2001/2005):
  - Compute B/M ratio; adjust momentum signal for B/M
  - Compute residual institutional ownership (orthogonalized to size)
  - Avoid shorting stocks with low institutional ownership
    (short-sale constraint proxy: IO < 30th percentile → no short)

Individual equity-specific adjustments:
  - Wider corridors: Δ = 0.02 – 0.04 (higher idiosyncratic vol)
  - Earnings announcement filter: flatten position 2 days before earnings
  - Sector neutrality: optional overlay to zero out sector bets
```

### 5.5 Phase 5 Success Criteria
```
□ Regime-adaptive weights improve Sharpe by >5% over fixed weights
□ Cross-asset lead-lag signals provide informational content (positive IC)
□ Neural network recovers S-curve shape consistent with theory
□ Equity extension produces Sharpe > 0.5 after transaction costs
□ Combined portfolio (futures + equities) Sharpe > 1.0
```

---

## Project Structure

```
fetm-strategy/
├── README.md                           ← project overview + quickstart
├── FETM_Strategy_Specification.md      ← this spec document
├── FETM_Dashboard_Specification.md     ← dashboard design spec
├── config/
│   ├── settings.yaml                   ← all parameters in one place
│   └── instruments.yaml                ← asset universe definitions
├── fetm/                               ← main Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py                 ← data fetching (yfinance Phase 1-2, Bloomberg Phase 3+)
│   │   ├── __main__.py                 ← CLI entry point: python -m fetm.data.download
│   │   ├── clean.py                    ← data cleaning and validation
│   │   ├── features.py                 ← derived features
│   │   └── bloomberg.py                ← Bloomberg API wrapper (Phase 3+)
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── volatility.py               ← EWMA, Parkinson, FET, Composite estimators
│   │   ├── momentum.py                 ← Linear, Binary, Nonlinear signal generators
│   │   └── regime.py                   ← Volatility regime classification (Phase 5)
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── construction.py             ← Position sizing, risk targeting
│   │   ├── rebalance.py                ← Adaptive rebalancing logic
│   │   └── costs.py                    ← Transaction cost models
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py                   ← Core backtest loop
│   │   ├── __main__.py                 ← CLI entry point: python -m fetm.backtest.engine
│   │   ├── metrics.py                  ← Performance calculations
│   │   ├── walk_forward.py             ← Walk-forward validation
│   │   ├── sensitivity.py              ← Parameter sensitivity analysis
│   │   └── report.py                   ← Generate static reports + exports
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py                      ← Streamlit main app entry point
│   │   ├── __main__.py                 ← CLI: python -m fetm.dashboard.app
│   │   ├── pages/
│   │   │   ├── 01_overview.py          ← KPI cards + equity curves
│   │   │   ├── 02_strategies.py        ← 4-strategy comparison
│   │   │   ├── 03_signals.py           ← Signal & S-curve diagnostics
│   │   │   ├── 04_volatility.py        ← Vol estimator analysis
│   │   │   ├── 05_crisis.py            ← Crisis period drill-down
│   │   │   ├── 06_assets.py            ← Per-asset breakdown (Phase 2+)
│   │   │   └── 07_parameters.py        ← Interactive parameter tuning
│   │   ├── components/
│   │   │   ├── charts.py               ← Plotly chart builders
│   │   │   ├── tables.py               ← Styled metric tables
│   │   │   ├── filters.py              ← Date range, asset, strategy selectors
│   │   │   └── kpi_cards.py            ← KPI card components
│   │   └── utils/
│   │       ├── data_loader.py          ← Load backtest results from parquet/json
│   │       ├── formatters.py           ← Number formatting, color coding
│   │       └── theme.py                ← Dark/light theme configuration
│   └── visualization/
│       ├── __init__.py
│       ├── equity_curves.py
│       ├── signal_diagnostics.py
│       ├── volatility_comparison.py
│       ├── crisis_analysis.py
│       └── style.py                    ← Consistent plot styling (static matplotlib)
├── notebooks/                           ← Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_volatility_estimators.ipynb
│   ├── 03_signal_analysis.ipynb
│   ├── 04_backtest_results.ipynb
│   └── 05_nltsmom_replication.ipynb
├── tests/
│   ├── test_volatility.py
│   ├── test_momentum.py
│   ├── test_backtest.py
│   └── test_data.py
├── data/                                ← data storage (gitignored)
│   ├── raw/
│   ├── processed/
│   └── futures/
├── output/                              ← backtest outputs (gitignored)
│   ├── runs/                            ← timestamped run directories
│   └── logs/
└── requirements.txt
```

## Configuration File (config/settings.yaml)

```yaml
# Master configuration — all tunable parameters

strategy:
  name: "FETM"
  target_vol: 0.15          # Phase 1 single-asset
  portfolio_target_vol: 0.12 # Phase 2+ multi-asset
  max_leverage: 3.0
  rebalance_frequency: 5     # trading days (weekly)
  rebalance_vol_threshold: 0.20  # force rebalance if vol changes >20%

volatility:
  ewma:
    halflife: 60             # days
    min_history: 60          # days before first estimate
  parkinson:
    window: 20               # days
  exit_time:
    corridor_width:          # Δ per asset class
      equity_index: 0.012
      energy: 0.020
      metals: 0.015
      agriculture: 0.018
      rates: 0.006
      currencies: 0.008
      equity_single: 0.025
    buffer_size: 30          # number of exit times to average
    max_no_exit_days: 60     # force reset if no exit
    bias_correction: true
  composite:
    blend_steepness: 5.0     # k parameter for sigmoid blending
    max_fet_weight: 0.8

momentum:
  lookback_days: [21, 63, 252]  # 1M, 3M, 12M
  horizon_weights: [0.25, 0.40, 0.35]
  scurve_lambda: 0.394       # Ferson-Siegel parameter
  signal_normalization_window: 252

costs:
  equity_etf_bps: 5
  commodity_etf_bps: 10
  bond_etf_bps: 3
  currency_etf_bps: 8
  futures_bps: 2             # futures are cheaper
  equity_single_bps: 8

backtest:
  start_date: "1994-01-01"
  warmup_days: 252
  in_sample_end: "2014-12-31"
  walk_forward:
    initial_window_years: 5
    refit_frequency_years: 1
  risk_free_rate: 0.02       # approximate

output:
  reports_dir: "output/reports"
  plots_dir: "output/plots"
  plot_style: "dark"         # "dark" or "light"
  figure_dpi: 150
```

---

## Implementation Notes for Claude Code

### Coding standards
```
- Python 3.11+
- Type hints on all function signatures
- Docstrings (Google style) on all public methods
- Logging via Python logging module (not print statements)
- Config loaded from YAML — no hardcoded parameters
- All randomness seeded for reproducibility
- Pandas for data, NumPy for computation, Matplotlib for plots
- Tests with pytest
```

### Key algorithms that need careful implementation

**1. Exit-time estimator from daily OHLC:**
The intra-day exit time approximation (Section 1.2, Estimator C) is the trickiest piece. The linear interpolation within the day using High/Low/Open is an approximation — document its limitations and test against synthetic paths where the true answer is known.

**2. S-curve transformation stability:**
When σ_composite is very small (quiet markets), the normalized signal s = r/σ can blow up. Cap |s| at 5.0 before applying the S-curve. Also: when σ_composite < 0.02 annualized (unrealistically low), fall back to EWMA.

**3. Walk-forward recalibration:**
Only recalibrate: corridor width Δ, horizon weights α. Do NOT recalibrate the S-curve functional form (it's theoretically motivated). Report both the in-sample and out-of-sample performance at each refit point.

**4. Transaction cost realism:**
The cost model should be sensitive to position size. For Phase 3 futures, implement a simple market impact model: cost = fixed_bps + impact_bps * sqrt(|trade_size| / ADV), where ADV is 20-day average dollar volume.

### What to build first
```
Start with Phase 1 in this exact order:
1. Data pipeline (download SPY, store as parquet)
2. EWMA volatility estimator (simplest, establishes baseline)
3. Linear TSMOM signal + basic backtest engine
4. Verify: does linear TSMOM on SPY produce SR ≈ 0.3-0.5? If not, debug.
5. Add Parkinson and FET volatility estimators
6. Add S-curve transformation
7. Add composite volatility blending
8. Full comparison of four strategies
9. Visualizations and diagnostics
10. Walk-forward validation
```

---

## Quickstart: How to Run

### Environment Setup

```bash
cd fetm-strategy
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Phase 1 — Single Asset MVP (SPY)

```bash
# Step 1: Download data
python -m fetm.data.download --ticker SPY --start 1993-01-01

# Step 2: Run full backtest (all 4 strategies compared)
python -m fetm.backtest.engine --config config/settings.yaml --phase 1

# Step 3: Generate report + plots
python -m fetm.backtest.report --run-dir output/runs/latest

# Step 4: Launch interactive dashboard
python -m fetm.dashboard.app --run-dir output/runs/latest
# → Opens browser at http://localhost:8501
```

### Phase 2 — Multi-Asset ETF Portfolio

```bash
python -m fetm.data.download --universe etf_proxy --start 2006-01-01
python -m fetm.backtest.engine --config config/settings.yaml --phase 2
python -m fetm.backtest.report --run-dir output/runs/latest
python -m fetm.dashboard.app --run-dir output/runs/latest
```

### Phase 3 — Bloomberg Futures Universe

```bash
# Requires: Bloomberg Terminal running + blpapi installed
pip install blpapi

python -m fetm.data.download --source bloomberg --universe futures
python -m fetm.backtest.engine --config config/settings.yaml --phase 3
python -m fetm.backtest.report --run-dir output/runs/latest
python -m fetm.dashboard.app --run-dir output/runs/latest
```

### CLI Reference

```bash
# Download specific tickers
python -m fetm.data.download --tickers SPY,GLD,TLT --start 2010-01-01

# Run backtest with custom config
python -m fetm.backtest.engine --config config/custom.yaml --phase 1 --verbose

# Run only walk-forward validation
python -m fetm.backtest.walk_forward --config config/settings.yaml --phase 1

# Run parameter sensitivity analysis (corridor width Δ)
python -m fetm.backtest.sensitivity --param corridor_width --range 0.005,0.03,0.005

# Compare two runs
python -m fetm.backtest.report --compare output/runs/run_001,output/runs/run_002

# Export results to Excel
python -m fetm.backtest.report --run-dir output/runs/latest --format xlsx

# Launch dashboard with multiple runs loaded
python -m fetm.dashboard.app --run-dirs output/runs/run_001,output/runs/run_002
```

### Output Structure

After a backtest run completes, the output directory contains:

```
output/runs/YYYYMMDD_HHMMSS/
├── config_snapshot.yaml          ← exact config used for this run
├── results.parquet               ← daily returns, positions, signals for all strategies
├── metrics.json                  ← all performance metrics (machine-readable)
├── metrics_summary.txt           ← human-readable performance summary
├── plots/
│   ├── equity_curves.png
│   ├── drawdowns.png
│   ├── rolling_sharpe.png
│   ├── volatility_comparison.png
│   ├── signal_scatter.png
│   ├── signal_timeseries.png
│   ├── exit_time_distribution.png
│   ├── corridor_sensitivity.png
│   ├── monthly_heatmap.png
│   ├── crisis_table.png
│   └── conditional_performance.png
└── diagnostics/
    ├── vol_estimator_comparison.parquet
    ├── exit_times_raw.parquet
    └── signal_components.parquet
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/test_volatility.py -v      # just volatility estimators
pytest tests/ -v -k "test_backtest"     # just backtest tests
```

---

## Web Dashboard

The strategy includes a Streamlit-based interactive dashboard. See the companion document `FETM_Dashboard_Specification.md` for the full dashboard design specification. The dashboard is built as Phase 1 deliverable alongside the backtest engine.

Key features:
- Real-time backtest results visualization
- Interactive parameter tuning
- Strategy comparison (4 strategies side-by-side)
- Crisis period drill-down
- Signal and volatility diagnostics
- Exportable reports

Launch: `python -m fetm.dashboard.app --run-dir output/runs/latest`

---

## Theoretical References

These papers provide the mathematical foundations. Do not re-derive them — use their results directly:

| Concept | Formula | Source |
|---------|---------|--------|
| Exit-time mean | E[τ] = Δ²/σ² | Borodin & Salminen (2002) |
| Exit-time variance | Var(τ) = 2Δ⁴/(3σ⁴) | Borodin & Salminen (2002) |
| Jensen bias correction | multiply σ̂ by n/(n+0.25) | Merrill & Sinclair (2014) |
| S-curve weighting | f(s) = λs/(s²+1) | Ferson & Siegel (2001) |
| λ calibration | λ ≈ 0.394 for unit variance | Moskowitz et al. (2025) |
| TSMOM signal | s = r_{t-h:t}/σ̂_t | Moskowitz et al. (2012) |
| Vol-targeting position | pos = σ_target/σ̂ × signal | Moskowitz et al. (2012) |

---

## Quickstart: How to Run

### Prerequisites

```
- Python 3.11+
- Git
- (Phase 3+) Bloomberg Terminal running + blpapi Python SDK installed
```

### Initial Setup

```bash
# Clone and enter project
cd fetm-strategy

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### CLI Design

The project exposes a single entry point `fetm` with subcommands. Claude Code must implement this as a CLI using `click` or `argparse`:

```bash
# ─── DATA ───────────────────────────────────────────────────
# Download data for a specific phase
fetm data download --phase 1                   # SPY only (yfinance)
fetm data download --phase 2                   # 17 ETF proxies (yfinance)
fetm data download --phase 3 --source bloomberg # 54 futures (blpapi)

# Verify data integrity
fetm data validate --phase 1

# ─── BACKTEST ───────────────────────────────────────────────
# Run backtest for a specific phase
fetm backtest run --phase 1                    # SPY single-asset
fetm backtest run --phase 2                    # Multi-asset ETF
fetm backtest run --phase 3                    # Bloomberg futures

# Run with custom config overrides
fetm backtest run --phase 1 --target-vol 0.12 --corridor-width 0.015

# Walk-forward validation
fetm backtest walk-forward --phase 1 --refit-years 1

# Parameter sensitivity analysis
fetm backtest sensitivity --phase 1 --param corridor_width --range 0.005,0.030,0.005

# ─── REPORT ─────────────────────────────────────────────────
# Generate static reports and plots
fetm report generate --run-id latest           # most recent run
fetm report generate --run-id 20260401_143022  # specific run by timestamp

# Compare multiple runs
fetm report compare --run-ids run1,run2,run3

# ─── DASHBOARD ──────────────────────────────────────────────
# Launch interactive web dashboard
fetm dashboard --port 8501                     # default Streamlit port
fetm dashboard --run-id latest                 # load specific run
fetm dashboard --live                          # auto-refresh on new data
```

### Run Naming Convention

Each backtest run produces a timestamped output directory:

```
output/
├── runs/
│   ├── 20260401_143022_phase1_SPY/
│   │   ├── config.yaml          ← frozen config snapshot
│   │   ├── returns.parquet      ← daily strategy returns (all 4 strategies)
│   │   ├── positions.parquet    ← daily positions
│   │   ├── signals.parquet      ← daily signals (raw + transformed)
│   │   ├── volatility.parquet   ← daily vol estimates (EWMA, FET, composite)
│   │   ├── exit_times.parquet   ← raw exit time data
│   │   ├── metrics.json         ← summary performance metrics
│   │   ├── plots/               ← generated PNG charts
│   │   └── report.html          ← self-contained HTML report
│   ├── 20260402_091500_phase2_ETF/
│   │   └── ...
│   └── latest -> 20260402_091500_phase2_ETF/  ← symlink to most recent
```

### Typical Workflow

```bash
# Phase 1: Start here. Takes ~2 minutes.
fetm data download --phase 1
fetm backtest run --phase 1
fetm dashboard

# Review results in browser at http://localhost:8501
# If Phase 1 success criteria pass, proceed:

fetm data download --phase 2
fetm backtest run --phase 2
fetm dashboard --run-id latest

# Phase 3 requires Bloomberg running:
fetm data download --phase 3 --source bloomberg
fetm backtest run --phase 3
fetm dashboard --run-id latest
```

### Environment Variables

```bash
# Optional — override config file location
export FETM_CONFIG=config/settings.yaml

# Phase 3+ Bloomberg connection
export BLOOMBERG_HOST=localhost
export BLOOMBERG_PORT=8194

# Dashboard
export FETM_DASHBOARD_PORT=8501
export FETM_DASHBOARD_THEME=dark
```

---

## PHASE 6: Interactive Web Dashboard
**Goal:** Build a Streamlit-based interactive dashboard for exploring backtest results, comparing strategies, and diagnosing signals in real time.
**This phase should be built alongside Phase 1 — not after.**

### 6.1 Technology Stack

```
Framework: Streamlit (pip install streamlit)
Plotting: Plotly (interactive charts, hover tooltips, zoom)
Data: reads directly from output/runs/ parquet files
State: Streamlit session state for filters and selections
Styling: dark theme, professional financial dashboard aesthetic

Do NOT use: Dash, Flask, Django, or any separate frontend framework.
Streamlit is chosen for speed of development and Python-native workflow.
```

### 6.2 Dashboard Layout — Multi-Page App

Implement as a Streamlit multi-page app with sidebar navigation:

```
dashboard/
├── app.py                    ← main entry point (streamlit run dashboard/app.py)
├── pages/
│   ├── 1_📊_Overview.py      ← executive summary
│   ├── 2_📈_Equity_Curves.py ← interactive equity curves
│   ├── 3_🔥_Volatility.py    ← volatility estimator comparison
│   ├── 4_🎯_Signals.py       ← momentum signal diagnostics
│   ├── 5_💀_Drawdowns.py     ← drawdown analysis
│   ├── 6_🌪️_Crisis.py       ← crisis period deep-dives
│   ├── 7_📋_Metrics.py       ← full metrics tables
│   └── 8_⚙️_Parameters.py    ← parameter sensitivity explorer
└── components/
    ├── data_loader.py        ← cached data loading from parquet
    ├── charts.py             ← reusable Plotly chart builders
    ├── filters.py            ← sidebar filter widgets
    └── styles.py             ← CSS overrides, color palettes
```

### 6.3 Page Specifications

**Page 1: Overview (📊)**

```
Layout: KPI cards at top, summary chart below, table at bottom

Top row — 4 KPI cards per strategy (FETM, NL+EWMA, Linear, Buy&Hold):
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ FETM         │ │ NL+EWMA      │ │ Linear TSMOM │ │ Buy & Hold   │
  │ SR: 0.84     │ │ SR: 0.72     │ │ SR: 0.51     │ │ SR: 0.43     │
  │ Ret: 12.3%   │ │ Ret: 10.1%   │ │ Ret: 7.8%    │ │ Ret: 9.2%    │
  │ MaxDD: -14%  │ │ MaxDD: -18%  │ │ MaxDD: -22%  │ │ MaxDD: -55%  │
  │ Calmar: 0.88 │ │ Calmar: 0.56 │ │ Calmar: 0.35 │ │ Calmar: 0.17 │
  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘

  Use color coding: green if metric is best-in-class, red if worst.
  KPI values update dynamically based on selected date range.

Middle — Compact equity curve (all 4 strategies, log scale, Plotly):
  - Hover shows exact date, value, drawdown
  - Click-drag to zoom into any period
  - Toggle strategies on/off via legend

Bottom — Monthly returns heatmap for selected strategy:
  - Year × Month grid
  - Color: green (positive) → red (negative)
  - Dropdown to switch between strategies

Sidebar filters (apply globally to all pages):
  - Run selector: dropdown of available runs
  - Date range: slider or date picker
  - Strategy toggle: checkboxes for which strategies to show
  - Phase indicator: badge showing which phase the run belongs to
```

**Page 2: Equity Curves (📈)**

```
Main chart — Full interactive equity curve (Plotly):
  - Log scale Y-axis (toggle to linear)
  - All 4 strategies overlaid
  - Shaded regions for crisis periods (GFC, COVID, etc.)
  - Vertical lines for rebalance events (optional toggle)
  - Secondary Y-axis: SPY price for context

Below main chart — Rolling metrics (user selects window):
  Row 1: Rolling 1Y Sharpe ratio (all strategies)
  Row 2: Rolling 1Y Return (all strategies)
  Row 3: Rolling correlation between FETM and SPY

Controls:
  - Window size selector: 63d, 126d, 252d, 504d
  - Benchmark toggle: add/remove benchmarks
  - Normalize toggle: index to 100 at start date
```

**Page 3: Volatility Lab (🔥)**

```
This is the key diagnostic page for the exit-time innovation.

Chart 1 — Triple volatility overlay:
  - σ_EWMA (blue line)
  - σ_FET (orange line)
  - σ_composite (green dashed line)
  - Shaded area between EWMA and FET when they diverge
  - Annotations at major divergence points: "FET detects regime change X days earlier"

Chart 2 — Exit time series:
  - Raw exit times τ as scatter plot (each dot = one barrier crossing)
  - Rolling mean of exit times (inverse of volatility)
  - Color dots by direction: red = downside exit, green = upside exit

Chart 3 — Exit time distribution:
  - Histogram of all exit times
  - Overlaid: fitted inverse Gaussian PDF
  - QQ plot in sidebar expander

Chart 4 — Composite blend weight w(t) over time:
  - Shows when strategy trusts FET vs EWMA
  - Highlight periods where w > 0.6 (FET dominant = regime transition)

Interactive controls:
  - Asset selector (Phase 2+: pick which asset to examine)
  - Corridor width slider: dynamically recompute FET with different Δ
  - Time range: zoom into specific periods
```

**Page 4: Signal Diagnostics (🎯)**

```
Chart 1 — S-curve visualization:
  - Scatter plot: raw signal s (x-axis) vs transformed f(s) (y-axis)
  - Color by horizon (1M=blue, 3M=orange, 12M=green)
  - Overlay theoretical curve f(s) = 0.394 * s / (s² + 1)
  - Each dot = one trading day

Chart 2 — Signal time series:
  - Top panel: combined Signal(t) over time
  - Bottom panel: position Position(t) over time
  - Color background by regime (crisis/high_vol/normal/low_vol)

Chart 3 — Signal decomposition:
  - Stacked area chart: contribution of each horizon to total signal
  - Shows when fast vs slow signal dominates

Chart 4 — Signal → Return relationship:
  - Binned scatter: signal quintile vs next-period return
  - Should show monotonic relationship (stronger signal → higher return)
  - Separate panels for long and short signals

Controls:
  - Horizon selector: view individual or combined
  - Signal type toggle: Linear / Binary / S-curve side by side
  - Asset selector (Phase 2+)
```

**Page 5: Drawdown Analysis (💀)**

```
Chart 1 — Underwater chart:
  - Drawdown from peak for all strategies
  - Hover shows: start date, trough date, recovery date, depth, duration

Chart 2 — Top 10 drawdowns table (interactive):
  - Columns: Rank, Start, Trough, Recovery, Depth, Duration, Recovery Time
  - Click a row → zooms Chart 1 to that drawdown period
  - Color: red gradient by depth

Chart 3 — Drawdown distribution:
  - Histogram of all drawdown depths
  - Overlaid: empirical CDF
  - VaR/CVaR annotation lines

Chart 4 — Recovery analysis:
  - Scatter: drawdown depth vs recovery time (days)
  - Compare FETM vs Buy&Hold recovery speed
  - Trend line showing expected recovery
```

**Page 6: Crisis Deep-Dives (🌪️)**

```
Pre-defined crisis periods (hardcoded):
  - Dot-com bust: 2000-03-24 to 2002-10-09
  - GFC: 2007-10-09 to 2009-03-09
  - Flash crash: 2015-08-18 to 2015-08-25
  - Q4 2018 selloff: 2018-10-01 to 2018-12-24
  - COVID crash: 2020-02-19 to 2020-03-23
  - 2022 rate hikes: 2022-01-03 to 2022-10-12
  - 2025 tariff/Iran: 2025-02-01 to 2025-06-30 (adjust if needed)

For each crisis:
  Row 1: Side-by-side equity curves (all 4 strategies) during crisis
  Row 2: Daily P&L bars colored by strategy
  Row 3: Volatility estimates during crisis (EWMA vs FET)
  Row 4: Signal values — did the strategy correctly position before/during crisis?

Summary table:
  ┌─────────────┬────────┬──────────┬──────────┬────────────┐
  │ Crisis      │ FETM   │ NL+EWMA  │ Linear   │ Buy&Hold   │
  ├─────────────┼────────┼──────────┼──────────┼────────────┤
  │ GFC         │ +18.2% │ +12.1%   │ +5.3%    │ -55.2%     │
  │ COVID       │ +8.7%  │ +4.2%    │ -1.1%    │ -33.8%     │
  │ ...         │        │          │          │            │
  └─────────────┴────────┴──────────┴──────────┴────────────┘

  Green = positive return during crisis, Red = negative.
  This is the strategy's "selling point" — tail hedging evidence.
```

**Page 7: Full Metrics (📋)**

```
Comprehensive metrics table — all numbers from backtest/metrics.py:

Display as a wide table with strategies as columns:
  ┌──────────────────────┬──────────┬──────────┬──────────┬──────────┐
  │ Metric               │ FETM     │ NL+EWMA  │ Linear   │ B&H      │
  ├──────────────────────┼──────────┼──────────┼──────────┼──────────┤
  │ Annualized Return    │          │          │          │          │
  │ Annualized Volatility│          │          │          │          │
  │ Sharpe Ratio         │          │          │          │          │
  │ Sortino Ratio        │          │          │          │          │
  │ Max Drawdown         │          │          │          │          │
  │ Max DD Duration      │          │          │          │          │
  │ Calmar Ratio         │          │          │          │          │
  │ VaR (95%)            │          │          │          │          │
  │ CVaR (95%)           │          │          │          │          │
  │ Skewness             │          │          │          │          │
  │ Kurtosis             │          │          │          │          │
  │ Win Rate             │          │          │          │          │
  │ Profit Factor        │          │          │          │          │
  │ Annual Turnover      │          │          │          │          │
  │ Avg Holding Period   │          │          │          │          │
  │ % Time Long          │          │          │          │          │
  │ % Time Short         │          │          │          │          │
  │ % Time Flat          │          │          │          │          │
  │ Best Month           │          │          │          │          │
  │ Worst Month          │          │          │          │          │
  │ Best Quarter         │          │          │          │          │
  │ Worst Quarter        │          │          │          │          │
  └──────────────────────┴──────────┴──────────┴──────────┴──────────┘

  Highlight best value in each row with green background.
  
Sub-sections (expandable):
  - In-sample metrics (1994-2014)
  - Out-of-sample metrics (2015-present)
  - By year: annual returns table
  - By regime: metrics split by volatility regime

Export: "Download as CSV" button for the full metrics table.
```

**Page 8: Parameter Explorer (⚙️)**

```
Interactive parameter sensitivity analysis.

User selects a parameter to vary:
  - Corridor width Δ: slider from 0.005 to 0.040
  - S-curve λ: slider from 0.1 to 1.0
  - Horizon weights: three sliders (auto-normalize to sum=1)
  - EWMA half-life: slider from 20 to 120 days
  - Target volatility: slider from 0.05 to 0.25
  - Rebalance frequency: dropdown (1, 5, 10, 21 days)

For the selected parameter:
  Chart 1: Sharpe ratio vs parameter value (line chart)
  Chart 2: Max drawdown vs parameter value
  Chart 3: Turnover vs parameter value
  Chart 4: Equity curve for 3 parameter values (low/mid/high)

This requires pre-computed sensitivity data OR on-the-fly computation.
Strategy:
  - For corridor width and horizon weights: pre-compute during backtest
    (store results of sensitivity sweep in output/runs/*/sensitivity/)
  - For other parameters: compute on-the-fly (fast enough for daily data)

"Run Custom Backtest" button:
  - User sets all parameters via sliders
  - Clicks "Run" → triggers fetm backtest run with those params
  - Results appear in dashboard after completion
  - Use Streamlit spinner during computation
```

### 6.4 Dashboard Styling

```python
# dashboard/components/styles.py

# Color palette (dark theme, financial aesthetic)
COLORS = {
    "fetm":       "#00D4AA",  # teal — primary strategy
    "nl_ewma":    "#FFB347",  # amber — nonlinear + EWMA
    "linear":     "#87CEEB",  # light blue — linear baseline
    "buyhold":    "#FF6B6B",  # coral — buy and hold
    "background": "#0E1117",  # dark charcoal
    "card_bg":    "#1E2130",  # slightly lighter
    "text":       "#FAFAFA",  # near-white
    "positive":   "#00D4AA",  # green
    "negative":   "#FF4444",  # red
    "neutral":    "#888888",  # gray
    "grid":       "#2A2D3E",  # subtle grid
}

# Plotly template
PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": COLORS["background"],
        "plot_bgcolor":  COLORS["background"],
        "font": {"color": COLORS["text"], "family": "Inter, sans-serif"},
        "xaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
        "yaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
        "hovermode": "x unified",
    }
}

# Streamlit page config (set in app.py)
PAGE_CONFIG = {
    "page_title": "FETM Strategy Dashboard",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Custom CSS injected via st.markdown
CUSTOM_CSS = """
<style>
    /* KPI cards
