# FETM Dashboard: Interactive Backtest Viewer
## Implementation Specification for Claude Code

**Companion to:** FETM_Strategy_Specification.md
**Framework:** Streamlit (multi-page app)
**Charts:** Plotly (interactive) for dashboard, Matplotlib (static) for exports
**Build alongside:** Phase 1 backtest engine — dashboard should work as soon as first backtest completes.

---

## Overview

The dashboard is the primary interface for analyzing FETM backtest results. It reads pre-computed backtest outputs (parquet files + JSON metrics) and presents them interactively. It does NOT run backtests — that's the CLI's job. The dashboard is read-only and fast.

**Launch command:**
```bash
python -m fetm.dashboard.app --run-dir output/runs/latest
# or with multiple runs:
python -m fetm.dashboard.app --run-dirs output/runs/run_001,output/runs/run_002
```

**URL:** `http://localhost:8501`

---

## Data Contract

The dashboard expects these files in the run directory:

```
output/runs/YYYYMMDD_HHMMSS/
├── config_snapshot.yaml              ← config used for this run
├── results.parquet                   ← main results table
├── metrics.json                      ← aggregated performance metrics
└── diagnostics/
    ├── vol_estimator_comparison.parquet
    ├── exit_times_raw.parquet
    └── signal_components.parquet
```

### results.parquet schema

```
Columns:
  date:               datetime64    — trading date
  asset:              str           — ticker symbol (e.g., "SPY", "ES1")

  # Price data
  open:               float64
  high:               float64
  low:                float64
  close:              float64
  volume:             float64
  log_return:         float64       — log(close_t / close_{t-1})

  # Volatility estimates
  vol_ewma:           float64       — annualized EWMA volatility
  vol_parkinson:      float64       — annualized Parkinson range vol
  vol_fet:            float64       — annualized first exit-time vol
  vol_composite:      float64       — blended composite vol
  fet_weight:         float64       — current weight on FET in composite (0 to 0.8)

  # Signals (per horizon)
  signal_raw_21d:     float64       — r/σ for 1M lookback
  signal_raw_63d:     float64       — r/σ for 3M lookback
  signal_raw_252d:    float64       — r/σ for 12M lookback
  signal_nl_21d:      float64       — S-curve transformed 1M
  signal_nl_63d:      float64       — S-curve transformed 3M
  signal_nl_252d:     float64       — S-curve transformed 12M
  signal_combined:    float64       — weighted combination

  # Positions & returns for each strategy
  position_buyhold:   float64       — always 1.0 (benchmark)
  position_linear:    float64       — linear TSMOM position
  position_binary:    float64       — binary TSMOM position
  position_fetm:      float64       — FETM (our strategy) position

  return_buyhold:     float64       — daily return of buy-and-hold
  return_linear:      float64       — daily return of linear TSMOM (after costs)
  return_binary:      float64       — daily return of binary TSMOM (after costs)
  return_fetm:        float64       — daily return of FETM (after costs)

  cumreturn_buyhold:  float64       — cumulative return
  cumreturn_linear:   float64
  cumreturn_binary:   float64
  cumreturn_fetm:     float64

  # Exit-time diagnostics
  exit_time_last:     float64       — most recent exit time τ (in years)
  exit_times_mean:    float64       — mean of buffer
  corridor_upper:     float64       — current upper barrier price
  corridor_lower:     float64       — current lower barrier price
  days_since_exit:    int64         — days since last barrier crossing

  # Regime
  vol_regime:         str           — "crisis"/"high_vol"/"normal"/"low_vol"

  # Meta
  is_rebalance_day:   bool          — was position updated today?
  turnover:           float64       — |Δposition| on this day
  transaction_cost:   float64       — cost incurred today
```

### metrics.json schema

```json
{
  "run_id": "20260401_143022",
  "phase": 1,
  "config": { /* snapshot of settings.yaml */ },
  "period": {
    "start": "1994-01-03",
    "end": "2026-03-28",
    "trading_days": 8103,
    "in_sample_end": "2014-12-31"
  },
  "strategies": {
    "buyhold": {
      "annualized_return": 0.098,
      "annualized_vol": 0.195,
      "sharpe_ratio": 0.42,
      "sortino_ratio": 0.61,
      "max_drawdown": -0.552,
      "max_drawdown_duration_days": 1024,
      "calmar_ratio": 0.18,
      "var_95": -0.019,
      "var_99": -0.032,
      "cvar_95": -0.028,
      "skewness": -0.21,
      "kurtosis": 11.4,
      "worst_month": -0.169,
      "worst_quarter": -0.298,
      "annual_turnover": 0.0,
      "avg_holding_period": null,
      "pct_time_long": 1.0,
      "pct_time_short": 0.0,
      "pct_time_flat": 0.0,
      "win_rate_monthly": 0.62,
      "profit_factor": 1.45
    },
    "linear": { /* same fields */ },
    "binary": { /* same fields */ },
    "fetm":   { /* same fields */ }
  },
  "crisis_performance": {
    "dotcom_2000_2002": { "buyhold": -0.42, "linear": 0.08, "binary": 0.11, "fetm": 0.15 },
    "gfc_2008_2009":    { /* ... */ },
    "covid_2020":       { /* ... */ },
    /* ... more crisis periods ... */
  },
  "conditional_performance": {
    "bottom_tercile": { "buyhold": -0.18, "linear": 0.04, "binary": 0.06, "fetm": 0.09 },
    "middle_tercile": { /* ... */ },
    "top_tercile":    { /* ... */ }
  },
  "corridor_sensitivity": {
    "delta_values": [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03],
    "sharpe_ratios": [0.38, 0.44, 0.51, 0.49, 0.45, 0.41, 0.37]
  },
  "vol_estimator_comparison": {
    "regime_detection_lag_days": { "ewma": 18.5, "parkinson": 12.3, "fet": 7.1, "composite": 8.4 },
    "correlation_with_realized": { "ewma": 0.87, "parkinson": 0.91, "fet": 0.89, "composite": 0.93 }
  }
}
```

---

## Page Specifications

### Global Layout

```
Sidebar:
  ├── FETM Strategy logo/title
  ├── Run selector dropdown (if multiple runs loaded)
  ├── Date range picker (start/end)
  ├── In-sample / Out-of-sample toggle
  │   Options: "Full Period" | "In-Sample Only" | "Out-of-Sample Only"
  ├── Asset selector (Phase 2+: multi-select from universe)
  └── Theme toggle: Dark / Light

All pages respect sidebar filters.
Default: Out-of-sample period, all assets.
```

**Streamlit config (.streamlit/config.toml):**
```toml
[theme]
primaryColor = "#4FC3F7"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1A1F2E"
textColor = "#E0E0E0"
font = "sans serif"

[server]
headless = true
port = 8501
```

---

### Page 1: Overview (01_overview.py)

**Purpose:** Executive summary — one screen that answers "is this strategy working?"

**Layout:**

```
Row 1: KPI Cards (4 cards, one per strategy)
  Each card shows:
    Strategy name + colored indicator
    Sharpe Ratio (large font, primary metric)
    Annualized Return | Max Drawdown (secondary)
    ▲/▼ vs benchmark (green/red delta badge)

  Card colors:
    Buy & Hold:    gray (#6B7280)
    Linear TSMOM:  blue (#3B82F6)
    Binary TSMOM:  amber (#F59E0B)
    FETM:          emerald (#10B981) ← hero strategy, slightly larger card

Row 2: Equity Curves (full width Plotly chart)
  - Log scale Y-axis
  - All 4 strategies overlaid
  - Shaded regions for crisis periods (light red background)
  - Vertical dashed line at in-sample/out-of-sample boundary
  - Hover tooltip: date, cumulative return, drawdown at that point
  - Toggle buttons: show/hide each strategy
  - Zoom/pan enabled

Row 3: Two columns
  Left: Drawdown Chart
    - Underwater plot (negative Y-axis)
    - All 4 strategies
    - Highlight max drawdown period for FETM
  Right: Rolling 12-Month Sharpe
    - All 4 strategies
    - Horizontal dashed line at 0 (break-even)
    - Horizontal dashed line at 1.0 (excellent)

Row 4: Monthly Returns Heatmap
  - Year (rows) × Month (columns)
  - Cell color: diverging red-white-green
  - Cell text: percentage return
  - Dropdown to select which strategy to display
```

---

### Page 2: Strategy Comparison (02_strategies.py)

**Purpose:** Deep side-by-side comparison of all 4 strategies.

**Layout:**

```
Row 1: Full Metrics Table
  Columns: Metric | Buy&Hold | Linear | Binary | FETM
  Rows: All metrics from metrics.json
  Color coding:
    - Best value in each row: bold green
    - Worst value: light red
  Sections:
    Returns (ann. return, Sharpe, Sortino, Calmar)
    Risk (vol, max DD, DD duration, VaR, CVaR, skew, kurtosis)
    Efficiency (turnover, holding period, win rate, profit factor)

Row 2: Annual Returns Bar Chart
  - Grouped bar chart: year × strategy
  - Each year shows 4 bars side-by-side
  - Red bars for negative years

Row 3: Return Distribution
  Two columns:
    Left: Overlaid histograms of daily returns (4 strategies)
    Right: QQ-plot of FETM vs normal distribution

Row 4: Crisis Performance Table
  - Rows: each crisis period (dates, description)
  - Columns: return for each strategy during that period
  - Highlight cells: green if positive, red if negative
  - Last row: average across all crises

Row 5: Conditional Performance
  - Replicates NLTSMOM paper Table 8
  - 3 columns: Bottom Tercile | Middle | Top Tercile (of benchmark returns)
  - Rows: Sharpe ratio of each strategy in that regime
  - Bar chart version alongside the table
```

---

### Page 3: Signal Diagnostics (03_signals.py)

**Purpose:** Understand the momentum signal construction and S-curve transformation.

**Layout:**

```
Row 1: S-Curve Visualization (key chart)
  - Interactive Plotly scatter plot
  - X-axis: raw normalized signal s (before S-curve)
  - Y-axis: transformed signal f(s) (after S-curve)
  - Overlay: theoretical S-curve f(s) = 0.394 * s / (s² + 1) as solid line
  - Scatter points: actual (s, f(s)) pairs colored by lookback horizon
    Blue = 21d, Orange = 63d, Green = 252d
  - Density heatmap option (toggle between scatter and hexbin)
  - Annotation: show where rollback begins (|s| > 1)

Row 2: Signal Time Series (full width)
  - Three subplots stacked vertically, sharing X-axis:
    Top: Price (log scale) with corridor bands (upper/lower barrier)
    Middle: Raw signal s_h(t) for each horizon
    Bottom: Combined Signal(t) with position overlay
  - Color-coded by horizon
  - Shaded background by vol regime (crisis=red, high=orange, normal=none, low=blue)

Row 3: Two columns
  Left: Signal Distribution
    - Histogram of combined signal
    - Overlay: fitted normal distribution
    - Statistics: mean, std, skew, kurtosis
    - Vertical lines at ±1 (where S-curve begins attenuating)
  Right: Horizon Contribution
    - Stacked area chart over time
    - Shows α₁*f(s_21) + α₂*f(s_63) + α₃*f(s_252) decomposition
    - See which horizon is driving the signal at each point

Row 4: Signal Quality Metrics
  - Rolling IC (information coefficient): corr(signal_t, return_{t+1})
  - Rolling hit rate: % of times signal direction matches return direction
  - Signal autocorrelation: should be high (persistent signals)
  - Turnover from signal changes
```

---

### Page 4: Volatility Analysis (04_volatility.py)

**Purpose:** Compare volatility estimators and validate exit-time innovation.

**Layout:**

```
Row 1: Volatility Estimator Time Series (full width)
  - 4 lines: EWMA, Parkinson, FET, Composite
  - Shaded bands for vol regimes
  - Zoom into specific periods via date picker
  - Annotation arrows at major regime transitions (e.g., Feb 2020 COVID spike)

Row 2: Two columns
  Left: Regime Transition Speed
    - For each major vol spike (defined as >50% increase in 20 days):
      Table showing how many days each estimator took to detect it
    - Bar chart: average detection lag by estimator
    - Key insight: FET should be 5-10 days faster than EWMA
  Right: FET Weight in Composite
    - Time series of w(t) — the sigmoid blending weight
    - Should spike during regime transitions
    - Overlay: |σ_FET - σ_EWMA| / σ_EWMA (the divergence that drives w)

Row 3: Exit-Time Deep Dive
  Three columns:
    Left: Exit Time Distribution
      - Histogram of all recorded exit times τ
      - Overlay: theoretical inverse Gaussian distribution
      - Fit statistics: K-S test p-value
    Middle: Exit Frequency Over Time
      - Rolling count of barrier crossings per month
      - Should inversely correlate with volatility
      - Overlay: inverse of σ_EWMA for comparison
    Right: Corridor Visualization
      - Recent 60-day price chart
      - Show actual corridor bands (upper/lower barriers)
      - Mark barrier crossing events with markers
      - Show corridor resets

Row 4: Corridor Width Sensitivity
  - Line chart: Sharpe ratio vs corridor width Δ
  - X-axis: Δ values tested
  - Y-axis: out-of-sample Sharpe ratio
  - Vertical line at selected Δ
  - Second line: annual turnover vs Δ (right Y-axis)
  - Goal: show that optimal Δ is robust (flat around peak)
```

---

### Page 5: Crisis Analysis (05_crisis.py)

**Purpose:** How does FETM perform when it matters most — during market stress?

**Layout:**

```
Row 1: Crisis Period Selector
  - Dropdown/buttons for predefined crisis periods:
    2000-2002 Dot-com Bust
    2007-2009 Global Financial Crisis
    2010 Flash Crash (May 6)
    2011 Euro Debt Crisis
    2015 Aug China Devaluation
    2018 Q4 Selloff
    2020 Feb-Mar COVID Crash
    2022 Rate Hiking Cycle
    2025 Tariff / Iran Crisis
  - "Custom Range" option with date pickers

Row 2: Crisis Zoom Panel (updates based on selection)
  Left (60% width): Equity Curves During Crisis
    - Rebased to 100 at start of crisis period
    - All 4 strategies
    - Volume bars at bottom
  Right (40% width): Crisis Metrics Card
    - Total return for each strategy
    - Max drawdown during period
    - Recovery time (days to recoup losses)
    - FETM vs benchmark excess return

Row 3: Volatility Behavior During Crisis
  - σ_EWMA vs σ_FET during the selected crisis
  - Highlight: how many days earlier did FET detect the vol spike?
  - Exit-time frequency: show accelerating barrier crossings
  - Composite weight w(t): show shift toward FET

Row 4: Signal Behavior During Crisis
  - Decomposed signal: which horizon drove the position change?
  - Position trace: show how FETM de-levered (or reversed) during crisis
  - Compare to binary TSMOM: did S-curve attenuation help?
  - Transaction costs incurred during crisis (high-turnover periods)

Row 5: All-Crisis Summary Table
  - Rows: each crisis
  - Columns: return, max DD, vol estimator lag, signal reversal speed
  - For each strategy
  - Color-coded performance
  - Bottom row: average across all crises
```

---

### Page 6: Asset Breakdown (06_assets.py) — Phase 2+ Only

**Purpose:** Per-asset and per-sector analysis for multi-asset portfolios.

```
Row 1: Asset Class Contribution
  - Stacked area chart: cumulative return contribution by sector
    (Equities, Commodities, Rates, FX)
  - Toggle: absolute contribution vs percentage

Row 2: Individual Asset Performance Table
  - Sortable table: asset | sector | sharpe | return | max DD | FET improvement
  - FET improvement = Sharpe(FETM) - Sharpe(Linear TSMOM) for that asset
  - Highlight: which assets benefit most from exit-time volatility?

Row 3: Correlation Matrix
  - Heatmap of pairwise correlations between asset signals
  - Toggle: return correlations vs signal correlations
  - Cluster by sector

Row 4: Per-Asset Exit-Time Analysis
  - Select individual asset from dropdown
  - Show same charts as Page 4 but for the selected asset
  - Compare: corridor width Δ sensitivity per asset class
```

---

### Page 7: Parameter Tuning (07_parameters.py)

**Purpose:** Interactive what-if analysis — change parameters and see impact.

**Important:** This page does NOT re-run backtests. It loads pre-computed sensitivity results or uses fast approximations on cached data.

```
Row 1: Parameter Controls
  Sliders:
    - Corridor width Δ:      0.005 to 0.03, step 0.001
    - 1M weight α₁:          0.0 to 0.6, step 0.05
    - 3M weight α₂:          0.0 to 0.6, step 0.05
    - 12M weight α₃:         0.0 to 0.6, step 0.05
      (display warning if α₁ + α₂ + α₃ ≠ 1.0)
    - S-curve λ:             0.2 to 0.8, step 0.05
    - Target vol:            0.05 to 0.25, step 0.01
    - EWMA halflife:         20 to 120 days, step 5
    - FET buffer size:       10 to 60, step 5

  Button: "Re-compute with these parameters"
  (Fast: only recomputes signals and positions on cached price data,
   does not re-download data. Should complete in <5 seconds for Phase 1.)

Row 2: Impact Preview
  - Show old Sharpe vs new Sharpe (big number comparison)
  - Equity curve: old (dashed) vs new (solid) overlay
  - Key metrics table: before vs after

Row 3: 2D Sensitivity Heatmap
  - Select two parameters (dropdowns)
  - Show Sharpe ratio as heatmap (color = Sharpe)
  - X-axis: parameter 1, Y-axis: parameter 2
  - Mark current selection with crosshair
  - This requires pre-computed grid (from sensitivity analysis CLI)

Row 4: Walk-Forward Stability
  - For each walk-forward window, show:
    Optimal Δ selected, Sharpe in-sample vs out-of-sample
  - Line chart: parameter values over time (are they stable?)
  - If parameters jump around → overfitting warning
```

---

## Technical Implementation Details

### Dependencies (add to requirements.txt)

```
streamlit>=1.32.0
plotly>=5.18.0
pandas>=2.1.0
numpy>=1.25.0
pyyaml>=6.0
watchdog>=3.0.0           # for file change detection
streamlit-extras>=0.3.0   # optional: for enhanced components
```

### Streamlit App Entry Point (dashboard/app.py)

```python
"""
FETM Dashboard — Interactive Backtest Viewer

Launch:
    python -m fetm.dashboard.app --run-dir output/runs/latest
    streamlit run fetm/dashboard/app.py -- --run-dir output/runs/latest
"""
import streamlit as st

st.set_page_config(
    page_title="FETM Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Parse CLI args for run directory
# Load data into st.session_state (cached)
# Render sidebar filters
# Main page content from pages/ modules
```

### Data Loading Strategy

```python
@st.cache_data(ttl=300)  # refresh every 5 minutes
def load_run(run_dir: str):
    """Load all backtest outputs into memory."""
    results = pd.read_parquet(run_dir / "results.parquet")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    config = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text())

    diagnostics = {}
    diag_dir = run_dir / "diagnostics"
    if diag_dir.exists():
        for f in diag_dir.glob("*.parquet"):
            diagnostics[f.stem] = pd.read_parquet(f)

    return {
        "results": results,
        "metrics": metrics,
        "config": config,
        "diagnostics": diagnostics,
    }
```

### Chart Standards

All Plotly charts should follow these conventions:

```python
CHART_DEFAULTS = {
    "template": "plotly_dark",     # matches Streamlit dark theme
    "height": 450,                 # consistent chart height
    "margin": dict(l=60, r=30, t=50, b=40),
    "font": dict(family="Inter, sans-serif", size=12),
    "hovermode": "x unified",
}

STRATEGY_COLORS = {
    "buyhold": "#6B7280",   # gray
    "linear":  "#3B82F6",   # blue
    "binary":  "#F59E0B",   # amber
    "fetm":    "#10B981",   # emerald (hero)
}

HORIZON_COLORS = {
    "21d":  "#60A5FA",   # light blue (fast)
    "63d":  "#FBBF24",   # yellow (medium)
    "252d": "#34D399",   # green (slow)
}

VOL_ESTIMATOR_COLORS = {
    "ewma":      "#9CA3AF",  # gray
    "parkinson": "#A78BFA",  # purple
    "fet":       "#F472B6",  # pink
    "composite": "#10B981",  # emerald
}

REGIME_COLORS = {
    "crisis":   "rgba(239, 68, 68, 0.15)",    # red tint
    "high_vol": "rgba(251, 191, 36, 0.10)",    # amber tint
    "normal":   "rgba(0, 0, 0, 0)",            # transparent
    "low_vol":  "rgba(96, 165, 250, 0.08)",    # blue tint
}
```

### KPI Card Component

```python
def kpi_card(label: str, value: float, delta: float = None,
             format_str: str = "{:.2f}", color: str = "#10B981"):
    """Render a styled KPI card."""
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta > 0 else "▼"
        delta_color = "#10B981" if delta > 0 else "#EF4444"
        delta_html = f'<span style="color:{delta_color};font-size:14px">{arrow} {abs(delta):.2f}</span>'

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}05);
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    ">
        <div style="color:#9CA3AF;font-size:13px;text-transform:uppercase;letter-spacing:0.5px">{label}</div>
        <div style="font-size:32px;font-weight:700;color:#F3F4F6;margin:4px 0">{format_str.format(value)}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)
```

### Export Functionality

Every page should include an export button:

```python
# At bottom of each page
with st.expander("📥 Export"):
    col1, col2, col3 = st.columns(3)
    with col1:
        # Download metrics as CSV
        csv = filtered_data.to_csv(index=False)
        st.download_button("Download CSV", csv, "fetm_data.csv", "text/csv")
    with col2:
        # Download current chart as HTML
        html = fig.to_html(include_plotlyjs="cdn")
        st.download_button("Download Chart (HTML)", html, "chart.html", "text/html")
    with col3:
        # Download as static PNG (requires kaleido)
        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        st.download_button("Download Chart (PNG)", img_bytes, "chart.png", "image/png")
```

---

## Mobile / Responsive Considerations

```
Streamlit handles basic responsiveness, but add:
- Use st.columns() with equal weights for 2-col layouts
- On narrow screens, charts stack vertically automatically
- KPI cards: use st.columns(4) on desktop, will wrap on mobile
- Tables: enable horizontal scroll with st.dataframe()
- Chart height: 450px desktop, 300px for mobile-friendly
```

---

## Performance Targets

```
Dashboard load time:
  - Phase 1 (single asset, 30 years): < 3 seconds
  - Phase 2 (17 assets, 20 years): < 5 seconds
  - Phase 3 (54 futures, 30+ years): < 10 seconds

Chart interaction:
  - Zoom/pan: instant (<100ms)
  - Strategy toggle: <200ms
  - Date range filter: <500ms

Parameter re-computation (Page 7):
  - Phase 1: < 5 seconds
  - Phase 2: < 15 seconds
  - Phase 3: < 30 seconds
```

---

## Build Order

```
Build the dashboard in this order:

1. app.py + data_loader.py + theme.py
   → Can open empty dashboard with sidebar

2. Page 1: Overview (KPI cards + equity curve)
   → Usable as soon as first backtest completes

3. Page 2: Strategy Comparison (metrics table + annual returns)
   → Core analysis capability

4. Page 4: Volatility Analysis
   → Validates the core innovation (FET vs EWMA)

5. Page 3: Signal Diagnostics (S-curve visualization)
   → Helps understand signal behavior

6. Page 5: Crisis Analysis
   → The "money slide" for convincing yourself the strategy works

7. Page 7: Parameter Tuning
   → Requires sensitivity analysis results

8. Page 6: Asset Breakdown
   → Only meaningful after Phase 2+ backtest
```

---

## Requirements Addition

Add these to the project's requirements.txt:

```
# Core
numpy>=1.25.0
pandas>=2.1.0
matplotlib>=3.8.0
pyyaml>=6.0
pyarrow>=14.0.0

# Data
yfinance>=0.2.31

# Backtesting
scipy>=1.11.0
tqdm>=4.66.0

# Dashboard
streamlit>=1.32.0
plotly>=5.18.0
kaleido>=0.2.1          # for static chart export

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Bloomberg (Phase 3 — install separately)
# blpapi>=3.19.0
```
