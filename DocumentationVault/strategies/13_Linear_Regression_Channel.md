# Linear Regression Channel Breakout & Reversion

> **Type:** Statistical / Trend-following / Mean-reversion (dual mode)
> **Markets:** Forex, B3, Crypto
> **Timeframes:** 1h, 4h, 1D
> **Direction:** Long & Short
> **Status:** Idea

---

## Overview

A Linear Regression Channel fits an Ordinary Least Squares (OLS) regression line to the closing prices of the past N bars, then constructs parallel channel boundaries at ±K standard errors above and below this line. The regression line itself is the "fair value" of price given its recent trend; the channel boundaries represent statistically unusual deviations from that trend.

This strategy operates in two modes:
1. **Trend-following mode:** Entry when price breaks out of the channel in the direction of the regression slope (the channel slope indicates trend direction, and a breakout above the upper band in an upward-sloping channel confirms strong momentum).
2. **Mean-reversion mode:** Entry when price touches the outer band against the channel slope, betting on reversion to the regression line (the statistical center of value).

The choice of mode can be controlled by the regression slope magnitude — steep slopes favor trend-following, flat slopes favor mean reversion. This gives the strategy an intrinsic regime-awareness based on the geometry of recent price action rather than a separate regime indicator.

---

## Indicators

### Indicator 1 — Linear Regression Line

- **Input:** Bar close prices (indexed as integers 0, 1, …, N−1 within the rolling window)
- **Formula:**
  ```
  For window of N close prices ending at bar t (indices i = 0..N−1):

  x̄ = (N−1) / 2     (mean of indices)
  ȳ = mean(close[t−N+1..t])

  β₁ = Σᵢ(i − x̄)(close[t−N+1+i] − ȳ) / Σᵢ(i − x̄)²    (slope)
  β₀ = ȳ − β₁ × x̄                                          (intercept)

  reg_line(t) = β₀ + β₁ × (N−1)    (value at the most recent bar, i = N−1)
  ```
  The regression line value at the current bar is the predicted price from the OLS fit.
- **Lookback:** `lr_period` bars
- **Parameters:**
  - `lr_period` (int, default `50`) — rolling window for OLS regression

### Indicator 2 — Regression Standard Error

- **Input:** Close prices, regression line values
- **Formula:**
  ```
  residuals(i) = close[t−N+1+i] − (β₀ + β₁ × i)    for i = 0..N−1

  StdErr(t) = sqrt(Σᵢ residuals(i)² / (N − 2))
  ```
  Sample standard error of the regression (degrees of freedom: N − 2 for slope and intercept).
- **Lookback:** Same as regression (lr_period bars)

### Indicator 3 — Regression Channel Bands

- **Input:** reg_line, StdErr
- **Formula:**
  ```
  upper_band(t) = reg_line(t) + lr_mult × StdErr(t)
  lower_band(t) = reg_line(t) − lr_mult × StdErr(t)
  ```
- **Parameters:**
  - `lr_mult` (float, default `2.0`) — channel width in standard errors

### Indicator 4 — Regression Slope Normalization

- **Input:** β₁ (slope), ATR
- **Formula:**
  ```
  slope_normalized(t) = β₁(t) × lr_period / ATR(t)
  ```
  Normalized slope: how many ATR units the regression line traverses over one full regression window. Positive = uptrend, Negative = downtrend, Near zero = flat.
- **Parameters:**
  - `slope_threshold` (float, default `0.3`) — |slope_normalized| above this → trend mode; below → reversion mode

### Indicator 5 — Average True Range (ATR)

- **Input:** High, Low, Close
- **Formula:** Simple rolling mean of True Range
- **Lookback:** `atr_period + 1` bars
- **Parameters:**
  - `atr_period` (int, default `14`)

---

## Entry Signal

### Trend-Following Mode (|slope_normalized| ≥ slope_threshold)

#### Long Entry (Trending Up)

1. `slope_normalized(t) ≥ slope_threshold` — regression line is sloping up
2. `close(t) > upper_band(t)` — close breaks above the upper channel band
3. `close(t−1) ≤ upper_band(t−1)` — this is a fresh breakout (previous bar was inside channel)
4. `ATR(t)` is not NaN

#### Short Entry (Trending Down)

1. `slope_normalized(t) ≤ −slope_threshold` — regression line is sloping down
2. `close(t) < lower_band(t)` — close breaks below the lower channel band
3. `close(t−1) ≥ lower_band(t−1)` — fresh breakout

### Mean-Reversion Mode (|slope_normalized| < slope_threshold)

#### Long Entry (Flat Channel, Touch Lower Band)

1. `|slope_normalized(t)| < slope_threshold` — regression line is approximately flat
2. `close(t) ≤ lower_band(t)` — price touches or breaks lower band
3. `ATR(t)` is not NaN

#### Short Entry (Flat Channel, Touch Upper Band)

1. `|slope_normalized(t)| < slope_threshold`
2. `close(t) ≥ upper_band(t)`

**Execution (all modes):** Bar close of bar `t`

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| Slope-based mode switching | Always active | Mode determined by |slope_normalized| vs threshold |
| Session filter | Off | B3: 09:00–18:00 BRT |
| Warm-up guard | Always active | Full lr_period and atr_period required |

---

## Exit Signal

### Primary Exits — Price-Based

| Exit Type | Long | Short | Exit Price |
|-----------|------|-------|-----------|
| Stop Loss | `bar_low ≤ entry − sl_atr_mult × ATR_entry` | `bar_high ≥ entry + sl_atr_mult × ATR_entry` | SL level |
| Take Profit | `bar_high ≥ entry + tp_atr_mult × ATR_entry` | `bar_low ≤ entry − tp_atr_mult × ATR_entry` | TP level |

### Secondary Exits — Signal-Based

#### Trend Mode Exits

| Exit Type | Condition | Exit Price |
|-----------|-----------|-----------|
| Channel re-entry | `close(t) < upper_band(t)` for long (price falls back inside channel) | Bar close |
| Slope flip | slope_normalized changes sign and crosses threshold | Bar close |

#### Reversion Mode Exits

| Exit Type | Condition | Exit Price |
|-----------|-----------|-----------|
| Regression line touch | `bar_high ≥ reg_line(t)` for long (price reverts to center line) | reg_line(t) |
| Mode switch | |slope_normalized| crosses threshold upward (market starts trending) | Bar close |
| Signal reversal | Opposite band touch entry fires | Bar close |
| Session-end | Last in-session bar | Bar close |
| End of data | Dataset ends | Last close |

### Exit Priority

1. Stop Loss
2. Take Profit (ATR-based)
3. Regression line touch (reversion mode TP)
4. Channel re-entry / slope flip (trend mode)
5. Mode switch exit
6. Signal reversal
7. Session-end

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset | **1** | |
| Stop type | **Fixed** | ATR-based at entry |
| Stop loss | `entry ± sl_atr_mult × ATR_entry` | Default: 2.0 × ATR |
| Take profit | `entry ± tp_atr_mult × ATR_entry` | Default: 3.0 × ATR (trend) / 1.5 × ATR (reversion) |
| Trailing stop | **No** | |
| Default position sizing | **1 unit** | |

### Markets

| Group | Assets | Notes |
|-------|--------|-------|
| Forex | EURUSD, EURCAD, GBPCHF | 4h and 1D; regime switching behavior well-suited to FX |
| B3 | WDO, WIN | 1h and 4h; session filter required |
| Crypto | BTCUSDT, ETHUSDT | 4h and 1D; strong trend regimes create clear channel breakouts |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| `lr_period` | `50` | `[20, 30, 50, 100]` | Linear regression window (bars) |
| `lr_mult` | `2.0` | `[1.5, 2.0, 2.5]` | Channel width in standard errors |
| `slope_threshold` | `0.3` | `[0.2, 0.3, 0.5]` | Normalized slope above which trend mode activates |
| `atr_period` | `14` | fixed | ATR period |
| `sl_atr_mult` | `2.0` | `[1.5, 2.0, 2.5]` | Stop loss ATR multiple |
| `tp_atr_mult` | `3.0` | `[2.0, 3.0, 4.0]` | Take profit ATR multiple |
| `mode` | `"auto"` | — | `"auto"` (slope-based switching), `"trend"`, or `"reversion"` |

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | lr_period | lr_mult | slope_threshold | sl_atr_mult | tp_atr_mult |
|-------|----|-------|-----------|---------|-----------------|-------------|-------------|
| | | | | | | | |

---

## Performance Summary

_(Fill in after backtesting)_

| Metric | Baseline | WFO-Optimized | Notes |
|--------|----------|--------------|-------|
| Sharpe (daily) | | | |
| Profit Factor | | | |
| Win Rate | | | |
| Max Drawdown | | | |
| P(profitable) block bootstrap | | | |

**Best configuration:** TBD
**Regime sensitivity:** The dual mode is designed to handle both trending and ranging regimes. The primary risk is incorrect regime classification near the slope threshold, generating neither good trend nor good reversion signals.

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- The regression line and bands are recalculated every bar using a rolling window; the endpoint of the regression changes each bar, making the indicator non-stationary and potentially unstable near the edges of the window
- Selecting `lr_period` is critical: too short produces noisy slope estimates; too long misses regime changes in a timely manner
- The t-statistic of the OLS slope can be used as a more principled trigger for mode switching than a fixed `slope_threshold` — slopes that are not statistically significant (t < 2) could be classified as flat/random
- Standard errors assume normally distributed residuals; heavy-tailed return distributions will understate the true uncertainty of the regression
- In trend mode, price may break out of the channel and immediately reverse — adding a minimum price extension filter (e.g., price must remain above band for 2 bars) could reduce false breakout entries

---

## Implementation

**Notebook:** `technical_analysis/13_linear_regression_channel.ipynb`
**Source module:** `source/strategy.py` — `LinearRegressionChannelStrategy`
**Parameters class:** `LinearRegressionChannelParams`

### Implementation Notes

- Rolling OLS is computed in **O(N)** total via cumulative sums (see
  `source.strategy._rolling_ols_channel`) so the regression line, slope, and
  standard error are all available at the right edge of every window cheaply.
- Reversion-mode dynamic TP at the regression-line touch is **approximated**
  by a fixed ATR-based TP — the existing `Backtester` only supports
  fixed-at-entry SL/TP.
- The mode-switch exit (close when `|slope_norm|` crosses the threshold) and
  channel re-entry exit are likewise **not implemented** — same hook
  limitation.
- The `mode` field accepts `"auto"` (default — slope-driven mode switching),
  `"trend"`, or `"reversion"` to force a single mode.
- For B3, baseline params include `session_start=9`, `session_end=18`.
- Crypto group skipped — no `data/crypto/`.

---

## References

1. Raff, G. (1991). *Trading the Regression Channel*. Futures Magazine. Original description of the regression channel concept for trading.
2. Kaufman, P.J. (2013). *Trading Systems and Methods* (5th ed.). Wiley. Chapter on regression-based channels.
3. Ehlers, J.F. (2004). *Cybernetic Analysis for Stocks and Futures*. Wiley. Application of linear predictors to price channel definition.
4. Elder, A. (2002). *Come Into My Trading Room*. Wiley. Discussion of regression channels within the Elder system.
5. White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity. *Econometrica*, 48(4), 817–838. Standard error robustness for regression applied to non-normal residuals.
