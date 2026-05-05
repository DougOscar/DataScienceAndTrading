# Hurst Exponent Regime Switcher

> **Type:** Statistical / Adaptive
> **Markets:** Forex, B3, Crypto
> **Timeframes:** 1h, 4h, 1D
> **Direction:** Long & Short
> **Status:** Idea

---

## Overview

The Hurst Exponent (H) characterizes the long-range dependence structure of a time series. For financial returns:

- **H > 0.5** — persistent series (trend-reinforcing): past trends tend to continue
- **H ≈ 0.5** — random walk: no exploitable autocorrelation
- **H < 0.5** — anti-persistent series (mean-reverting): past moves tend to reverse

By computing H over a rolling window, this strategy identifies the local market regime and selects the appropriate sub-strategy: a trend-following signal when H is high, a mean-reverting signal when H is low, and no position when H is near 0.5 (random walk region where neither edge is reliable). This adaptive mechanism avoids the common failure mode of applying a single strategy regardless of regime.

The hypothesis is grounded in the Fractal Market Hypothesis (Peters, 1994), which argues that markets exhibit self-similar, persistent structure across time scales during trending conditions and anti-persistent structure during ranging periods.

---

## Indicators

### Indicator 1 — Rolling Hurst Exponent (R/S Analysis)

- **Input:** Bar close prices (log returns: `r(t) = log(close(t)/close(t−1))`)
- **Formula (R/S method):**
  ```
  For a window of N log-returns ending at bar t:
  
  For each sub-window size n in {n_min, n_min×2, …, N/2}:
    Split window into non-overlapping sub-windows of size n
    For each sub-window:
      mean_r  = mean of returns in sub-window
      cum_dev = cumulative sum of (r[i] - mean_r)
      R       = max(cum_dev) - min(cum_dev)
      S       = std(returns in sub-window)
      RS      = R / S
    RS_avg(n) = mean(RS) across all sub-windows of size n
  
  H = slope of OLS regression:
      log(RS_avg) ~ H × log(n) + constant
  ```
  `H` is estimated as the slope of the log-log regression of rescaled range against sub-window size.
- **Lookback:** `hurst_window` bars (minimum ~64 for stable estimates)
- **Computation:** Calculated every `hurst_step` bars to reduce computational cost; held constant between recalculations
- **Parameters:**
  - `hurst_window` (int, default `128`) — rolling window of log-returns used for H estimation
  - `hurst_n_min` (int, default `8`) — smallest sub-window size
  - `hurst_step` (int, default `1`) — recalculate H every N bars (1 = every bar)

### Indicator 2 — Trend Sub-Signal (SMA Crossover)

Used when H is in the persistent regime. Identical definition to Strategy 01.
- **Formula:** `SMA_fast(t) > SMA_slow(t)` and crossover on previous bar
- **Parameters:**
  - `trend_fast` (int, default `20`)
  - `trend_slow` (int, default `50`)

### Indicator 3 — Mean Reversion Sub-Signal (RSI Extreme Crossover)

Used when H is in the anti-persistent regime. Identical definition to Strategy 02.
- **Formula:** RSI crosses back through oversold/overbought threshold
- **Parameters:**
  - `rsi_period` (int, default `14`)
  - `rsi_lower` (int, default `30`)
  - `rsi_upper` (int, default `70`)

### Indicator 4 — Average True Range (ATR)

- **Input:** High, Low, Close
- **Formula:** Simple rolling mean of True Range
- **Lookback:** `atr_period + 1` bars
- **Parameters:**
  - `atr_period` (int, default `14`)

---

## Entry Signal

### Regime Classification

At each bar `t`, classify the current regime based on `H(t)`:

```
if H(t) > hurst_trend_threshold:
    regime(t) = "trending"
elif H(t) < hurst_mean_rev_threshold:
    regime(t) = "mean_reverting"
else:
    regime(t) = "random_walk"   →  no trade
```

Default thresholds:
- `hurst_trend_threshold` = 0.60
- `hurst_mean_rev_threshold` = 0.40

### Long Entry (Trending Regime)

1. `regime(t) = "trending"`
2. `SMA_fast(t) > SMA_slow(t)` AND `SMA_fast(t−1) ≤ SMA_slow(t−1)` (golden cross)
3. `ATR(t)` is not NaN

### Long Entry (Mean-Reverting Regime)

1. `regime(t) = "mean_reverting"`
2. `RSI(t) > rsi_lower` AND `RSI(t−1) ≤ rsi_lower` (RSI crosses up through oversold)
3. `ATR(t)` is not NaN

**Execution (both cases):** Bar close of bar `t`

### Short Entry (Trending Regime)

1. `regime(t) = "trending"`
2. `SMA_fast(t) < SMA_slow(t)` AND `SMA_fast(t−1) ≥ SMA_slow(t−1)` (death cross)

### Short Entry (Mean-Reverting Regime)

1. `regime(t) = "mean_reverting"`
2. `RSI(t) < rsi_upper` AND `RSI(t−1) ≥ rsi_upper` (RSI crosses down through overbought)

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| Random walk gate | Always active | `regime = "random_walk"` → no new entries |
| Regime consistency | Optional | Require H to have been in the same regime for at least `regime_min_bars` consecutive bars before entering |
| Session filter | Off | B3: 09:00–18:00 BRT |
| Warm-up guard | Always active | All indicators including `hurst_window` bars of data required |

---

## Exit Signal

### Primary Exits — Price-Based (both regimes)

| Exit Type | Long | Short | Exit Price |
|-----------|------|-------|-----------|
| Stop Loss | `bar_low ≤ entry − sl_atr_mult × ATR_entry` | `bar_high ≥ entry + sl_atr_mult × ATR_entry` | SL level |
| Take Profit | `bar_high ≥ entry + tp_atr_mult × ATR_entry` | `bar_low ≤ entry − tp_atr_mult × ATR_entry` | TP level |

### Secondary Exits — Regime Change Exit

| Exit Type | Condition | Exit Price |
|-----------|-----------|-----------|
| Regime switch | `regime(t)` changes away from the entry regime (e.g., was "trending" at entry, now "mean_reverting" or "random_walk") | Bar close |
| Signal reversal | Sub-strategy generates opposite entry signal | Bar close |
| Session-end | Last in-session bar | Bar close |
| End of data | Dataset ends with open position | Last close |

### Exit Priority

1. Stop Loss
2. Take Profit
3. Regime switch exit (if `use_regime_exit = True`)
4. Signal reversal
5. Session-end

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset | **1** | |
| Stop type | **Fixed** | ATR at entry bar |
| Stop loss | `entry ± sl_atr_mult × ATR_entry` | Default: 2.0 × ATR |
| Take profit | `entry ± tp_atr_mult × ATR_entry` | Default: 3.0 × ATR |
| Trailing stop | **No** | |
| Default position sizing | **1 unit** | |

### Markets

| Group | Assets | Notes |
|-------|--------|-------|
| Forex | EURUSD, EURCAD, GBPCHF | 4h and 1D; long hurst_window (128+) needs substantial data |
| B3 | WDO, WIN | 1h; session filter required |
| Crypto | BTCUSDT, ETHUSDT | 4h; Crypto shows strong H-regime cycles between trending and ranging |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| `hurst_window` | `128` | `[64, 128, 256]` | Rolling window of log-returns for H estimation |
| `hurst_n_min` | `8` | fixed | Smallest sub-window size in R/S analysis |
| `hurst_step` | `1` | `[1, 4]` | Recalculate H every N bars |
| `hurst_trend_threshold` | `0.60` | `[0.55, 0.60, 0.65]` | H above this → trending regime |
| `hurst_mean_rev_threshold` | `0.40` | `[0.35, 0.40, 0.45]` | H below this → mean-reverting regime |
| `regime_min_bars` | `0` | `[0, 3, 5]` | Bars regime must be stable before entry |
| `trend_fast` | `20` | `[10, 20, 30]` | Fast SMA for trending sub-strategy |
| `trend_slow` | `50` | `[40, 60, 100]` | Slow SMA for trending sub-strategy |
| `rsi_period` | `14` | `[10, 14]` | RSI for mean-reversion sub-strategy |
| `rsi_lower` | `30` | `[25, 30, 35]` | RSI oversold threshold |
| `rsi_upper` | `70` | `[65, 70, 75]` | RSI overbought threshold |
| `atr_period` | `14` | fixed | ATR for stops |
| `sl_atr_mult` | `2.0` | `[1.5, 2.0, 2.5]` | Stop loss ATR multiple |
| `tp_atr_mult` | `3.0` | `[2.0, 3.0, 4.0]` | Take profit ATR multiple |
| `use_regime_exit` | `True` | — | Exit on regime change |

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | hurst_window | h_trend | h_mean_rev | trend_fast | trend_slow |
|-------|----|-------|--------------|---------|------------|------------|------------|
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
**Regime sensitivity:** By design, the strategy adapts to regimes. The key validation is whether the Hurst-based regime classification adds value beyond simpler filters (ADX, BBWidth)

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- R/S analysis Hurst estimates are noisy on short windows and can lag regime changes; DFA (Detrended Fluctuation Analysis) may give more robust estimates
- The Hurst Exponent computation is O(N²) per bar in the naive implementation; vectorized computation or periodic recalculation (every 4–8 bars) is necessary for performance
- The two-threshold gap (0.40 to 0.60) creates a dead zone where no trades are taken; this can lead to long periods of inactivity during sustained random-walk phases
- Finite-sample Hurst estimates are biased upward for small windows; a correction factor should be applied (Lo, 1991)
- Consider DFA (Detrended Fluctuation Analysis) as an alternative H estimator — it is more robust to non-stationarity

---

## Implementation

**Notebook:** `technical_analysis/08_hurst_regime_switcher.ipynb`
**Source module:** `source/strategy.py` — `HurstRegimeSwitcherStrategy`
**Parameters class:** `StrategyParams`

---

## References

1. Hurst, H.E. (1951). Long-Term Storage Capacity of Reservoirs. *Transactions of the American Society of Civil Engineers*, 116, 770–799. Original R/S analysis.
2. Mandelbrot, B.B., & Wallis, J.R. (1969). Robustness of the Rescaled Range R/S in the Measurement of Noncyclic Long Run Statistical Dependence. *Water Resources Research*, 5(5), 967–988.
3. Peters, E.E. (1994). *Fractal Market Analysis: Applying Chaos Theory to Investment and Economics*. Wiley. Application of Hurst exponent to financial markets.
4. Lo, A.W. (1991). Long-Term Memory in Stock Market Prices. *Econometrica*, 59(5), 1279–1313. Bias-corrected R/S statistic.
5. Peng, C.K., et al. (1994). Mosaic Organization of DNA Nucleotides. *Physical Review E*, 49(2), 1685–1689. Original DFA paper — alternative H estimator.
6. Di Matteo, T., Aste, T., & Dacorogna, M.M. (2005). Long-term memories of developed and emerging markets: Using the scaling analysis to characterize their stage of development. *Journal of Banking & Finance*, 29(4), 827–851. Application to financial markets including FX.
