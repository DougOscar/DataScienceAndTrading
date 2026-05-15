# MACD Histogram Momentum

> **Type:** Trend-following / Momentum
> **Markets:** Forex, B3, Crypto
> **Timeframes:** 1h, 4h, 1D
> **Direction:** Long & Short
> **Status:** Idea

---

## Overview

MACD (Moving Average Convergence/Divergence) measures the distance between a fast and slow exponential moving average, capturing medium-term momentum. The histogram (MACD Line − Signal Line) represents the second derivative of price, revealing whether momentum is accelerating or decelerating. This strategy enters when the MACD histogram crosses the zero line (MACD Line crosses its Signal Line) and confirms that momentum is accelerating — histogram magnitude is growing — to filter out low-conviction crossovers in choppy markets.

The idea is that a MACD zero-line crossover backed by histogram acceleration is more likely to be the early stage of a sustained trend than a simple crossover in flat conditions. A volume surge confirmation (tick_vol ratio) further separates genuine institutional momentum from noise.

---

## Indicators

### Indicator 1 — MACD Line

- **Input:** Bar close prices
- **Formula:**
  ```
  EMA_fast(t) = EMA(close, macd_fast)[t]     (α = 2 / (macd_fast + 1))
  EMA_slow(t) = EMA(close, macd_slow)[t]     (α = 2 / (macd_slow + 1))
  MACD(t)     = EMA_fast(t) − EMA_slow(t)
  ```
  Standard exponential moving average with smoothing factor `α = 2 / (period + 1)`.
- **Lookback:** `macd_slow` bars before EMA_slow is numerically stable (in practice, ~3–4× period for full convergence)
- **Parameters:**
  - `macd_fast` (int, default `12`)
  - `macd_slow` (int, default `26`)

### Indicator 2 — Signal Line

- **Input:** MACD Line
- **Formula:**
  ```
  Signal(t) = EMA(MACD, signal_period)[t]    (α = 2 / (signal_period + 1))
  ```
- **Lookback:** `macd_slow + signal_period` bars (compound lookback)
- **Parameters:**
  - `signal_period` (int, default `9`)

### Indicator 3 — MACD Histogram

- **Input:** MACD Line, Signal Line
- **Formula:**
  ```
  Histogram(t) = MACD(t) − Signal(t)
  ```
- **Lookback:** Same as Signal Line

### Indicator 4 — Average True Range (ATR)

- **Input:** High, Low, Close
- **Formula:** Simple rolling mean of True Range
- **Lookback:** `atr_period + 1` bars
- **Parameters:**
  - `atr_period` (int, default `14`)

---

## Entry Signal

### Long Entry

All conditions must be true at bar close `t`:

1. `Histogram(t) > 0` — MACD histogram is positive (MACD Line above Signal Line)
2. `Histogram(t−1) ≤ 0` — histogram was non-positive on the previous bar (fresh zero-line crossover)
3. `Histogram(t) > Histogram(t−1)` — histogram is growing (momentum accelerating) — this is automatically true given conditions 1 and 2, but recorded explicitly for clarity
4. `ATR(t)` is not NaN
5. _(If vol filter active)_ `tick_vol(t) / SMA(tick_vol, vol_period)(t) ≥ vol_ratio_min`

**Execution:**
- **Price:** bar close of bar `t`
- **Bar:** signal bar `t`

### Short Entry

Mirror of Long Entry:

1. `Histogram(t) < 0`
2. `Histogram(t−1) ≥ 0`
3. `ATR(t)` is not NaN
4. _(If vol filter active)_ `vol_ratio(t) ≥ vol_ratio_min`

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| Volume confirmation | Off | Require elevated tick_vol at the crossover bar |
| MACD zero-line gate | Optional | Only enter longs when MACD Line > 0 (above the price zero-line) — adds trend direction confirmation |
| Session filter | Off | B3: 09:00–18:00 BRT recommended |
| Warm-up guard | Always active | All indicators must be fully initialized |

---

## Exit Signal

### Primary Exits — Price-Based

| Exit Type | Long | Short | Exit Price |
|-----------|------|-------|-----------|
| Stop Loss | `bar_low ≤ entry − sl_atr_mult × ATR_entry` | `bar_high ≥ entry + sl_atr_mult × ATR_entry` | SL level |
| Take Profit | `bar_high ≥ entry + tp_atr_mult × ATR_entry` | `bar_low ≤ entry − tp_atr_mult × ATR_entry` | TP level |

### Secondary Exits — Signal-Based

| Exit Type | Condition | Exit Price |
|-----------|-----------|-----------|
| Histogram reversal | `Histogram(t)` crosses back through zero in the opposite direction | Bar close |
| MACD deceleration exit | `|Histogram(t)| < |Histogram(t−1)|` for two consecutive bars (momentum slowing) AND `|Histogram(t)| < decel_threshold × ATR` | Bar close |
| Signal reversal | Opposite MACD crossover fires | Bar close |
| Session-end forced close | Last in-session bar | Bar close |
| End of data | Dataset ends with open position | Last close |

### Exit Priority

1. Stop Loss
2. Take Profit
3. Histogram reversal
4. MACD deceleration exit (if `use_decel_exit = True`)
5. Signal reversal
6. Session-end forced close

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset | **1** | No pyramiding |
| Stop type | **Fixed** | ATR-based, set at entry bar |
| Stop loss | `entry ± sl_atr_mult × ATR_entry` | Default: 2.0 × ATR |
| Take profit | `entry ± tp_atr_mult × ATR_entry` | Default: 3.5 × ATR |
| Trailing stop | **No** | Consider ATR trailing as extension |
| Default position sizing | **1 unit** | vol_scaled for cross-asset comparison |

### Markets

| Group | Assets | Notes |
|-------|--------|-------|
| Forex | EURUSD, EURCAD, GBPCHF | 4h and 1D most suitable; MACD is slow on 1h |
| B3 | WDO, WIN | 15min and 30min; standard MACD params (12/26/9) need rescaling for minute bars |
| Crypto | BTCUSDT, ETHUSDT | 4h and 1D; crypto trends often show strong MACD momentum |

### Time Restrictions

| Rule | Forex | B3 | Crypto |
|------|-------|----|--------|
| Session filter | None | 09:00–18:00 BRT | None (24/7) |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| `macd_fast` | `12` | `[8, 10, 12, 16]` | Fast EMA period |
| `macd_slow` | `26` | `[20, 24, 26, 30]` | Slow EMA period |
| `signal_period` | `9` | `[6, 9, 12]` | Signal line EMA period |
| `atr_period` | `14` | fixed | ATR period |
| `sl_atr_mult` | `2.0` | `[1.5, 2.0, 2.5, 3.0]` | Stop loss ATR multiple |
| `tp_atr_mult` | `3.5` | `[2.5, 3.0, 3.5, 4.0, 5.0]` | Take profit ATR multiple |
| `vol_period` | `20` | fixed | Tick volume SMA period |
| `vol_ratio_min` | `1.3` | `[1.0, 1.3, 1.5, 2.0]` | Min tick_vol ratio to confirm entry |
| `use_vol_filter` | `False` | — | Enable volume filter |
| `use_decel_exit` | `False` | — | Enable deceleration-based exit |
| `use_zero_line_gate` | `False` | — | Only enter longs when MACD > 0 |

**Constraints:** `macd_fast < macd_slow`

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | macd_fast | macd_slow | signal_period | sl_atr_mult | tp_atr_mult |
|-------|----|-------|-----------|-----------|---------------|-------------|-------------|
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
**Regime sensitivity:** Expected to perform well in trending periods; in ranging markets, histogram zero-line crossovers become frequent and generate many losses

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- Standard MACD parameters (12/26/9) were calibrated for daily bars; on minute or hourly data they should be rescaled proportionally or re-optimized
- MACD is a lagging indicator — by the time the histogram crosses zero, a significant portion of the move may have already occurred
- The deceleration exit can lead to premature exits in volatile trending moves where the histogram oscillates slightly while still above zero
- The zero-line gate (only trade in the direction of MACD vs zero) adds a higher-timeframe trend confirmation but also increases the lookback and warm-up requirements
- Consider combining with an ADX filter to limit entries to periods of genuine directional momentum

---

## Implementation

**Notebook:** `technical_analysis/04_macd_histogram_momentum.ipynb`
**Source module:** `source/strategy.py` — `MACDHistogramStrategy`
**Parameters class:** `MACDHistogramParams`

### Implementation Notes

- `use_decel_exit` is exposed but **disabled** — the deceleration exit
  (`|Histogram(t)| < |Histogram(t-1)|` for two consecutive bars) requires a
  custom exit hook on `Backtester` that isn't wired yet.
- `use_zero_line_gate` is implemented inside `generate_signals` as an entry
  filter only (suppressing entries against MACD's sign).
- `use_vol_filter` is disabled in the baseline so the notebook runs cleanly
  on datasets that may not carry `tick_vol`.
- The `macd_fast < macd_slow` constraint is enforced in
  `MACDHistogramParams.__post_init__`. The WFO grid is constructed so every
  combo satisfies it (max fast = 16 < min slow = 20).
- For B3, baseline params include `session_start=9`, `session_end=18`.
- Crypto is listed as a target market in the doc but the repo has no
  `data/crypto/` files; the notebook runs on Forex + B3 only.

---

## References

1. Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors*. FT Press. Original description of MACD.
2. Murphy, J.J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance. Chapter on MACD.
3. Kaufman, P.J. (2013). *Trading Systems and Methods* (5th ed.). Wiley. Chapter 7: Momentum and Oscillators.
4. Pring, M.J. (2002). *Technical Analysis Explained* (4th ed.). McGraw-Hill. Chapter on momentum indicators.
