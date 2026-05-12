# EMA Ribbon with RSI Trend Filter

> **Type:** Trend-following
> **Markets:** Forex, B3, Crypto
> **Timeframes:** 15min, 1h, 4h
> **Direction:** Long & Short
> **Status:** Idea

---

## Overview

An EMA Ribbon uses multiple exponential moving averages of increasing period to visualize trend strength and alignment. When all EMAs are stacked in order (fastest above slowest for an uptrend), it indicates that trend momentum is broadly consistent across multiple time horizons. The crossover of the fastest EMA through the slowest EMA acts as a compact entry signal, and full ribbon alignment serves as an ongoing position-holding condition.

An RSI filter prevents entering in overbought or oversold extremes, which often coincide with trend exhaustion, reducing the entry to areas where trend momentum exists but the price has not yet overextended. A volume confirmation filter (tick_vol) improves signal quality by ensuring the crossover is accompanied by genuine market participation.

---

## Indicators

### Indicator 1 — EMA Ribbon (4 levels)

- **Input:** Bar close prices
- **Formula:**
  ```
  EMA_1(t) = EMA(close, ema_p1)[t]      (fastest)
  EMA_2(t) = EMA(close, ema_p2)[t]
  EMA_3(t) = EMA(close, ema_p3)[t]
  EMA_4(t) = EMA(close, ema_p4)[t]      (slowest)
  ```
  Standard EMA: `EMA(t) = close(t) × α + EMA(t−1) × (1 − α)` where `α = 2 / (period + 1)`
- **Lookback:** `ema_p4` bars (the slowest EMA dominates)
- **Parameters:**
  - `ema_p1` (int, default `5`) — fastest EMA period
  - `ema_p2` (int, default `8`)
  - `ema_p3` (int, default `13`)
  - `ema_p4` (int, default `21`) — slowest EMA period
  - Fibonacci-sequence defaults (5, 8, 13, 21) are conventional and evenly spaced in log-period space

### Indicator 2 — Ribbon Alignment Score

- **Input:** EMA_1, EMA_2, EMA_3, EMA_4
- **Formula:**
  ```
  bullish_aligned(t) = 1 if EMA_1(t) > EMA_2(t) > EMA_3(t) > EMA_4(t), else 0
  bearish_aligned(t) = 1 if EMA_1(t) < EMA_2(t) < EMA_3(t) < EMA_4(t), else 0
  ```
  No partial values — strict monotonic ordering required.
- **Lookback:** Same as EMA Ribbon

### Indicator 3 — RSI (Trend Filter)

- **Input:** Bar close prices
- **Formula:** Wilder's RSI (same as Strategy 02)
- **Lookback:** `rsi_period + 1` bars
- **Parameters:**
  - `rsi_period` (int, default `14`)
  - `rsi_overbought` (float, default `65`) — entry suppressed for longs above this
  - `rsi_oversold` (float, default `35`) — entry suppressed for shorts below this

### Indicator 4 — Average True Range (ATR)

- **Input:** High, Low, Close
- **Formula:** Simple rolling mean of True Range
- **Lookback:** `atr_period + 1` bars
- **Parameters:**
  - `atr_period` (int, default `14`)

---

## Entry Signal

### Long Entry

All conditions at bar close `t`:

1. `bullish_aligned(t) = 1` — full ribbon aligned bullish
2. `bullish_aligned(t−1) = 0` — ribbon was NOT fully aligned on the previous bar (fresh alignment event)
3. `RSI(t) < rsi_overbought` — not overbought
4. `ATR(t)` is not NaN
5. _(Optional)_ `EMA_1(t−1) ≤ EMA_4(t−1)` AND `EMA_1(t) > EMA_4(t)` — additionally require the fastest EMA to have just crossed the slowest (tighter entry trigger)

Condition 2 ensures the signal fires only at the moment of ribbon alignment, not on every bar while aligned.

**Execution:**
- **Price:** bar close of bar `t`
- **Bar:** signal bar `t`

### Short Entry

Mirror of Long Entry:

1. `bearish_aligned(t) = 1`
2. `bearish_aligned(t−1) = 0`
3. `RSI(t) > rsi_oversold`
4. `ATR(t)` is not NaN

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| RSI overbought/oversold suppression | Always active (if rsi_period is set) | Prevent entries at trend exhaustion levels |
| Strict EMA_1/EMA_4 crossover gate | Off | Require EMA_1 to cross EMA_4 in the same bar as full alignment |
| Session filter | Off | B3: 09:00–18:00 BRT recommended |
| Warm-up guard | Always active | All four EMAs and RSI must be initialized |

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
| Ribbon collapse | `bullish_aligned(t) = 0` for long / `bearish_aligned(t) = 0` for short — ribbon is no longer fully aligned | Bar close |
| Signal reversal | Opposite ribbon alignment signal fires | Bar close |
| Session-end forced close | Last in-session bar | Bar close |
| End of data | Dataset ends with open position | Last close |

### Exit Priority

1. Stop Loss
2. Take Profit
3. Ribbon collapse exit (if `use_ribbon_exit = True`)
4. Signal reversal
5. Session-end forced close

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset | **1** | |
| Stop type | **Fixed** | ATR-based at entry |
| Stop loss | `entry ± sl_atr_mult × ATR_entry` | Default: 2.0 × ATR |
| Take profit | `entry ± tp_atr_mult × ATR_entry` | Default: 3.0 × ATR |
| Trailing stop | **No** | Ribbon collapse exit acts as trailing mechanism |
| Default position sizing | **1 unit** | |

### Markets

| Group | Assets | Notes |
|-------|--------|-------|
| Forex | EURUSD, EURCAD, GBPCHF | 1h and 4h preferred |
| B3 | WDO, WIN | 15min and 1h; session filter recommended |
| Crypto | BTCUSDT, ETHUSDT | 1h and 4h; crypto trends align the ribbon clearly during bull/bear cycles |

### Time Restrictions

| Rule | Forex | B3 | Crypto |
|------|-------|----|--------|
| Session filter | None | 09:00–18:00 BRT | None |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| `ema_p1` | `5` | `[3, 5, 8]` | Fastest EMA period |
| `ema_p2` | `8` | `[6, 8, 10, 13]` | Second EMA period |
| `ema_p3` | `13` | `[10, 13, 15, 21]` | Third EMA period |
| `ema_p4` | `21` | `[18, 21, 30, 34]` | Slowest EMA period |
| `rsi_period` | `14` | `[10, 14]` | RSI period |
| `rsi_overbought` | `65` | `[60, 65, 70]` | RSI upper gate for long entry |
| `rsi_oversold` | `35` | `[30, 35, 40]` | RSI lower gate for short entry |
| `atr_period` | `14` | fixed | ATR period |
| `sl_atr_mult` | `2.0` | `[1.5, 2.0, 2.5]` | Stop loss ATR multiple |
| `tp_atr_mult` | `3.0` | `[2.0, 3.0, 4.0]` | Take profit ATR multiple |
| `use_ribbon_exit` | `True` | — | Exit on ribbon de-alignment |

**Constraints:** `ema_p1 < ema_p2 < ema_p3 < ema_p4`

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | ema_p1 | ema_p2 | ema_p3 | ema_p4 | sl_atr_mult | tp_atr_mult |
|-------|----|-------|--------|--------|--------|--------|-------------|-------------|
| | | | | | | | | |

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
**Regime sensitivity:** Full ribbon alignment is rare in choppy markets (reducing trade frequency) but may miss the early stages of a trend. In strong trends, ribbon stays aligned and generates a single well-timed entry followed by a clean ribbon collapse exit.

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- Four EMA periods introduce four free parameters plus RSI; this is a high-dimensional search space for WFO and risks overfitting
- The Fibonacci default (5, 8, 13, 21) is conventional but not mathematically motivated for financial data; reducing to two or three EMAs may be more robust
- Ribbon collapse exit can be noisy in moderately volatile markets where the ribbon frequently goes in and out of full alignment; ATR trailing stop may be more stable
- RSI filter adds an additional parameter dependency; consider fixing RSI thresholds based on literature values (30/70 or 35/65) rather than optimizing them

---

## Implementation

**Notebook:** `technical_analysis/06_ema_ribbon_rsi_filter.ipynb`
**Source module:** `source/strategy.py` — `EMARibbonStrategy`
**Parameters class:** `EMARibbonParams`

### Implementation Notes

- Only the **fresh-alignment** entry trigger is implemented; the ribbon-
  collapse exit (`use_ribbon_exit`) would require a custom `Backtester` exit
  hook and is left **disabled**.
- The strict EMA ordering `ema_p1 < ema_p2 < ema_p3 < ema_p4` is enforced
  inside `generate_signals` (invalid combos emit zero signals) so the WFO
  grid iterator works.
- WFO grid is **trimmed** vs. the full doc grid (576 combos vs ~31k); every
  combo satisfies the monotonic ordering by construction.
- For B3, baseline params include `session_start=9`, `session_end=18`.
- Crypto group skipped — no `data/crypto/` files.

---

## References

1. Murphy, J.J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance. Chapter on moving averages.
2. Elder, A. (1993). *Trading for a Living*. Wiley. Original description of multiple EMA alignment as a trend confirmation tool.
3. Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors*. FT Press. Chapter on exponential moving averages.
4. Wilder, J.W. (1978). *New Concepts in Technical Trading Systems*. Trend Research. RSI methodology.
