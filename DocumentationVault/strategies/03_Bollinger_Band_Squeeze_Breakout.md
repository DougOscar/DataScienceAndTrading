# Bollinger Band Squeeze Breakout

> **Type:** Breakout / Volatility expansion
> **Markets:** Forex, B3, Crypto
> **Timeframes:** 1h, 4h, 1D
> **Direction:** Long & Short
> **Status:** Idea

---

## Overview

Bollinger Bands contract during low-volatility consolidation periods as the price distribution narrows. This contraction — a "squeeze" — stores energy that is typically released in a sharp directional move when volatility expands. The strategy identifies squeeze conditions using Bollinger Band Width relative to its own rolling minimum, then waits for a confirmed breakout above the upper band or below the lower band before entering.

The core idea exploits the predictable volatility cycle: prolonged compression is followed by expansion, and the direction of the first strong breakout bar tends to persist. Keltner Channels provide an alternative squeeze definition with cleaner signal properties. A volume (tick_vol) surge filter is used to confirm genuine breakouts from false ones.

---

## Indicators

### Indicator 1 — Bollinger Bands (BB)

- **Input:** Bar close prices
- **Formula:**
  ```
  BB_mid(t)   = SMA(close, bb_period)[t]
  BB_std(t)   = rolling std(close, bb_period)[t]      (sample std, ddof=1)
  BB_upper(t) = BB_mid(t) + bb_mult × BB_std(t)
  BB_lower(t) = BB_mid(t) − bb_mult × BB_std(t)
  BB_width(t) = (BB_upper(t) − BB_lower(t)) / BB_mid(t)   (normalized)
  ```
- **Lookback:** `bb_period` bars
- **Parameters:**
  - `bb_period` (int, default `20`)
  - `bb_mult` (float, default `2.0`)

### Indicator 2 — Squeeze Indicator

- **Input:** `BB_width` series
- **Formula:**
  ```
  squeeze_min(t) = rolling min(BB_width, squeeze_lookback)[t]
  squeeze(t)     = 1 if BB_width(t) == squeeze_min(t), else 0
  ```
  A bar is "in squeeze" when the current BB Width equals the lowest BB Width seen in the past `squeeze_lookback` bars — i.e., volatility is at a local minimum.
- **Lookback:** `bb_period + squeeze_lookback` bars total
- **Parameters:**
  - `squeeze_lookback` (int, default `20`) — rolling window for finding the minimum BB Width

### Indicator 3 — Average True Range (ATR)

- **Input:** High, Low, Close
- **Formula:** Simple rolling mean of True Range (same as Strategy 01)
- **Lookback:** `atr_period + 1` bars
- **Parameters:**
  - `atr_period` (int, default `14`)

### Indicator 4 — Tick Volume Ratio (optional)

- **Input:** `tick_vol` column
- **Formula:**
  ```
  tick_vol_sma(t) = SMA(tick_vol, vol_period)[t]
  vol_ratio(t)    = tick_vol(t) / tick_vol_sma(t)
  ```
- **Lookback:** `vol_period` bars
- **Parameters:**
  - `vol_period` (int, default `20`)
  - `vol_ratio_min` (float, default `1.5`) — minimum vol_ratio to confirm breakout

---

## Entry Signal

### Long Entry

All conditions must be true at bar `t`:

1. `squeeze(t−1) = 1` OR `squeeze(t−2) = 1` — a squeeze was active on one of the last two bars (breakout is happening now)
2. `close(t) > BB_upper(t)` — price closes above the upper Bollinger Band
3. `ATR(t)` is not NaN
4. _(If vol filter active)_ `vol_ratio(t) ≥ vol_ratio_min` — breakout is confirmed by elevated tick volume

**Execution:**
- **Price:** bar close of bar `t`
- **Bar:** signal bar `t`

### Short Entry

Mirror of Long Entry:

1. `squeeze(t−1) = 1` OR `squeeze(t−2) = 1`
2. `close(t) < BB_lower(t)` — price closes below the lower Bollinger Band
3. `ATR(t)` is not NaN
4. _(If vol filter active)_ `vol_ratio(t) ≥ vol_ratio_min`

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| Volume confirmation | Off | Require `vol_ratio ≥ vol_ratio_min` to suppress low-volume false breakouts |
| Squeeze recency window | Always active | Squeeze must have occurred within last 2 bars; stale squeezes (>2 bars ago) do not count |
| Session filter | Off | B3: `session_start=9, session_end=18` recommended |
| Warm-up guard | Always active | No signal until all indicators have their required bars |

---

## Exit Signal

### Primary Exits — Price-Based

Stops fixed at entry.

| Exit Type | Long | Short | Exit Price |
|-----------|------|-------|-----------|
| Stop Loss | `bar_low ≤ entry − sl_atr_mult × ATR_entry` | `bar_high ≥ entry + sl_atr_mult × ATR_entry` | SL level |
| Take Profit | `bar_high ≥ entry + tp_atr_mult × ATR_entry` | `bar_low ≤ entry − tp_atr_mult × ATR_entry` | TP level |

### Secondary Exits — Signal-Based

| Exit Type | Condition | Exit Price |
|-----------|-----------|-----------|
| Band re-entry | Price closes back inside the band it broke out from (`close < BB_upper` for long / `close > BB_lower` for short) | Bar close |
| Signal reversal | Opposite breakout signal fires | Bar close |
| Session-end forced close | Last in-session bar with open position | Bar close |
| End of data | Dataset ends with open position | Last close |

### Exit Priority

1. Stop Loss
2. Take Profit
3. Band re-entry exit (if `use_band_reentry_exit = True`)
4. Signal reversal
5. Session-end forced close

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset | **1** | No pyramiding |
| Stop type | **Fixed** | ATR-based, set at entry |
| Stop loss | `entry ± sl_atr_mult × ATR_entry` | Default: 2.0 × ATR |
| Take profit | `entry ± tp_atr_mult × ATR_entry` | Default: 3.0 × ATR |
| Trailing stop | **No** | |
| Default position sizing | **1 unit** | vol_scaled recommended for cross-asset comparison |

### Markets

| Group | Assets | Notes |
|-------|--------|-------|
| Forex | EURUSD, EURCAD, GBPCHF | 4h and 1D preferred to reduce false breakouts |
| B3 | WDO, WIN | Session filter recommended; 1h is primary timeframe |
| Crypto | BTCUSDT, ETHUSDT | 4h and 1D; crypto volatility makes squeeze detection noisier on lower TFs |

### Time Restrictions

| Rule | Forex | B3 | Crypto |
|------|-------|----|--------|
| Session filter | None | 09:00–18:00 BRT | None (24/7) |
| Days of week | None tested | None tested | None tested |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| `bb_period` | `20` | `[10, 15, 20, 30]` | Bollinger Band SMA period |
| `bb_mult` | `2.0` | `[1.5, 2.0, 2.5]` | Bollinger Band standard deviation multiplier |
| `squeeze_lookback` | `20` | `[10, 15, 20, 30]` | Rolling window for identifying BB Width minimum |
| `atr_period` | `14` | fixed | ATR period |
| `sl_atr_mult` | `2.0` | `[1.0, 1.5, 2.0, 2.5, 3.0]` | Stop loss ATR multiple |
| `tp_atr_mult` | `3.0` | `[2.0, 3.0, 4.0, 5.0]` | Take profit ATR multiple |
| `vol_period` | `20` | fixed | Tick volume SMA period |
| `vol_ratio_min` | `1.5` | `[1.2, 1.5, 2.0]` | Min tick_vol ratio to confirm breakout |
| `use_vol_filter` | `False` | — | Enable volume confirmation filter |
| `use_band_reentry_exit` | `False` | — | Exit when price closes back inside the broken band |

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | bb_period | bb_mult | squeeze_lookback | sl_atr_mult | tp_atr_mult |
|-------|----|-------|-----------|---------|------------------|-------------|-------------|
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
**Regime sensitivity:** Expected to perform best during volatility expansion cycles; will generate false signals in continuously trending markets where BB Width never contracts

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- The "squeeze = BB Width at rolling minimum" definition can trigger in many bars during prolonged low-vol periods, leading to premature entries
- False breakouts (price briefly exits the band then immediately reverses) are the main source of losses; the band re-entry exit attempts to mitigate this but introduces whipsaw risk
- In high-volatility assets (Crypto), Bollinger Bands may never truly squeeze, making the indicator less discriminating
- Squeeze lookback and BB period interact strongly — their ratio should be considered as a single parameter during WFO
- Keltner Channel squeeze definition (BB inside KC) is an alternative that may reduce false signals

---

## Implementation

**Notebook:** `technical_analysis/03_bb_squeeze_breakout.ipynb`
**Source module:** `source/strategy.py` — `BBSqueezeStrategy`
**Parameters class:** `BBSqueezeParams`

### Implementation Notes

- `use_band_reentry_exit` is exposed but **disabled** in the baseline — the
  band-re-entry exit cannot be expressed cleanly through the existing
  `Backtester`'s SL / TP / signal-reversal contract; adding a custom-exit
  hook is a follow-up.
- `use_vol_filter` is disabled in the baseline so the notebook runs cleanly
  on datasets that may not carry `tick_vol`.
- For B3, baseline params include `session_start=9`, `session_end=18` (per the
  doc's "Time Restrictions" table).
- Crypto is listed as a target market in this doc but the repo has no
  `data/crypto/` files; the notebook runs on Forex + B3 only and prints a
  warning.

---

## References

1. Bollinger, J. (2001). *Bollinger on Bollinger Bands*. McGraw-Hill.
2. Carter, J. (2005). *Mastering the Trade*. McGraw-Hill. Chapter on Bollinger Squeeze.
3. Durenard, E.A. (2013). *Professional Automated Trading: Theory and Practice*. Wiley. Chapter on volatility breakout systems.
4. Lazybear (2014). *TTM Squeeze indicator* [TradingView script]. Adaptation of Carter's squeeze using Keltner Channel as squeeze reference.
