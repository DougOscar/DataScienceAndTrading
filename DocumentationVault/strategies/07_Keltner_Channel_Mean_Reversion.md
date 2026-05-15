# Keltner Channel Mean Reversion

> **Type:** Mean-reversion
> **Markets:** Forex, Crypto
> **Timeframes:** 15min, 1h
> **Direction:** Long & Short
> **Status:** Idea

---

## Overview

Keltner Channels place ATR-based bands around an exponential moving average. Unlike Bollinger Bands (which use standard deviation), Keltner Channel width responds to Average True Range — a more robust measure of realized volatility that is less sensitive to single extreme bars. When price touches or crosses the outer band, it has moved an unusually large amount relative to recent average range, creating conditions for mean reversion back toward the EMA (middle band).

This strategy exploits short-term overextension in low-to-moderate trend environments. The middle EMA serves as the reversion target, providing a natural, volatility-adaptive take profit level. An ADX filter ensures mean-reversion trades are only taken in ranging markets where the EMA acts as a genuine attractor, not in strongly trending markets where price can remain outside the channel for extended periods.

---

## Indicators

### Indicator 1 — Exponential Moving Average (Middle Band)

- **Input:** Bar close prices
- **Formula:**
  ```
  KC_mid(t) = EMA(close, kc_period)[t]     (α = 2 / (kc_period + 1))
  ```
- **Lookback:** `kc_period` bars
- **Parameters:**
  - `kc_period` (int, default `20`) — EMA period for middle band

### Indicator 2 — Average True Range (ATR)

- **Input:** High, Low, Close
- **Formula:** Simple rolling mean of True Range over `atr_period` bars
- **Lookback:** `atr_period + 1` bars
- **Parameters:**
  - `atr_period` (int, default `10`) — Keltner ATR period (shorter than standard to be more reactive)

### Indicator 3 — Keltner Channel Bands

- **Input:** KC_mid, ATR
- **Formula:**
  ```
  KC_upper(t) = KC_mid(t) + kc_mult × ATR(t)
  KC_lower(t) = KC_mid(t) − kc_mult × ATR(t)
  ```
- **Parameters:**
  - `kc_mult` (float, default `2.0`) — band width multiplier

### Indicator 4 — ADX (Regime Filter)

- **Input:** High, Low, Close
- **Formula:** Wilder's ADX (same computation as described in Strategy 02)
- **Lookback:** `2 × adx_period` bars
- **Parameters:**
  - `adx_period` (int, default `14`)
  - `adx_max` (float, default `25`) — maximum ADX for entries to be allowed

### Indicator 5 — Stop ATR (for fixed stops)

- **Input:** High, Low, Close
- **Formula:** Simple rolling mean of True Range over `stop_atr_period` bars
- **Parameters:**
  - `stop_atr_period` (int, default `14`) — may differ from `atr_period` used for bands

---

## Entry Signal

### Long Entry

All conditions at bar close `t`:

1. `close(t) ≤ KC_lower(t)` — price closes at or below the lower Keltner Channel band
2. `ATR(t)` is not NaN
3. _(If ADX filter active)_ `ADX(t) < adx_max` — market is ranging, not trending strongly
4. No long position currently open

**Execution:**
- **Price:** bar close of bar `t`
- **Bar:** signal bar `t`

### Short Entry

Mirror of Long Entry:

1. `close(t) ≥ KC_upper(t)` — price closes at or above the upper band
2. `ATR(t)` is not NaN
3. _(If ADX filter active)_ `ADX(t) < adx_max`
4. No short position currently open

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| ADX regime filter | On | Suppress entries when `ADX ≥ adx_max` — key distinguisher from trend-following |
| Re-entry suppression | Optional | After a stop loss, suppress new entries in same direction for `reentry_cooldown` bars |
| Session filter | Off | For B3-correlated FX pairs: 09:00–18:00 BRT |
| Warm-up guard | Always active | EMA, ATR, and ADX must all be initialized |

---

## Exit Signal

### Primary Exits — Price-Based

Stop loss is fixed at entry. Take profit is the EMA middle band (dynamic, computed at exit evaluation each bar).

| Exit Type | Long | Short | Exit Price |
|-----------|------|-------|-----------|
| Stop Loss | `bar_low ≤ entry − sl_atr_mult × stop_ATR_entry` | `bar_high ≥ entry + sl_atr_mult × stop_ATR_entry` | SL level |
| Take Profit (EMA) | `bar_high ≥ KC_mid(t)` | `bar_low ≤ KC_mid(t)` | KC_mid(t) at exit bar |

The EMA-based take profit is dynamic: it moves each bar as the EMA evolves. The exit triggers when the bar's high (for long) touches or crosses the current EMA value.

### Secondary Exits — Signal-Based

| Exit Type | Condition | Exit Price |
|-----------|-----------|-----------|
| Opposite band touch | Long: `close(t) ≥ KC_upper(t)` (price reverses from lower band to upper) | Bar close |
| Signal reversal | Opposite Keltner band touch entry fires | Bar close |
| Session-end forced close | Last in-session bar | Bar close |
| End of data | Dataset ends with open position | Last close |

### Exit Priority

1. Stop Loss
2. Take Profit (EMA touch)
3. Opposite band touch (forced close)
4. Signal reversal
5. Session-end forced close

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset | **1** | |
| Stop type | **Fixed** (ATR at entry) | EMA TP is dynamic |
| Stop loss | `entry ± sl_atr_mult × stop_ATR_entry` | Default: 2.5 × ATR — wider than TP to allow reversion |
| Take profit | EMA touch (`KC_mid`) | Typical distance at entry: 1× to 2× ATR from entry price |
| Trailing stop | **No** | |
| Default position sizing | **1 unit** | vol_scaled recommended for multi-asset testing |

Note: The reward-to-risk ratio is variable here since the TP target (EMA) is fixed in terms of distance at entry time but the SL is also fixed. At entry, when price is at the band, the distance to the EMA is roughly `kc_mult × ATR`. The stop is `sl_atr_mult × ATR` on the other side, giving an effective RR of approximately `kc_mult / sl_atr_mult`. With defaults: 2.0 / 2.5 = 0.8 — less than 1:1. Win rate must therefore exceed 55% for profitability.

### Markets

| Group | Assets | Notes |
|-------|--------|-------|
| Forex | EURUSD, EURCAD, GBPCHF | 1h and 4h; ranging FX pairs preferred |
| Crypto | BTCUSDT, ETHUSDT | 15min and 1h; Crypto often has mean-reverting microstructure on short TFs |

### Time Restrictions

| Rule | Forex | Crypto |
|------|-------|--------|
| Session filter | None | None |
| News blackout | Not implemented | Not implemented |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| `kc_period` | `20` | `[10, 15, 20, 30]` | EMA period for middle band |
| `atr_period` | `10` | `[7, 10, 14]` | ATR period for channel width |
| `kc_mult` | `2.0` | `[1.5, 2.0, 2.5, 3.0]` | Keltner Channel multiplier |
| `stop_atr_period` | `14` | fixed | ATR period for stop loss computation |
| `sl_atr_mult` | `2.5` | `[1.5, 2.0, 2.5, 3.0]` | Stop loss ATR multiple |
| `adx_period` | `14` | fixed | ADX period |
| `adx_max` | `25` | `[20, 25, 30]` | Max ADX for entry to be allowed |
| `use_adx_filter` | `True` | — | Enable ADX regime filter (strongly recommended) |
| `reentry_cooldown` | `0` | `[0, 3, 5]` | Bars to wait after a stop loss before re-entering |

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | kc_period | kc_mult | sl_atr_mult | adx_max |
|-------|----|-------|-----------|---------|-------------|---------|
| | | | | | | |

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
**Regime sensitivity:** Performance is highly dependent on the ADX filter threshold. In trending market regimes without the ADX filter, losses can be severe. Expected win rate: 55–65% in ranging conditions.

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- Reward-to-risk ratio below 1:1 requires consistently high win rate; any degradation in regime detection rapidly turns the system unprofitable
- EMA take profit is dynamic — when the EMA is moving strongly in one direction, it may move away from the entry price, requiring a larger move to reach TP
- ADX is a lagging indicator; by the time ADX signals a trend, the trending move may be mostly complete
- In crypto, Keltner Band touches can be very frequent during volatile periods, leading to overtrading; a minimum cooldown between trades may be necessary
- Consider using an ATR-based take profit (fixed, simpler) rather than the EMA touch as an alternative for more consistent risk management

---

## Implementation

**Notebook:** `technical_analysis/07_keltner_channel_reversion.ipynb`
**Source module:** `source/strategy.py` — `KeltnerReversionStrategy`
**Parameters class:** `KeltnerReversionParams`

### Implementation Notes

- The doc's dynamic **EMA-touch take profit** is **approximated** by a fixed
  ATR-based TP (`tp_atr_mult × ATR_stop`) since the existing `Backtester`
  only supports fixed-at-entry SL/TP. Implementing true EMA-touch TP needs
  a custom exit hook.
- Two ATR periods are tracked: `atr_period` feeds the Backtester's SL/TP
  placement (the `atr` column), and `kc_atr_period` is used only for the
  Keltner channel width.
- ADX filter is **on** by default per the doc.
- `reentry_cooldown` from the doc is **not implemented** — would need a
  stateful signal-generation pass.
- Crypto group skipped — no `data/crypto/` files. B3 is not a target for
  this strategy in the doc, so the notebook runs on Forex only.

---

## References

1. Keltner, C.W. (1960). *How to Make Money in Commodities*. Keltner Statistical Service. Original channel concept.
2. Chester Kase popularized ATR-based Keltner Channels (replacing the original high-low midpoint version). See: Kase, C. (1993). *Trading with the Odds*. McGraw-Hill.
3. Wilder, J.W. (1978). *New Concepts in Technical Trading Systems*. Trend Research. ADX and ATR.
4. Kaufman, P.J. (2013). *Trading Systems and Methods* (5th ed.). Wiley. Chapter on channel mean reversion.
