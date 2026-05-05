# RSI Mean Reversion with ATR Stops

> **Type:** Mean-reversion
> **Markets:** Forex, Crypto
> **Timeframes:** 15min, 1h, 4h
> **Direction:** Long & Short
> **Status:** Idea

---

## Overview

The Relative Strength Index (RSI) is a bounded momentum oscillator that measures the speed and magnitude of recent price changes. When RSI crosses into extreme territory — above 70 (overbought) or below 30 (oversold) — it signals that the immediate move is overstretched. This strategy waits for RSI to re-cross back through those thresholds, treating the crossover as confirmation that the extreme has been rejected and a mean reversion is underway. ATR-based stops protect against the case where an extreme reading turns out to be the start of a genuine trend rather than an overreaction.

The hypothesis exploits the bounded nature of RSI in ranging or mildly trending markets. In strongly trending environments RSI can stay near extremes for extended periods, making the RSI midline (50) a secondary exit target and an ADX-based trend filter a key risk control.

---

## Indicators

### Indicator 1 — Relative Strength Index (RSI)

- **Input:** Bar close prices
- **Formula:**
  ```
  delta(t)  = close(t) − close(t−1)
  gain(t)   = max(delta(t), 0)
  loss(t)   = max(−delta(t), 0)

  avg_gain(t) = EMA(gain, period=rsi_period)[t]    (Wilder's smoothing: α = 1/rsi_period)
  avg_loss(t) = EMA(loss, period=rsi_period)[t]

  RS(t)  = avg_gain(t) / avg_loss(t)
  RSI(t) = 100 − 100 / (1 + RS(t))
  ```
  Wilder's EMA uses `α = 1/period`, i.e. `EMA(t) = EMA(t−1) × (1 − α) + x(t) × α`.
- **Lookback:** Requires `rsi_period + 1` bars for the first valid value (one bar for `delta`, then `rsi_period` bars to seed Wilder's EMA).
- **Parameters:**
  - `rsi_period` (int, default `14`) — Wilder smoothing period

### Indicator 2 — Average True Range (ATR)

- **Input:** High, Low, Close
- **Formula:**
  ```
  TR(t)  = max(high[t] − low[t],
               |high[t] − close[t−1]|,
               |low[t]  − close[t−1]|)
  ATR(t) = simple rolling mean of TR over atr_period bars
  ```
- **Lookback:** `atr_period + 1` bars
- **Parameters:**
  - `atr_period` (int, default `14`)

### Indicator 3 — ADX (Regime Filter, optional)

- **Input:** High, Low, Close
- **Formula:** Wilder's ADX — directional movement index averaged over `adx_period` bars.
  ```
  +DM(t) = max(high[t] − high[t−1], 0)  if > −DM, else 0
  −DM(t) = max(low[t−1] − low[t], 0)   if > +DM, else 0
  +DI    = 100 × Wilder_EMA(+DM) / ATR
  −DI    = 100 × Wilder_EMA(−DM) / ATR
  DX     = 100 × |+DI − −DI| / (+DI + −DI)
  ADX    = Wilder_EMA(DX, adx_period)
  ```
- **Lookback:** `2 × adx_period` bars (to stabilise both DM and DX smoothing)
- **Parameters:**
  - `adx_period` (int, default `14`)

---

## Entry Signal

### Long Entry

All conditions must be true simultaneously at bar close `t`:

1. `RSI(t) > rsi_lower` — RSI is now above the oversold threshold
2. `RSI(t−1) ≤ rsi_lower` — RSI was at or below the threshold on the previous bar (fresh crossover)
3. `ATR(t)` is not NaN
4. _(If ADX filter active)_ `ADX(t) < adx_max` — market is not strongly trending

**Execution:**
- **Price:** bar close of bar `t`
- **Bar:** signal bar `t`

### Short Entry

Mirror of Long Entry:

1. `RSI(t) < rsi_upper`
2. `RSI(t−1) ≥ rsi_upper`
3. `ATR(t)` is not NaN
4. _(If ADX filter active)_ `ADX(t) < adx_max`

**Execution:** bar close of bar `t`

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| ADX filter | Off | Suppress entries when `ADX > adx_max` (default 25) to avoid entering against strong trends |
| Session filter | Off | For Crypto: no session restriction. For B3-adjacent FX pairs: see session filter from strategy 01 |
| Warm-up guard | Always active | No signal until RSI, ATR, and optionally ADX have their required bars |

---

## Exit Signal

### Primary Exits — Price-Based

Stops are fixed at entry.

| Exit Type | Long | Short | Exit Price |
|-----------|------|-------|-----------|
| Stop Loss | `bar_low ≤ entry − sl_atr_mult × ATR_entry` | `bar_high ≥ entry + sl_atr_mult × ATR_entry` | SL level |
| Take Profit | `bar_high ≥ entry + tp_atr_mult × ATR_entry` | `bar_low ≤ entry − tp_atr_mult × ATR_entry` | TP level |

### Secondary Exits — Signal-Based

| Exit Type | Condition | Exit Price |
|-----------|-----------|-----------|
| RSI midline exit | RSI(t) crosses 50 in the direction of the trade (long: RSI crosses ≥ 50; short: RSI crosses ≤ 50) | Bar close |
| Signal reversal | Opposite RSI crossover entry signal fires | Bar close |
| Session-end forced close | Last bar of session with open position | Bar close |
| End of data | Dataset ends with open position | Last close |

### Exit Priority

1. Stop Loss
2. Take Profit
3. RSI midline exit (if `use_midline_exit = True`)
4. Signal reversal
5. Session-end forced close

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset | **1** | No pyramiding |
| Stop type | **Fixed** | Set at entry bar's ATR |
| Stop loss | `entry ± sl_atr_mult × ATR_entry` | Default: 1.5 × ATR |
| Take profit | `entry ± tp_atr_mult × ATR_entry` | Default: 2.0 × ATR |
| Trailing stop | **No** | |
| Default position sizing | **1 unit** | Or vol_scaled / fixed_frac as in strategy 01 |

### Markets

| Group | Assets | Notes |
|-------|--------|-------|
| Forex | EURUSD, EURCAD, GBPCHF | Ranging-biased pairs preferred |
| Crypto | BTCUSDT, ETHUSDT | 1h preferred; 15min has high noise |

### Time Restrictions

| Rule | Forex | Crypto |
|------|-------|--------|
| Session filter | None (24/5) | None (24/7) |
| News blackout | Not implemented | Not implemented |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| `rsi_period` | `14` | `[7, 10, 14, 20]` | RSI Wilder smoothing period |
| `rsi_lower` | `30` | `[25, 30, 35]` | Oversold threshold for long entry |
| `rsi_upper` | `70` | `[65, 70, 75]` | Overbought threshold for short entry |
| `atr_period` | `14` | fixed | ATR period |
| `sl_atr_mult` | `1.5` | `[1.0, 1.5, 2.0, 2.5]` | Stop loss ATR multiple |
| `tp_atr_mult` | `2.0` | `[1.5, 2.0, 3.0]` | Take profit ATR multiple |
| `adx_period` | `14` | fixed | ADX period |
| `adx_max` | `25` | `[20, 25, 30]` | Maximum ADX for entry to be allowed |
| `use_adx_filter` | `False` | — | Enable ADX regime filter |
| `use_midline_exit` | `True` | — | Exit at RSI 50 crossover |

**Constraint:** `rsi_lower < 50 < rsi_upper` must always hold.

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | rsi_period | rsi_lower | rsi_upper | sl_atr_mult | tp_atr_mult |
|-------|----|-------|------------|-----------|-----------|-------------|-------------|
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
**Regime sensitivity:** Expected to perform well in ranging / low-ADX conditions; deteriorate in strong trends

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- RSI can remain in extreme territory for many bars during strong trends, generating multiple false entries
- Fixed ATR stop may be too wide for short timeframes where RSI extremes are common
- RSI threshold symmetry (30/70) may not be optimal — asymmetric thresholds could be tested
- Midline exit (RSI=50) often triggers too early and leaves profit on the table; consider trailing stop as alternative
- No volume confirmation — a tick_vol filter (e.g. entry only when tick_vol is above its rolling median) could improve signal quality

---

## Implementation

**Notebook:** `technical_analysis/02_rsi_mean_reversion.ipynb`
**Source module:** `source/strategy.py` — `RSIMeanReversionStrategy`
**Parameters class:** `StrategyParams`

---

## References

1. Wilder, J.W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
2. Colby, R.W. (2003). *The Encyclopedia of Technical Market Indicators* (2nd ed.). McGraw-Hill. Chapter on RSI.
3. Lo, A.W., Mamaysky, H., & Wang, J. (2000). Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation. *Journal of Finance*, 55(4), 1705–1765.
