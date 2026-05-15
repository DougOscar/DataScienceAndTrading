# Dual Thrust Intraday Breakout

> **Type:** Breakout / Intraday
> **Markets:** B3, Crypto
> **Timeframes:** 1min, 5min, 15min (intraday)
> **Direction:** Long & Short
> **Status:** Idea

---

## Overview

The Dual Thrust system defines session breakout levels each day using the historical range of the past N sessions. A "thrust" amount is computed from this range and added/subtracted from the session's opening price to create upper and lower trigger levels. Once price breaks through a trigger intraday, the strategy enters in that direction.

The key insight is that the range is computed from the maximum of two sub-ranges — `HH−LC` and `HC−LL` — which captures both gap-adjusted range and directional price drift, making the triggers more adaptive to recent market character than a simple high-low range. The system was popularized in the algorithmic trading community through its application to equity index futures and has demonstrated robust behavior across multiple asset classes including Forex futures and crypto.

The strategy is strictly intraday: all positions are closed at session end regardless of PnL.

---

## Indicators

### Indicator 1 — Dual Thrust Range

- **Input:** Daily (session) High, Low, Open, Close prices
  - Requires a daily aggregation layer on top of intraday OHLC data
  - Each session's daily OHLC is computed from the intraday bars (open = first bar open, high = max of highs, low = min of lows, close = last bar close)
- **Formula:**
  ```
  For the N sessions preceding today (sessions t−1 through t−N):
  
  HH  = max(high[t−1], high[t−2], …, high[t−N])   (N-session highest high)
  HC  = max(close[t−1], close[t−2], …, close[t−N]) (N-session highest close)
  LL  = min(low[t−1], low[t−2], …, low[t−N])       (N-session lowest low)
  LC  = min(close[t−1], close[t−2], …, close[t−N]) (N-session lowest close)
  
  Range = max(HH − LC, HC − LL)
  ```
  The range is computed fresh each session from the prior N sessions' data.
- **Lookback:** N prior sessions of daily OHLC data
- **Parameters:**
  - `dt_lookback` (int, default `4`) — number of prior sessions used (N)

### Indicator 2 — Daily Trigger Levels

- **Input:** Range (from Indicator 1), current session's open price
- **Formula:**
  ```
  Open_today = open price of today's first intraday bar
  
  Upper_trigger = Open_today + k1 × Range
  Lower_trigger = Open_today − k2 × Range
  ```
  These levels are fixed for the entire session once the opening bar is available.
- **Parameters:**
  - `k1` (float, default `0.5`) — upper trigger multiplier
  - `k2` (float, default `0.5`) — lower trigger multiplier (can differ from k1 for asymmetric triggers)

### Indicator 3 — Average True Range (ATR)

- **Input:** High, Low, Close (intraday)
- **Formula:** Simple rolling mean of True Range over `atr_period` bars
- **Lookback:** `atr_period + 1` bars
- **Parameters:**
  - `atr_period` (int, default `14`) — intraday bars

---

## Entry Signal

### Long Entry

At intraday bar close `t`:

1. `close(t) > Upper_trigger` — intraday close exceeds the upper trigger level
2. No long position currently open (one position per session; no re-entry after exit)
3. Bar `t` is within the session window (`session_start ≤ bar_hour < session_end − entry_cutoff`)
4. `ATR(t)` is not NaN

**Entry cutoff:** Do not enter in the last `entry_cutoff_minutes` of the session to avoid forced immediate closeout.

**Execution:**
- **Price:** bar close of bar `t`
- **Bar:** signal bar `t`
- If a short position is open, it is closed at this bar's close (reversal)

### Short Entry

Mirror of Long Entry:

1. `close(t) < Lower_trigger`
2. No short position currently open
3. Bar within session window

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| Session window | Required | Only enter within `[session_start, session_end − entry_cutoff)` |
| Entry cutoff | 30 min before session end | Prevent entries too late to allow profitable development |
| One-direction-per-session | Off | Optionally: once triggered long, do not re-enter short the same session even after a stop loss |
| Warm-up guard | Always active | Requires N prior sessions of data for Range computation |

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
| Opposite trigger | Intraday price crosses the opposite trigger level | Bar close |
| Session-end forced close | Bar `t` hour = `session_end` − 1 (last session bar) | Bar close |
| End of data | Dataset ends with open position | Last close |

### Exit Priority

1. Stop Loss
2. Take Profit
3. Opposite trigger hit (reversal)
4. Session-end forced close

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset | **1** | No pyramiding |
| Max trades per session per asset | **2** (optional) | One initial entry + one reversal after trigger cross |
| Stop type | **Fixed** | ATR-based at entry |
| Stop loss | `entry ± sl_atr_mult × ATR_entry` | Default: 1.5 × ATR |
| Take profit | `entry ± tp_atr_mult × ATR_entry` | Default: 2.5 × ATR |
| Session-end forced close | Always | All positions closed at session end |
| Default position sizing | **1 unit** | vol_scaled for cross-asset comparison |

### Markets

| Group | Assets | Notes |
|-------|--------|-------|
| B3 | WDO (USD/BRL mini), WIN (Bovespa mini) | Primary market; strong intraday trends; session 09:00–17:55 BRT |
| Crypto | BTCUSDT, ETHUSDT | 24h market — define artificial "sessions" (e.g. 00:00–23:59 UTC) and compute daily opens |

### Time Restrictions

| Rule | B3 | Crypto |
|------|-----|--------|
| Session definition | 09:00–18:00 BRT | UTC daily sessions (00:00–00:00) |
| Entry cutoff | 17:30 BRT (last 30 min) | 23:30 UTC |
| Days of week | Mon–Fri only | Mon–Sun |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| `dt_lookback` | `4` | `[2, 4, 6, 8]` | Prior sessions used to compute Range |
| `k1` | `0.5` | `[0.3, 0.4, 0.5, 0.6, 0.7]` | Upper trigger multiplier |
| `k2` | `0.5` | `[0.3, 0.4, 0.5, 0.6, 0.7]` | Lower trigger multiplier |
| `atr_period` | `14` | fixed | Intraday ATR period |
| `sl_atr_mult` | `1.5` | `[1.0, 1.5, 2.0, 2.5]` | Stop loss ATR multiple |
| `tp_atr_mult` | `2.5` | `[2.0, 2.5, 3.0, 4.0]` | Take profit ATR multiple |
| `session_start` | `9` | fixed | Session start hour (BRT for B3) |
| `session_end` | `18` | fixed | Session end hour (exclusive) |
| `entry_cutoff_minutes` | `30` | `[15, 30, 60]` | Minutes before session end when new entries are blocked |

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | dt_lookback | k1 | k2 | sl_atr_mult | tp_atr_mult |
|-------|----|-------|-------------|----|----|-------------|-------------|
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
**Regime sensitivity:** Expected to perform best during trending intraday days (directional openings); loses on range-bound sessions where both triggers may be hit before meaningful follow-through

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- The `Range` computation requires daily OHLC data to be aligned correctly with intraday bars; an incorrect daily session aggregation can produce look-ahead bias in the trigger levels
- `k1 = k2 = 0.5` is symmetric; in practice, one direction may have better statistical properties on specific instruments (e.g., WIN tends to gap up more than down) — asymmetric k values should be explored
- No gap filter: if today opens with a very large gap, the trigger levels may be placed far from recent consolidation and the strategy may never trigger or trigger inappropriately
- Forced session-end close limits the ability to capture overnight gap moves; extending to next-session continuation is an alternative variant

---

## Implementation

**Notebook:** `technical_analysis/09_dual_thrust_intraday.ipynb`
**Source module:** `source/strategy.py` — `DualThrustStrategy`
**Parameters class:** `DualThrustParams`

### Implementation Notes

- Daily OHLC is computed from **in-session bars only** (`session_start` /
  `session_end` define the window). The current session's triggers come from
  the prior `dt_lookback` sessions, shifted by one to avoid look-ahead.
- The Backtester's existing `session_start` / `session_end` handling provides
  the **session-end forced close** referenced in the doc — `DualThrustParams`
  defaults `session_start=9`, `session_end=18` (B3).
- `entry_cutoff_minutes` is honoured by suppressing entries when the bar is
  within `cutoff` minutes of `session_end`.
- The notebook targets B3 on 5min and 15min. The doc's 1min timeframe is
  **dropped** from the notebook for WFO cost reasons — the user can add it
  back by editing `GROUP_TIMEFRAMES` in §2.
- Crypto group skipped — no `data/crypto/` files.

---

## References

1. Chalek, M. (1994). Original Dual Thrust system concept. Documented in: Kaufman, P.J. (2013). *Trading Systems and Methods* (5th ed.). Wiley. Chapter on opening range breakout systems.
2. Liu, M., Zheng, W., & others (multiple practitioners). "Dual Thrust Trading Algorithm" [widely circulated in Chinese quantitative trading community, 2010s]. Formalization of Chalek's original concept with the `max(HH−LC, HC−LL)` range definition.
3. Tomasini, E., & Jaekle, U. (2009). *Trading Systems: A New Approach to System Development and Portfolio Optimisation*. Harriman House. Chapter on intraday breakout systems.
4. Kaufman, P.J. (2013). *Trading Systems and Methods* (5th ed.). Wiley. Chapter 11: Day Trading.
