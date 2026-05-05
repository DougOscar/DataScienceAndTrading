# Donchian Channel Trend Following

> **Type:** Trend-following / Breakout
> **Markets:** Forex, B3, Crypto
> **Timeframes:** 4h, 1D
> **Direction:** Long & Short
> **Status:** Idea

---

## Overview

The Donchian Channel defines breakout levels as the highest high and lowest low over a rolling lookback window. When price closes above the N-period high it signals a new upside breakout — implying that all sellers who entered in the past N bars are now underwater and the breakout represents genuine new demand. This is the core principle behind the famous Turtle Trading system.

This implementation uses a two-channel approach: a wider entry channel (N1 bars) for trade entry and a narrower exit channel (N2 bars, N2 < N1) as a trailing stop mechanism. The entry channel filters out minor breakouts; the exit channel allows the trade to stay open while the trend persists, but closes it when price falls back to a more recent lower low (for longs) or higher high (for shorts). ATR-based stops serve as catastrophic loss protection.

---

## Indicators

### Indicator 1 — Donchian Entry Channel

- **Input:** High and Low prices
- **Formula:**
  ```
  DC_high_entry(t) = max(high[t−dc_entry+1], …, high[t−1])    (excludes current bar)
  DC_low_entry(t)  = min(low[t−dc_entry+1],  …, low[t−1])
  ```
  The current bar `t` is excluded from the calculation to avoid look-ahead: entry occurs when the current bar's close exceeds the prior N-bar extreme.
- **Lookback:** `dc_entry` bars (prior bars only)
- **Parameters:**
  - `dc_entry` (int, default `20`) — breakout channel period in bars

### Indicator 2 — Donchian Exit Channel

- **Input:** High and Low prices
- **Formula:**
  ```
  DC_high_exit(t) = max(high[t−dc_exit+1], …, high[t−1])
  DC_low_exit(t)  = min(low[t−dc_exit+1],  …, low[t−1])
  ```
  Same construction as entry channel with shorter period.
- **Lookback:** `dc_exit` bars
- **Parameters:**
  - `dc_exit` (int, default `10`) — exit channel period (must be < dc_entry)

### Indicator 3 — Average True Range (ATR)

- **Input:** High, Low, Close
- **Formula:** Simple rolling mean of True Range
- **Lookback:** `atr_period + 1` bars
- **Parameters:**
  - `atr_period` (int, default `14`)

---

## Entry Signal

### Long Entry

All conditions must hold at bar close `t`:

1. `close(t) > DC_high_entry(t)` — close exceeds the N1-period prior high
2. No long position currently open
3. `ATR(t)` is not NaN
4. _(If session filter active)_ bar within session window

This is a strict close-based breakout: the closing price must exceed the prior-bar Donchian high. Intrabar highs that exceed the level but close back below do not trigger.

**Execution:**
- **Price:** bar close of bar `t`
- **Bar:** signal bar `t`
- If a short position is open, it is closed at the same close price before the long is entered (reversal)

### Short Entry

Mirror of Long Entry:

1. `close(t) < DC_low_entry(t)` — close breaks below the N1-period prior low
2. No short position currently open
3. `ATR(t)` is not NaN

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| Session filter | Off | B3: 09:00–18:00 BRT; do not enter breakouts in the last 30 min of session |
| Warm-up guard | Always active | Both Donchian channels and ATR must be initialized |

---

## Exit Signal

### Primary Exits — Price-Based

| Exit Type | Long | Short | Exit Price |
|-----------|------|-------|-----------|
| Stop Loss (catastrophic) | `bar_low ≤ entry − sl_atr_mult × ATR_entry` | `bar_high ≥ entry + sl_atr_mult × ATR_entry` | SL level |
| Take Profit (optional) | `bar_high ≥ entry + tp_atr_mult × ATR_entry` | `bar_low ≤ entry − tp_atr_mult × ATR_entry` | TP level |

ATR stop is set at entry and remains fixed. The Donchian exit channel acts as the primary trend-following stop (see below).

### Secondary Exits — Signal-Based

| Exit Type | Condition | Exit Price |
|-----------|-----------|-----------|
| Donchian channel exit | Long: `close(t) < DC_low_exit(t)` — price closes below N2-period prior low | Bar close |
| Donchian channel exit | Short: `close(t) > DC_high_exit(t)` — price closes above N2-period prior high | Bar close |
| Signal reversal | Opposite breakout signal fires (reversal) | Bar close |
| Session-end forced close | Last in-session bar with open position | Bar close |
| End of data | Dataset ends with open position | Last close |

### Exit Priority

1. Stop Loss (catastrophic, fixed at entry)
2. Donchian channel exit (trend following stop)
3. Take Profit (if enabled)
4. Signal reversal
5. Session-end forced close

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset | **1** | No pyramiding (simplified variant; original Turtle system allows pyramiding) |
| Stop type | **Fixed ATR** + **Donchian channel trailing** | ATR stop is worst-case; Donchian exit is primary trailing stop |
| Stop loss (ATR) | `entry ± sl_atr_mult × ATR_entry` | Default: 3.0 × ATR — wide enough not to interfere with Donchian channel |
| Take profit (optional) | `entry ± tp_atr_mult × ATR_entry` | Default: disabled (use Donchian exit instead) |
| Default position sizing | **1 unit** | Original Turtle system uses ATR-scaled unit sizing |

### Turtle-style ATR Unit Sizing (reference)

Original Turtle system position sizing formula:
```
Unit = (equity × risk_fraction) / (ATR × dollar_per_point)
```
This scales position size inversely to ATR, giving equal dollar risk across assets. The `vol_scaled` mode in the existing backtest engine approximates this.

### Markets

| Group | Assets | Notes |
|-------|--------|-------|
| Forex | EURUSD, EURCAD, GBPCHF | 1D works well; 4h has more false breakouts |
| B3 | WDO, WIN | 1h and 4h; session filter required; breakouts often occur at market open |
| Crypto | BTCUSDT, ETHUSDT, BNBUSDT | 4h and 1D; crypto has strong trend regimes well-suited to Donchian |

### Time Restrictions

| Rule | Forex | B3 | Crypto |
|------|-------|----|--------|
| Session filter | None | 09:00–18:00 BRT | None (24/7) |
| Days of week | None tested | None tested | None tested |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| `dc_entry` | `20` | `[10, 15, 20, 30, 50, 55]` | Entry channel breakout period (bars) |
| `dc_exit` | `10` | `[5, 8, 10, 15, 20]` | Exit channel trailing stop period (bars) |
| `atr_period` | `14` | fixed | ATR period |
| `sl_atr_mult` | `3.0` | `[2.0, 3.0, 4.0, 5.0]` | Catastrophic ATR stop multiple |
| `tp_atr_mult` | — | — | Disabled by default; use Donchian exit |
| `use_tp` | `False` | — | Enable ATR-based take profit |

**Constraint:** `dc_exit < dc_entry`

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | dc_entry | dc_exit | sl_atr_mult |
|-------|----|-------|----------|---------|-------------|
| | | | | | |

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
**Regime sensitivity:** Strongly biased toward trending markets. Expected to have low win rate (30–40%) with large winners compensating many small losses — classic trend-following profile

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- Low win rate (~30–35%) is psychologically demanding and may lead to premature abandonment after losing streaks
- In ranging markets, breakouts are repeatedly faded, generating many small losses; an ADX filter could suppress entries in low-trend environments
- Entry on close means the entire breakout move of the entry bar is missed; entering on the next bar open after confirmation reduces slippage but adds delay
- The system is highly sensitive to the choice of `dc_entry` period — a 20-bar vs 55-bar system can behave very differently on the same data
- Original Turtle system includes a trade correlation filter (do not enter if a previous trade in the same direction was a winner); this is an advanced extension

---

## Implementation

**Notebook:** `technical_analysis/05_donchian_channel_breakout.ipynb`
**Source module:** `source/strategy.py` — `DonchianBreakoutStrategy`
**Parameters class:** `StrategyParams`

---

## References

1. Donchian, R.D. (1960). High Finance in Copper. *Financial Analysts Journal*, 16(6), 133–142.
2. Faith, C. (2003). *The Original Turtle Trading Rules* [online publication]. TurtleTrader.com.
3. Covel, M.W. (2007). *The Complete TurtleTrader*. HarperCollins.
4. Seykota, E. (1992). Cited in Schwager, J.D. *The New Market Wizards*. HarperCollins. Discussion of channel breakout systems.
5. Kaufman, P.J. (2013). *Trading Systems and Methods* (5th ed.). Wiley. Chapter on channel breakout systems.
