# Stochastic Oscillator with Price Divergence

> **Type:** Mean-reversion / Momentum
> **Markets:** Forex, B3, Crypto
> **Timeframes:** 15min, 1h, 4h
> **Direction:** Long & Short
> **Status:** Idea

---

## Overview

The Stochastic Oscillator measures the position of the current close relative to the high-low range of the past N bars, normalized to 0–100. It captures the intuition that in uptrends, prices tend to close near the session high, and in downtrends near the session low. When the oscillator diverges from price — price makes a new high but the stochastic makes a lower high — it signals that the recent price move lacks the breadth to sustain itself, suggesting an imminent reversal.

This strategy uses two entry modes: (1) stochastic extreme crossovers (similar to RSI mean reversion in Strategy 02 but using a different oscillator with different mathematical properties), and (2) price-stochastic divergence as a higher-conviction reversal signal. The divergence mode tends to generate fewer but better-quality signals, while the crossover mode generates more signals with lower individual accuracy.

The Stochastic Oscillator is more sensitive to recent price action than RSI due to its look-back range normalization, making it well-suited to shorter timeframes and instruments with regular intraday cyclicality.

---

## Indicators

### Indicator 1 — Stochastic %K (Fast)

- **Input:** High, Low, Close
- **Formula:**
  ```
  lowest_low(t)   = min(low[t−k_period+1], …, low[t])
  highest_high(t) = max(high[t−k_period+1], …, high[t])

  %K(t) = 100 × (close(t) − lowest_low(t)) / (highest_high(t) − lowest_low(t))
  ```
  If `highest_high = lowest_low` (zero range bar), `%K = 50` (neutral, undefined range).
- **Lookback:** `k_period` bars
- **Parameters:**
  - `k_period` (int, default `14`) — look-back period for raw %K

### Indicator 2 — Stochastic %D (Smooth)

- **Input:** %K
- **Formula:**
  ```
  %D(t) = SMA(%K, d_period)[t]
  ```
  Simple moving average of %K. This is the "slow" signal line.
- **Lookback:** `k_period + d_period − 1` bars total
- **Parameters:**
  - `d_period` (int, default `3`) — smoothing period for %D signal line

### Indicator 3 — Stochastic Divergence Detector

- **Input:** Close price, %K
- **Formula:**
  ```
  # Bullish divergence:
  # Price makes a lower low over divergence_window bars
  # while %K makes a higher low over the same window

  price_lower_low(t)  = close(t) < min(close[t−divergence_window+1 .. t−1])
                        AND close(t−lookback_swing) was a prior local low

  stoch_higher_low(t) = %K(t) > min(%K[t−divergence_window+1 .. t−1])

  bullish_divergence(t) = price_lower_low(t) AND stoch_higher_low(t)

  # Bearish divergence: mirror
  ```
  Simplified detection: within a rolling window, find if price makes a new extreme while %K does not confirm it.

  **Practical implementation:** Use a swing-high/swing-low detection on close (using rolling argmax/argmin) to identify pivot points, then compare the last two pivots in both price and %K.
- **Lookback:** `divergence_window` bars
- **Parameters:**
  - `divergence_window` (int, default `20`) — rolling window to search for divergence

### Indicator 4 — Average True Range (ATR)

- **Input:** High, Low, Close
- **Formula:** Simple rolling mean of True Range
- **Lookback:** `atr_period + 1` bars
- **Parameters:**
  - `atr_period` (int, default `14`)

---

## Entry Signal

### Entry Mode 1: Stochastic Crossover (Mean Reversion)

#### Long Entry (Crossover Mode)

1. `%K(t) > stoch_oversold` AND `%K(t−1) ≤ stoch_oversold` — %K crosses back above oversold threshold
2. `%D(t) < 50` — signal line confirms oversold territory (optional filter)
3. `ATR(t)` is not NaN

#### Short Entry (Crossover Mode)

1. `%K(t) < stoch_overbought` AND `%K(t−1) ≥ stoch_overbought`
2. `%D(t) > 50` (optional)

### Entry Mode 2: Divergence (Higher Conviction)

#### Long Entry (Divergence Mode)

1. `bullish_divergence(t) = True` — price made lower low, %K made higher low in past `divergence_window` bars
2. `%K(t) < 50` — stochastic is still below midpoint (not already extended)
3. `%K(t) > %K(t−1)` — %K is currently rising (momentum confirmation)
4. `ATR(t)` is not NaN

#### Short Entry (Divergence Mode)

1. `bearish_divergence(t) = True`
2. `%K(t) > 50`
3. `%K(t) < %K(t−1)` (falling)

**Execution (all modes):** Bar close of bar `t`

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| Entry mode | `"crossover"` | `"crossover"` or `"divergence"` — select which entry mode to use |
| ADX filter | Off | For crossover mode, suppress entries when ADX > adx_max (trend regime) |
| Session filter | Off | B3: 09:00–18:00 BRT |
| Warm-up guard | Always active | All indicators must be initialized |

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
| Stochastic midline exit | `%K(t) ≥ 50` for long / `%K(t) ≤ 50` for short — oscillator has reverted to neutral | Bar close |
| Signal reversal | Opposite entry signal fires | Bar close |
| Session-end | Last in-session bar | Bar close |
| End of data | Dataset ends | Last close |

### Exit Priority

1. Stop Loss
2. Take Profit
3. Stochastic midline exit (if `use_midline_exit = True`)
4. Signal reversal
5. Session-end

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset | **1** | |
| Stop type | **Fixed** | ATR at entry |
| Stop loss | `entry ± sl_atr_mult × ATR_entry` | Default: 1.5 × ATR |
| Take profit | `entry ± tp_atr_mult × ATR_entry` | Default: 2.0 × ATR |
| Trailing stop | **No** | |
| Default position sizing | **1 unit** | |

### Markets

| Group | Assets | Notes |
|-------|--------|-------|
| Forex | EURUSD, EURCAD, GBPCHF | 1h and 4h; crossover mode |
| B3 | WDO, WIN | 15min and 1h; crossover mode with session filter |
| Crypto | BTCUSDT, ETHUSDT | 1h; divergence mode may add value on larger timeframes |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| `k_period` | `14` | `[5, 9, 14, 21]` | Stochastic %K lookback period |
| `d_period` | `3` | `[3, 5]` | %D smoothing period |
| `stoch_oversold` | `20` | `[15, 20, 25]` | Oversold threshold for long crossover |
| `stoch_overbought` | `80` | `[75, 80, 85]` | Overbought threshold for short crossover |
| `divergence_window` | `20` | `[15, 20, 30]` | Rolling window for divergence detection |
| `atr_period` | `14` | fixed | ATR period |
| `sl_atr_mult` | `1.5` | `[1.0, 1.5, 2.0, 2.5]` | Stop loss ATR multiple |
| `tp_atr_mult` | `2.0` | `[1.5, 2.0, 3.0]` | Take profit ATR multiple |
| `entry_mode` | `"crossover"` | — | `"crossover"` or `"divergence"` |
| `use_d_filter` | `False` | — | Require %D to confirm oversold/overbought |
| `use_midline_exit` | `True` | — | Exit at %K = 50 |
| `adx_max` | `25` | — | Max ADX when using crossover mode |
| `use_adx_filter` | `False` | — | Enable ADX filter for crossover mode |

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | k_period | stoch_oversold | stoch_overbought | sl_atr_mult | tp_atr_mult |
|-------|----|-------|----------|----------------|------------------|-------------|-------------|
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
**Regime sensitivity:** Like RSI, the stochastic performs poorly during strong trends. The divergence mode should be more regime-robust as it requires a structural signal, not just an extreme reading.

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- Divergence detection is inherently subjective; the programmatic swing-high/swing-low pivot detection is sensitive to the lookback window and can produce different results for slightly different implementations
- %K can spike to 0 or 100 on a single extreme bar without indicating a reversal — a smoothed version (slow stochastic, using SMA of %K before computing %D) reduces this noise
- The divergence signal can be early: price may continue making lower lows for several more bars before reversing, triggering the stop before the anticipated reversion
- Consider testing a %K/%D crossover (rather than raw %K threshold crossover) as an alternative entry trigger — smoother but laggier

---

## Implementation

**Notebook:** `technical_analysis/11_stochastic_divergence.ipynb`
**Source module:** `source/strategy.py` — `StochasticDivergenceStrategy`
**Parameters class:** `StrategyParams`

---

## References

1. Lane, G.C. (1984). Lane's Stochastics. *Technical Analysis of Stocks & Commodities*, 2(3). Original publication of the Stochastic Oscillator.
2. Murphy, J.J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance. Chapter on oscillators and stochastics.
3. Colby, R.W. (2003). *The Encyclopedia of Technical Market Indicators* (2nd ed.). McGraw-Hill. Entry on Stochastic Oscillator.
4. Schwager, J.D. (1996). *Technical Analysis* (Schwager on Futures series). Wiley. Chapter on momentum oscillators and divergence.
