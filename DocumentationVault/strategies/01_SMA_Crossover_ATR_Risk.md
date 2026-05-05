# SMA Crossover with ATR Risk

> **Type:** Trend-following
> **Markets:** Forex, B3 (Bovespa mini-futures)
> **Timeframes:** Forex — 1h, 4h, 1D · B3 — 1min, 5min, 15min, 30min
> **Direction:** Long & Short
> **Status:** Backtested

---

## Overview

A classic dual moving-average crossover strategy that uses the Average True Range (ATR) to set dynamic, volatility-proportional stop loss and take profit levels. The hypothesis is that when a short-term trend (fast SMA) crosses a long-term trend (slow SMA), momentum is likely to continue in the crossover direction for long enough to reach a profit target placed at a multiple of current volatility. Risk is bounded symmetrically — the stop is placed at a smaller ATR multiple than the target, creating a reward-to-risk ratio greater than 1 on every trade. The strategy runs continuously in both directions (always in the market once indicators are warm), reversing position on each crossover.

---

## Indicators

### Indicator 1 — Simple Moving Average (Fast)

- **Input:** Bar close price
- **Formula:** `SMA_fast(t) = mean(close[t − fast + 1] … close[t])`
- **Lookback:** Requires exactly `fast` closed bars before producing a value. No partial windows — `min_periods = fast`.
- **Parameters:**
  - `fast` (int, default `20`) — lookback period in bars

### Indicator 2 — Simple Moving Average (Slow)

- **Input:** Bar close price
- **Formula:** `SMA_slow(t) = mean(close[t − slow + 1] … close[t])`
- **Lookback:** Requires exactly `slow` closed bars. `min_periods = slow`.
- **Parameters:**
  - `slow` (int, default `50`) — lookback period in bars

> Both SMAs must be valid (non-NaN) for a signal to fire. In practice this means the first `slow` bars of any dataset produce no signals.

### Indicator 3 — Average True Range (ATR)

- **Input:** High, Low, Close
- **Formula:**
  ```
  TR(t)  = max(high[t] − low[t],
               |high[t] − close[t−1]|,
               |low[t]  − close[t−1]|)

  ATR(t) = mean(TR[t − atr_period + 1] … TR[t])
  ```
  This is a **simple rolling mean** of True Range (not Wilder's EMA).
- **Lookback:** Requires `atr_period + 1` bars (one extra for the `close[t−1]` in the first TR computation). `min_periods = atr_period`.
- **Parameters:**
  - `atr_period` (int, default `14`) — lookback period in bars

> The ATR value used for SL/TP is the ATR at the **entry bar**. It is **not** updated while the position is open — stops are fixed from entry.

---

## Entry Signal

### Long Entry

All conditions must be true simultaneously on the same bar:

1. `SMA_fast(t) > SMA_slow(t)` — fast SMA is above slow SMA at bar close
2. `SMA_fast(t−1) ≤ SMA_slow(t−1)` — fast SMA was at or below slow SMA on the previous bar
3. `ATR(t)` is not NaN (indicators are warm)
4. _(If session filter active)_ bar hour `h` satisfies `session_start ≤ h < session_end`

This is a strict golden cross: the fast SMA must cross **from below or equal to above** in a single bar. A fast SMA that stays above does not re-trigger.

**Execution:**
- Price: bar close price of bar `t`
- Timing: end of bar `t` (no look-ahead into bar `t+1`)
- If a short position is currently open, it is closed at this same close price before the long is entered (reversal — see Exit Signal)

### Short Entry

Mirror of Long Entry with direction reversed:

1. `SMA_fast(t) < SMA_slow(t)`
2. `SMA_fast(t−1) ≥ SMA_slow(t−1)`
3. `ATR(t)` is not NaN
4. _(If session filter active)_ bar hour within session window

**Execution:** bar close of bar `t`; any open long is closed simultaneously.

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| Session filter | Off | When `session_start` and `session_end` are set, signals outside `[session_start, session_end)` hours are suppressed entirely. Bars outside this window produce `signal = 0`. |
| Warm-up guard | Always active | No signal fires until both SMAs and ATR have their required number of bars. |

---

## Exit Signal

### Primary Exits — Price-Based (Intrabar Check)

Stop and target are computed at entry and remain **fixed** for the life of the trade.

| Exit Type | Formula | Notes |
|-----------|---------|-------|
| Stop Loss (SL) | `entry − direction × sl_atr_mult × ATR_entry` | Uses ATR from the entry bar |
| Take Profit (TP) | `entry + direction × tp_atr_mult × ATR_entry` | Uses ATR from the entry bar |

Where `direction = +1` for long, `−1` for short.

**Intrabar trigger conditions:**

| Position | SL triggers when | TP triggers when |
|----------|-----------------|-----------------|
| Long | `bar_low ≤ SL_price` | `bar_high ≥ TP_price` |
| Short | `bar_high ≥ SL_price` | `bar_low ≤ TP_price` |

Exit price equals the SL or TP level exactly (not the bar close). This assumes the order fills at the level, ignoring slippage beyond the configured `slippage_points`.

### Secondary Exits — Signal-Based

| Exit Type | Trigger | Exit Price |
|-----------|---------|-----------|
| Signal reversal | An opposite crossover signal fires on bar `t` while a position is open | Bar close of bar `t` |
| Session-end forced close | Bar `t` is inside the session window AND bar `t+1` is outside (or does not exist) | Bar close of bar `t` |
| End of data | Dataset ends with an open position | Last bar's close price (reason: `"EOD"`) |

### Exit Priority (same-bar conflict resolution)

When multiple exits could trigger on the same bar, the following priority applies:

1. **Stop Loss** — checked first (pessimistic assumption)
2. **Take Profit** — checked only if SL did not trigger
3. **Signal reversal** — checked only if neither SL nor TP triggered
4. **Session-end forced close** — checked only if no price exit or reversal triggered

> This priority order is the *most conservative* assumption about intrabar execution. In reality, on a bar where both SL and TP levels are breached, we cannot know which filled first without tick data. Using SL-first understates profitability.

### PnL Calculation

```
pnl = (exit_price − entry_price) × direction × size − 2 × slippage_points × size
```

The `2 × slippage` models one slippage deduction at entry and one at exit.

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset | **1** | No pyramiding; always flat between entry and exit |
| Max simultaneous positions across assets | **1 per asset** | Multi-asset portfolio runs each asset independently |
| Stop type | **Fixed** (set at entry) | SL price does not move after entry |
| Stop loss | `entry ± sl_atr_mult × ATR_entry` | Default: `2.0 × ATR` |
| Take profit | `entry ± tp_atr_mult × ATR_entry` | Default: `3.0 × ATR` |
| Baseline reward-to-risk ratio | **1.5 : 1** | `tp_atr_mult / sl_atr_mult = 3.0 / 2.0` |
| Trailing stop | **No** | Stop is fixed from entry; no trailing logic |
| Default position sizing | **1 unit** per trade | PnL in price points |

### Position Sizing Modes

| Mode | Size Formula | PnL Unit | When to Use |
|------|-------------|----------|-------------|
| `unit` | `1` | Price points | Simple comparison, default |
| `vol_scaled` | `1 / ATR_entry` | ATR-normalized (dimensionless) | Comparing across timeframes or assets |
| `fixed_frac` | `equity × risk_fraction / (sl_atr_mult × ATR_entry)` | Price points × contracts | Risk-consistent position sizing per trade |

`fixed_frac` requires `initial_capital` to be set on the `Backtester`. Default risk fraction: `1%` of equity per trade.

### Approved Markets

| Group | Assets | Notes |
|-------|--------|-------|
| Forex | EURUSD, EURCAD, GBPCHF | M1 raw data resampled to target TF |
| B3 | WDO (USD/BRL mini), WIN (Bovespa mini) | M1 raw data resampled; session filter recommended |

B3 and Forex PnL are in different currencies/units — **do not compare raw PnL numbers across groups**.

### Time Restrictions

| Rule | Forex | B3 |
|------|-------|----|
| Session filter | None (24/5 continuous) | Recommended: `session_start=9, session_end=18` (local BRT) |
| Days of week | None tested | None tested |
| News/event blackout | Not implemented | Not implemented |

**B3 session filter rationale:** WDO and WIN trade 09:00–18:00 BRT. Bars after 18:00 are low-liquidity and produce false signals. The filter suppresses new entries outside the window and forces close at the last in-session bar. `session_end` is exclusive: a bar at hour 18 is outside the session.

---

## Parameters

| Parameter | Default | WFO Grid | Type | Description |
|-----------|---------|----------|------|-------------|
| `fast` | `20` | `[5, 10, 20, 30, 50]` | int | Fast SMA lookback (bars) |
| `slow` | `50` | `[40, 60, 100, 150, 200]` | int | Slow SMA lookback (bars) |
| `atr_period` | `14` | fixed | int | ATR simple moving average lookback |
| `sl_atr_mult` | `2.0` | `[1.0, 1.5, 2.0, 2.5, 3.0]` | float | Stop loss = this multiple of ATR from entry |
| `tp_atr_mult` | `3.0` | `[1.5, 2.0, 3.0, 4.0, 5.0]` | float | Take profit = this multiple of ATR from entry |
| `session_start` | `None` | — | int ∣ None | Hour (0–23) from which entries are allowed (inclusive) |
| `session_end` | `None` | — | int ∣ None | Hour (0–23) at which entries are blocked and positions forced closed (exclusive) |
| `sizing_mode` | `"unit"` | — | str | `"unit"` / `"vol_scaled"` / `"fixed_frac"` |
| `risk_fraction` | `0.01` | — | float | Fraction of equity risked per trade (`fixed_frac` mode only) |

**Constraint:** `fast < slow` must always hold. The crossover is only meaningful if the two periods are distinct; having `fast ≥ slow` produces constant or degenerate signals.

### WFO-Optimized Parameters (most frequent selection across 5 folds)

| Group | TF | Asset | fast | slow | sl_atr_mult | tp_atr_mult |
|-------|----|-------|------|------|-------------|-------------|
| Forex | 1h | EURCAD | 50 | 100 | 3.0 | 4.0 |
| Forex | 1h | EURUSD | 5 | 40 | 2.5 | 2.0 |
| Forex | 1h | GBPCHF | 10 | 100 | 3.0 | 5.0 |
| Forex | 4h | EURCAD | 10 | 60 | 1.0 | 1.5 |
| Forex | 4h | EURUSD | 50 | 100 | 3.0 | 1.5 |
| Forex | 4h | GBPCHF | 5 | 150 | 2.0 | 2.0 |
| Forex | 1D | EURCAD | 30 | 40 | 1.0 | 1.5 |
| Forex | 1D | EURUSD | 30 | 40 | 1.5 | 5.0 |
| Forex | 1D | GBPCHF | 10 | 40 | 1.0 | 3.0 |
| B3 | 1min | WDO | 30 | 200 | 1.0 | 4.0 |
| B3 | 1min | WIN | 50 | 100 | 2.5 | 2.0 |
| B3 | 5min | WDO | 20 | 60 | 2.0 | 1.5 |
| B3 | 5min | WIN | 10 | 200 | 1.5 | 1.5 |
| B3 | 15min | WDO | 50 | 100 | 1.5 | 2.0 |
| B3 | 15min | WIN | 50 | 100 | 3.0 | 1.5 |
| B3 | 30min | WDO | 20 | 60 | 2.5 | 1.5 |
| B3 | 30min | WIN | 50 | 150 | 1.5 | 2.0 |

---

## Performance Summary

All results use `slippage_points = 0` unless noted.

### Baseline Performance (fast=20, slow=50, sl=2×ATR, tp=3×ATR)

#### Forex

| TF | Trades | Total PnL | Win Rate | Profit Factor | Sharpe (daily) | Max DD |
|----|--------|-----------|----------|---------------|----------------|--------|
| 1h | 4,193 | -0.555 | 38.7% | 0.93 | -0.49 | -0.58 |
| 4h | 1,106 | -0.153 | 38.4% | 0.96 | -0.14 | -0.48 |
| 1D | 192 | -0.809 | 30.2% | 0.63 | -0.74 | -0.80 |

#### B3

| TF | Trades | Total PnL | Win Rate | Profit Factor | Sharpe (daily) | Max DD |
|----|--------|-----------|----------|---------------|----------------|--------|
| 1min | 31,989 | -31,315 | 38.7% | 0.98 | -0.36 | -56,859 |
| 5min | 6,247 | +14,746 | 39.7% | 1.02 | +0.16 | -19,256 |
| 15min | 2,100 | -3,645 | 38.5% | 0.99 | -0.04 | -40,836 |
| 30min | 992 | -4,467 | 40.0% | 0.99 | -0.05 | -27,662 |

### WFO-Optimized Full-History Performance

#### Forex

| TF | Trades | Sharpe (daily) | Profit Factor | Max DD |
|----|--------|----------------|---------------|--------|
| 1h | 4,192 | +0.069 | 1.011 | -0.58 |
| 4h | 907 | +0.160 | 1.050 | -0.35 |
| **1D** | **264** | **+0.404** | **1.262** | **-0.24** |

#### B3

| TF | Trades | Sharpe (daily) | Profit Factor | Max DD |
|----|--------|----------------|---------------|--------|
| **1min** | **13,098** | **+0.332** | **1.023** | -28,488 |
| 5min | 4,287 | -0.294 | 0.959 | -39,077 |
| **15min** | **958** | **+0.342** | **1.098** | -19,534 |
| 30min | 629 | +0.241 | 1.107 | -34,052 |

### Robustness (Best OOS Timeframe: Forex 1D · B3 30min)

| Test | Forex 1D | B3 30min |
|------|----------|----------|
| Block bootstrap P(profitable) | **89.4%** | **66.8%** |
| Sub-period consistency | Good 2016–2018; declining 2019–2024 | Good 2021–2022; declining 2023–2025 |

### Extension Impact (B3 30min, baseline params)

| Configuration | Sharpe | Profit Factor |
|---------------|--------|---------------|
| Unit, equal weights | -0.05 | 0.99 |
| + Session filter (09–18) | +0.03 | ~1.01 |
| + vol_scaled sizing | +0.17 | 1.04 |
| + min_var portfolio weights | **+0.51** | **1.16** |

**Best validated configuration:** B3 30min · session filter · vol_scaled sizing · min_var portfolio weighting (not yet jointly tested — values above are individual extension results).

### Regime Sensitivity (Critical)

The strategy shows strong performance only during trending regimes:
- **Forex:** 2016–2018 (Sharpe 0.83–2.07/year). Edge degrades sharply from 2019.
- **B3:** 2021–2022 (Sharpe 1.50–2.69/year). Edge turns negative from 2023.

A regime detection filter (e.g. ADX threshold, long-term MA slope test) is the highest-priority improvement before live deployment.

---

## Known Weaknesses & Improvement Ideas

| Weakness | Impact | Suggested Fix |
|----------|--------|---------------|
| No regime filter | Strategy trades in choppy, non-trending markets and loses consistently | Add ADX > threshold or 200-period MA slope filter as a global on/off switch |
| Fixed stops (no trail) | Gives back large open profits on trend reversals | Add ATR trailing stop that ratchets as price moves in favor |
| Bar-close execution | Assumes fill at close; real fills may be worse | Add configurable `slippage_points` and test sensitivity |
| No commissions | Overstates net PnL, especially at high frequencies (B3 1min) | Apply per-trade cost calibrated to broker/exchange schedule |
| SL-before-TP intrabar assumption | Understates true TP hit rate | Validate with tick data on a sample period |
| Anchored WFO folds | Fixed equal slices do not adapt to changing regimes | Replace with expanding-window or regime-anchored WFO |
| No volume data | Cannot confirm crossover signal with volume surge | Extend data pipeline to include volume |
| B3 1min: 32k trades | Very high churn — commissions would likely destroy the edge | 5min or 15min preferred for B3 |

---

## Implementation

**Notebook:** `technical_analysis/01_baseline_sma_crossover.ipynb`
**Source module:** `source/strategy.py` — `SMACrossoverStrategy`
**Backtester:** `source/backtest.py` — `Backtester`
**Parameters class:** `source/strategy.py` — `StrategyParams`

### Minimal Usage Example

```python
from source import SMACrossoverStrategy, StrategyParams, Backtester

params = StrategyParams(
    fast=10, slow=60,
    atr_period=14,
    sl_atr_mult=1.5, tp_atr_mult=2.5,
    session_start=9, session_end=18,   # B3 only
    sizing_mode="vol_scaled",
)
result = Backtester(SMACrossoverStrategy(params), slippage_points=0.0).run(df)
print(result.trades[["entry_time","exit_time","direction","pnl_points","reason"]])
```

See [[04_Backtesting_and_Metrics]] for multi-asset portfolio construction.
See [[05_Walk_Forward_Optimization]] for WFO methodology and optimized param tables.
See [[06_Robustness_Testing]] for Monte Carlo, block bootstrap, and sub-period results.
