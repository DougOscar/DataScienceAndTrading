# Multi-Filter Portfolio System

> **Type:** Trend-following / Multi-confirmation portfolio system
> **Markets:** Forex, Crypto, B3
> **Timeframes:** 15min–1D (swing + intraday)
> **Direction:** Long & Short
> **Status:** Backtested (v1, committed unexecuted)

---

## Overview

A single **portfolio-level** system traded on one shared account across three
markets. The edge it targets is *quality of confluence*: a trend-following
trigger only fires a trade when it is corroborated by independent trend,
momentum and trend-strength filters **and** confirmed by relative volume —
filtering out the low-conviction crossovers that make naive trend systems
bleed. Capital is rationed by a hard per-trade risk budget, a concurrency cap
and (in forex) a currency-exposure cap, so the account is never concentrated in
one correlated bet.

The portfolio rules are *cross-asset* and cannot be expressed by the
single-asset `Backtester`; they live in a dedicated engine
(`source/portfolio_backtest.py`).

---

## Indicators

All computed by `MultiFilterSystemStrategy.compute_indicators`. With
`full=True` the whole library is materialised (notebook feature panel); during
signal generation only the indicators the *active* filters need are computed.

### Trigger indicators

- **EMA fast/slow** — `Close.ewm(span=ema_fast|ema_slow)`. Lookback = span.
- **Donchian channel** — rolling max(high)/min(low) over `donchian_period`.
- **MACD** — `EMA(macd_fast) − EMA(macd_slow)`, signal `EMA(macd_signal)`,
  histogram = line − signal.

### Confirmation indicators

- **Trend EMA** — `Close.ewm(span=trend_ema)` (default 200).
- **RSI** — Wilder smoothing, `rsi_period` (default 14).
- **ADX / +DI / −DI** — Wilder, `adx_period` (default 14).
- **Ichimoku** *(optional)* — Tenkan/Kijun/Senkou A/Senkou B
  (`tenkan_period`/`kijun_period`/`senkou_b_period`).
- **Bollinger Bands** *(optional)* — `bb_period`, `bb_mult`.
- **SuperTrend** *(optional)* — ATR envelope, `supertrend_period`,
  `supertrend_mult`.

### Volume & panel-only indicators

- **Volume ratio** — `tick_vol / SMA(tick_vol, vol_period)`.
- Panel-only (read-only, `full=True`): CCI, Williams %R, ROC, OBV, intraday
  VWAP, Stochastic %K/%D, ATR, TR.

---

## Entry Signal

### Long Entry

All must be true:

1. **Trigger** bullish (`entry_signal`): fast EMA crosses above slow EMA, or
   close breaks the prior `donchian_period` high, or MACD crosses above signal.
2. `Close > EMA(trend_ema)`  *(confirmation 1 — trend)*.
3. `RSI > rsi_mid`  *(confirmation 2 — momentum)*.
4. `ADX ≥ adx_min`  *(confirmation 3 — strength)*.
5. *(optional, if enabled)* close above the Ichimoku Kumo / not over-extended
   past the upper Bollinger band / SuperTrend bullish.
6. **Volume filter (separate):** `tick_vol / SMA(tick_vol) ≥ vol_ratio_min`.

Execution: at the **signal bar's close**.

### Short Entry

Mirror of Long Entry.

### Entry Filters

- **Confirmations 1–3** above are the ≥ 3 mandatory confirmation filters.
- **Volume filter** is applied separately, independent of the confirmations.
- **Session filter:** `session_start ≤ hour < session_end` when set (B3).

---

## Exit Signal

### Primary Exits (Price-based)

| Exit Type | Long Condition | Short Condition | Exit Price |
|-----------|----------------|-----------------|-----------|
| Stop Loss | `low ≤ entry − sl_atr_mult·ATR_entry` | `high ≥ entry + sl_atr_mult·ATR_entry` | stop level |
| Take Profit *(optional)* | `high ≥ entry + tp_atr_mult·ATR_entry` | `low ≤ entry − tp_atr_mult·ATR_entry` | TP level |

Take profit is only active when `use_take_profit=True` — an explicit
optimisation candidate (off by default).

### Secondary Exits (Signal-based)

| Exit Type | Condition | Exit Price |
|-----------|-----------|-----------|
| Opposite trigger | The opposite trigger fires (`signal == −position`) | Bar close |
| Session end | Last in-session bar (when session set) | Bar close |
| End of data | Last bar of the asset | Bar close |

The exit is **systematic and filter-free** (no confirmations gate it).

### Exit Priority

1. Stop loss (pessimistic — checked before TP on the same bar)
2. Take profit (if enabled)
3. Opposite-trigger signal
4. Session end
5. End of data

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions | **3** | Portfolio-wide (`max_concurrent`) |
| Risk per trade | **≤ 2 %** | `size = equity·risk_fraction / (sl_atr_mult·ATR)` ⇒ risk-to-stop = `equity·risk_fraction` |
| Stop type | Fixed (ATR at entry) | |
| Stop loss | `entry ∓ sl_atr_mult·ATR_entry` | mandatory |
| Take profit | `entry ± tp_atr_mult·ATR_entry` | **optional** (optimisation knob) |
| Trailing stop | No | out of scope v1 (engine has no per-bar exit hook) |
| Position sizing mode | `fixed_frac` | shared running equity |
| Pyramiding | No | one open position per asset |

### Markets

| Group | Assets | Conditions |
|-------|--------|-----------|
| forex | 31 pairs/metals | **Currency exclusion ON** — an open `EURUSD` blocks any other instrument containing `EUR` or `USD` |
| crypto | BTC/ETH/LTC USD | Concurrency cap only (no currency exclusion) |
| b3 | 23 stocks + WIN/WDO | Concurrency cap; session 09–18 |

### Time Restrictions

| Rule | Description |
|------|-------------|
| Session filter | B3: `session_start=9`, `session_end=18`; forex/crypto 24 h |
| Days of week | none |
| News/event blackout | none |

---

## Parameters

**Parameters class:** `MultiFilterSystemParams`

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| entry_signal | `ema_cross` | ema_cross / donchian / macd_cross | primary trigger |
| ema_fast | 20 | 10, 20 | trigger fast EMA |
| ema_slow | 50 | 50, 100 | trigger slow EMA |
| trend_ema | 200 | 150, 200 | trend confirmation EMA |
| donchian_period | 20 | 20, 40 | Donchian trigger lookback |
| adx_min | 20 | 15, 25 | min ADX (strength confirmation) |
| use_take_profit | False | False, True | enable ATR take-profit |
| tp_atr_mult | 3.0 | 2.0, 3.0 | take-profit ATR multiple |
| sl_atr_mult | 2.0 | 1.5, 2.0, 3.0 | stop ATR multiple |
| vol_ratio_min | 1.2 | 1.0, 1.3 | volume-filter threshold |
| risk_fraction | 0.02 | fixed | ≤ 2 % account risk / trade |

`MultiFilterSystemParams.is_valid()` enforces `ema_fast < ema_slow < trend_ema`
and `macd_fast < macd_slow`; an invalid grid combo emits zero signals and is
scored `−inf` by the WFO driver (never raises).

### WFO-Optimized Values

_(Fill in after running §6 — `portfolio_walk_forward` on the representative
subset, ranked by the consolidated index.)_

| Group | TF | entry_signal | ema_fast | ema_slow | sl_atr_mult | use_tp |
|-------|----|--------------|----------|----------|-------------|--------|
| | | | | | | |

---

## Performance Summary

> Populated by running the notebook (committed unexecuted).

The optimisation objective is a single **consolidated index**
(`source.portfolio_backtest.consolidated_index`):

```
index =  0.35·clip(annualised Sharpe, -3, 5)
       + 0.15·(win_rate − 0.5)·4
       + 0.25·clip(total_pnl / |max_drawdown|, -2, 5)
       + 0.10·tanh(profit_factor − 1)
       + 0.15·min(|t-stat|, 4)/4
```

| Metric | Baseline | WFO-Optimized | Notes |
|--------|----------|---------------|-------|
| Sharpe (annualised) | | | |
| Profit Factor | | | |
| Win Rate | | | |
| Max Drawdown | | | |
| Calmar | | | |
| Consolidated index | | | |

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- No trailing/Chandelier/Kumo-cross exit — needs a per-bar dynamic-exit hook in
  the engine (consistent gap across all repo notebooks).
- Named metals (`Platinum`, `Palladium`) lack a 3-letter code so they only
  currency-exclude themselves.
- The cross-asset event loop is single-threaded per `(group, tf)` cell;
  intraday timeframes over the full universe are slow (cells run in parallel,
  but a single 15min×31-asset forex cell is the bottleneck).
- Equity is realised-PnL (mark-on-close), not mark-to-market intrabar.

---

## Implementation

**Notebook:** `technical_analysis/15_multi_filter_portfolio_system.ipynb`
**Source module:** `source/strategy.py` — `MultiFilterSystemStrategy`
**Parameters class:** `MultiFilterSystemParams`
**Portfolio engine:** `source/portfolio_backtest.py` — `PortfolioBacktester`,
`portfolio_walk_forward`, `consolidated_index`, `portfolio_parameter_sensitivity`,
`overfitting_report`
**Data layer:** `source/spark_loader.py` — PySpark M1 → multi-TF parquet cache
(`build_spark_grid`); requires `pyspark>=3.5` + **JDK 17 or 21**.

### Implementation Notes

- **PySpark data engine** (chosen over the repo's lazy-CSV loader for this
  system): every M1 file is window-aggregated to the target timeframes via a
  tumbling Spark `window` and parquet-cached by source mtime. Backtests then
  read small parquet slices and fan out across processes.
- **JDK constraint.** Spark 4.x runs only on **JDK 17 or 21**; JDK 24+ removed
  `jdk.internal.ref.Cleaner` so `SparkContext` init throws
  `ClassNotFoundException` (JVM `--add-opens` flags cannot fix a missing
  class). `get_spark()` auto-detects a compatible JDK (`find_compatible_jdk`),
  pins `JAVA_HOME` for the Spark JVM only, accepts an explicit
  `java_home=`, and raises an actionable error if only an unsupported JDK is
  present. PySpark 4.1 imports/starts on Python 3.14 in practice, but
  3.11/3.12 is the certified range.
- **Portfolio constraints are new infrastructure.** `≤ 2 % risk`,
  `≤ 3 concurrent`, and forex currency exclusion are cross-asset and are *not*
  expressible via the single-asset `Backtester` — hence `PortfolioBacktester`.
- **Optional confirmations** (Ichimoku / Bollinger-extension / SuperTrend) are
  wired and grid-selectable but **disabled by default** (`use_*_filter=False`)
  so the 3 mandatory confirmations define the baseline.
- **Take profit disabled by default** (`use_take_profit=False`) and exposed as
  an optimisation candidate, per the spec.
- **Exit limited to fixed-at-entry SL/TP + opposite-signal + session/EOD**, the
  same Backtester-contract constraint noted in every other strategy doc here.
- **WFO / sensitivity run on a representative subset** (`WFO_ASSETS`); the
  simple baseline backtest uses all assets.
- Notebook committed **unexecuted**.
