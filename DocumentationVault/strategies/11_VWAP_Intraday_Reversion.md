# VWAP Intraday Mean Reversion

> **Type:** Mean-reversion / Intraday
> **Markets:** B3, Crypto
> **Timeframes:** 1min, 5min, 15min
> **Direction:** Long & Short
> **Status:** Idea

---

## Overview

The Volume Weighted Average Price (VWAP) represents the average transaction price for the session, weighted by volume at each price level. It is widely used by institutional traders as a benchmark for execution quality — desks buying below VWAP have outperformed, those buying above have underperformed. This institutional anchor creates a self-fulfilling mean-reversion dynamic: when price deviates significantly from VWAP, algorithmic execution programs tend to fade the deviation, pushing price back toward the benchmark.

This strategy enters mean-reversion trades when price deviates from the session VWAP by a multiple of its rolling standard deviation (VWAP bands). The take profit is the VWAP itself. All positions are closed at session end. The strategy requires tick_vol data (available in the OHLCV files) to compute VWAP and is therefore not applicable to synthetic or volume-free data.

---

## Indicators

### Indicator 1 — Session VWAP

- **Input:** Close, High, Low, and `tick_vol` (or `volume`) per bar. Session must be defined by `session_start` and `session_end` parameters.
- **Formula:**
  ```
  typical_price(t) = (high(t) + low(t) + close(t)) / 3

  VWAP(t) = Σ(typical_price[s] × tick_vol[s], s = session_start_bar .. t)
             ─────────────────────────────────────────────────────────────
             Σ(tick_vol[s], s = session_start_bar .. t)
  ```
  VWAP resets to NaN at the start of each new session (each trading day).
- **Lookback:** From session start to bar `t` (cumulative within session)
- **Parameters:** None (computed deterministically from price and volume)
- **Note:** If only `tick_vol` is available (not actual traded volume), use `tick_vol` as the weight. VWAP computed from tick_vol is an approximation but retains the mean-reversion anchor property.

### Indicator 2 — VWAP Standard Deviation Bands

- **Input:** VWAP, typical_price, tick_vol
- **Formula:**
  ```
  variance(t) = Σ(tick_vol[s] × (typical_price[s] − VWAP(t))², s = session_start..t)
                ────────────────────────────────────────────────────────────────────
                Σ(tick_vol[s], s = session_start..t)

  VWAP_std(t) = sqrt(variance(t))

  VWAP_upper(t) = VWAP(t) + n_sigma × VWAP_std(t)
  VWAP_lower(t) = VWAP(t) − n_sigma × VWAP_std(t)
  ```
  This is the volume-weighted standard deviation of price around VWAP — it measures how dispersed prices have been throughout the session.
- **Parameters:**
  - `n_sigma` (float, default `2.0`) — number of standard deviations for the bands

### Indicator 3 — Average True Range (ATR)

- **Input:** High, Low, Close (intraday)
- **Formula:** Simple rolling mean of True Range
- **Lookback:** `atr_period + 1` bars
- **Parameters:**
  - `atr_period` (int, default `14`) — intraday bars

### Indicator 4 — Minimum Bars Since Session Start

VWAP bands are unreliable in the first `vwap_warmup_bars` bars of each session when the cumulative sums are too small to produce stable estimates.
- **Parameters:**
  - `vwap_warmup_bars` (int, default `10`) — bars from session start before entries are allowed

---

## Entry Signal

### Long Entry

All conditions at bar close `t`:

1. `close(t) ≤ VWAP_lower(t)` — price closes at or below the lower VWAP band
2. Bar `t` is at least `vwap_warmup_bars` into the current session
3. Bar `t` is at least `entry_cutoff_minutes` before session end
4. No long position currently open in this session
5. `VWAP_std(t) > 0` (non-zero volume variance — session has seen price movement)

**Execution:**
- **Price:** bar close of bar `t`
- **Bar:** signal bar `t`

### Short Entry

Mirror of Long Entry:

1. `close(t) ≥ VWAP_upper(t)`
2. Bars 2–5 same as Long Entry

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| Session warmup guard | Always active | No entries in first `vwap_warmup_bars` bars of session |
| Entry cutoff | Always active | No entries within `entry_cutoff_minutes` of session end |
| One-trade-per-session-per-direction | Optional | After a loss in one direction, suppress same-direction entries for rest of session |
| Session filter | Required | Session boundaries must be configured to reset VWAP daily |

---

## Exit Signal

### Primary Exits — Price-Based

| Exit Type | Long | Short | Exit Price |
|-----------|------|-------|-----------|
| Stop Loss | `bar_low ≤ entry − sl_atr_mult × ATR_entry` | `bar_high ≥ entry + sl_atr_mult × ATR_entry` | SL level |
| Take Profit (VWAP touch) | `bar_high ≥ VWAP(t)` | `bar_low ≤ VWAP(t)` | VWAP(t) at the exit bar |

The VWAP take profit is dynamic: the VWAP value evolves each bar as new bars are added. The TP triggers when the bar's intrabar high (for long) reaches the current VWAP level.

### Secondary Exits — Signal-Based

| Exit Type | Condition | Exit Price |
|-----------|-----------|-----------|
| Opposite band touch | Long: price reaches upper VWAP band (`close ≥ VWAP_upper`) — full reversion and overshoot | Bar close |
| Session-end forced close | Last session bar (bar hour = `session_end − 1`) | Bar close |
| End of data | Dataset ends with open position | Last close |

### Exit Priority

1. Stop Loss
2. Take Profit (VWAP touch)
3. Opposite band touch
4. Session-end forced close

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset per session | **1** | |
| Stop type | **Fixed** (ATR) | TP is dynamic (VWAP) |
| Stop loss | `entry ± sl_atr_mult × ATR_entry` | Default: 2.0 × ATR |
| Take profit | VWAP touch | Typically 1–2× ATR distance from entry at band touch |
| Session-end forced close | Always active | VWAP is intraday only; no overnight exposure |
| Default position sizing | **1 unit** | vol_scaled for cross-asset comparison |

### Reward-to-Risk Approximation

At entry (price at VWAP ± n_sigma × VWAP_std):
- Distance to TP (VWAP) ≈ n_sigma × VWAP_std
- Distance to SL ≈ sl_atr_mult × ATR

The RR ratio is asset and session dependent. ATR and VWAP_std should be compared empirically. Typical guidance: ensure sl_atr_mult × ATR is not greater than n_sigma × VWAP_std (i.e., RR ≥ 1:1).

### Markets

| Group | Assets | Notes |
|-------|--------|-------|
| B3 | WDO (USD/BRL mini), WIN (Bovespa mini) | Primary market; strong institutional VWAP usage; 5min preferred |
| Crypto | BTCUSDT, ETHUSDT | Sessions defined as UTC calendar days; 5min and 15min |

### Time Restrictions

| Rule | B3 | Crypto |
|------|----|--------|
| Session definition | 09:00–18:00 BRT | 00:00–00:00 UTC (daily) |
| VWAP warmup | 10 bars | 10 bars |
| Entry cutoff | 17:30 BRT | 23:30 UTC |
| Days of week | Mon–Fri | Mon–Sun |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| `n_sigma` | `2.0` | `[1.5, 2.0, 2.5, 3.0]` | VWAP band width in standard deviations |
| `atr_period` | `14` | fixed | ATR period (intraday bars) |
| `sl_atr_mult` | `2.0` | `[1.5, 2.0, 2.5]` | Stop loss ATR multiple |
| `vwap_warmup_bars` | `10` | `[5, 10, 20]` | Bars from session start before entries allowed |
| `entry_cutoff_minutes` | `30` | `[15, 30, 60]` | Minutes before session end to block new entries |
| `session_start` | `9` | fixed | Session start hour (BRT for B3; UTC for Crypto) |
| `session_end` | `18` | fixed | Session end hour (exclusive) |
| `vwap_volume_col` | `"tick_vol"` | — | Column name for volume weight (`tick_vol` or `volume`) |

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | n_sigma | sl_atr_mult | vwap_warmup_bars |
|-------|----|-------|---------|-------------|------------------|
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
**Regime sensitivity:** Expected to perform best on sessions with moderate intraday volatility and no large directional trends. Strong trending sessions will push VWAP in one direction and price may never revert within the session.

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- VWAP_std is sensitive to the distribution of tick_vol across the session; in the first hour of trading when volume is concentrated, the std will be compressed and bands will be narrow — this creates many false entries
- Using tick_vol as a proxy for actual volume introduces approximation error; on B3, real traded contracts are preferred when available
- A price trend during the session causes VWAP to trend in one direction; touching one band may be followed by price continuing away from VWAP rather than reverting
- Adding a slope filter on VWAP (close only when VWAP slope is near zero) could improve signal quality in directional sessions
- VWAP-based strategies can be monitored in real time and are susceptible to gaming by sophisticated market participants who know the levels

---

## Implementation

**Notebook:** `technical_analysis/11_vwap_intraday_reversion.ipynb`
**Source module:** `source/strategy.py` — `VWAPReversionStrategy`
**Parameters class:** `VWAPReversionParams`

### Implementation Notes

- The dynamic **VWAP-touch take profit** is **approximated** by a fixed
  ATR-based TP (`tp_atr_mult × ATR_stop`) — the existing `Backtester` only
  supports fixed-at-entry SL/TP.
- VWAP and bands reset cleanly at each session boundary using a
  `groupby(date)` cumulative sum on volume-weighted typical price.
- Volume weight is `tick_vol` (no contract-volume data is shipped; the doc
  notes this caveat).
- B3 only — no `data/crypto/` files. 1min timeframe dropped from the
  baseline (M1 source × WFO × 2 assets × 5 folds gets expensive); user can
  re-add via `GROUP_TIMEFRAMES`.
- Session-end forced close is handled by the Backtester via the
  `session_start` / `session_end` params (defaults `9` / `18`).

---

## References

1. Berkowitz, S.A., Logue, D.E., & Noser, E.A. (1988). The Total Cost of Transactions on the NYSE. *Journal of Finance*, 43(1), 97–112. Early institutional VWAP framework.
2. Madhavan, A. (2002). VWAP Strategies. *Transaction Performance: The Changing Face of Trading*, Investment Insights. ITG Inc.
3. Kissell, R., & Malamut, R. (2006). Algorithmic Decision Making Framework. *Journal of Trading*, 1(1), 12–21. VWAP execution benchmarks.
4. Obizhaeva, A., & Wang, J. (2013). Optimal Trading Strategy and Supply/Demand Dynamics. *Journal of Financial Markets*, 16(1), 1–32. Theoretical framework for price-impact and reversion around volume-weighted prices.
