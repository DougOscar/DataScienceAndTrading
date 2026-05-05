# Step 7 — Extensions

**Notebook section:** §7

These extensions are implemented and tested within the baseline notebook. Each can be toggled independently via `StrategyParams`.

---

## 7.1 Session / Time-of-Day Filter (B3)

**Motivation:** B3 mini-futures (WDO, WIN) officially trade 09:00–18:00 local time. Data extends to 18:29, but bars after 18:00 are low-liquidity and distort signals.

**Effect of the filter:**
1. Suppresses new entry signals outside the session window.
2. Forces positions to close at the last bar inside the window (avoids holding through the liquidity gap).

**Usage:**

```python
session_params = StrategyParams(
    fast=20, slow=50, atr_period=14, sl_atr_mult=2.0, tp_atr_mult=3.0,
    session_start=9,   # no new entries before 09:00
    session_end=18,    # force close at 18:00
)
```

**Results — Baseline B3 with vs without session filter:**

| TF | Base Sharpe | Session Sharpe | Base PnL | Session PnL |
|----|-------------|---------------|----------|-------------|
| 1min | -0.36 | -0.49 | -31,315 | -42,162 |
| 5min | +0.16 | +0.18 | +14,746 | +15,891 |
| 15min | -0.04 | +0.29 | -3,645 | +24,792 |
| 30min | -0.05 | +0.03 | -4,467 | +2,252 |

**Key finding:** Session filter significantly improves 15min (+0.33 Sharpe improvement). Mixed results elsewhere — it helps on 15min/30min but mildly hurts 1min.

---

## 7.2 Position Sizing

**Motivation:** Unit sizing ignores per-trade risk magnitude. Volume-scaled or fixed-fractional sizing can improve risk-adjusted returns.

**Three modes compared:**

| Mode | Size Formula | PnL Unit |
|------|-------------|----------|
| `unit` | 1 contract always | price points |
| `vol_scaled` | `1 / ATR` | ATR-normalized (dimensionless) |
| `fixed_frac` | `equity × risk_fraction / (sl_mult × ATR)` | price points × contracts |

**Usage:**

```python
params = StrategyParams(
    fast=20, slow=50, atr_period=14, sl_atr_mult=2.0, tp_atr_mult=3.0,
    sizing_mode="fixed_frac",
    risk_fraction=0.01,   # risk 1% of equity per trade
)
result = Backtester(SMACrossoverStrategy(params), initial_capital=100_000).run(df)
```

**Results (best-OOS timeframes at baseline params):**

### Forex 1D

| Mode | Sharpe | PF | Max DD | Total PnL |
|------|--------|----|--------|-----------|
| unit | -0.74 | 0.63 | -0.80 | -0.81 |
| vol_scaled | -0.70 | 0.66 | -85.4 | -84.98 |
| fixed_frac | -0.72 | 0.65 | -39,597 | -40,004 |

Sizing doesn't salvage a losing strategy — all modes remain negative on Forex 1D baseline.

### B3 30min

| Mode | Sharpe | PF | Max DD | Total PnL |
|------|--------|----|--------|-----------|
| unit | -0.05 | 0.99 | -27,662 | -4,467 |
| vol_scaled | +0.17 | 1.04 | -73.7 | +38.1 |
| fixed_frac | +0.12 | 1.03 | -45,679 | +15,554 |

`vol_scaled` improves B3 30min — ATR normalization equalizes trade risk, turning a marginal loser into a mild winner.

---

## 7.3 Correlation-Aware Portfolio Weighting

**Motivation:** Within a group, assets may be correlated (EURUSD and EURCAD share the EUR leg). Equal nominal weight over-concentrates risk on correlated factors.

**Three methods:**

| Method | Logic |
|--------|-------|
| `equal` | 1/n per asset — ignores correlation |
| `inv_vol` | weight ∝ 1/σ — lower-volatility assets get higher weight |
| `min_var` | minimum-variance portfolio via Σ⁻¹·1 (long-only constrained) |

**Usage:**

```python
weights = correlation_weights(eq_curves, lookback_days=60, method="min_var")
trades_w, equity_w = weighted_portfolio(per_asset_results, weights)
```

PnL is reported as `pnl_weighted = pnl_points × weight`, making Sharpe comparable across methods.

**Results — Baseline params:**

### Forex 1D

| Method | Weights (EUR/USD/GBP) | Sharpe | PF | Max DD |
|--------|----------------------|--------|----|--------|
| equal | 0.333 / 0.333 / 0.333 | -0.74 | 0.63 | -0.27 |
| inv_vol | 0.292 / 0.416 / 0.292 | -0.68 | 0.65 | -0.24 |
| min_var | 0.250 / 0.504 / 0.246 | -0.61 | 0.68 | -0.21 |

`min_var` reduces drawdown by 22% and improves Sharpe by 0.13 by overweighting EURUSD (lower correlated volatility).

### B3 30min

| Method | Weights (WDO / WIN) | Sharpe | PF | Total PnL |
|--------|---------------------|--------|----|-----------|
| equal | 0.50 / 0.50 | -0.05 | 0.99 | -2,234 |
| inv_vol | 0.98 / 0.02 | +0.38 | 1.09 | +1,312 |
| min_var | 1.00 / 0.00 | +0.51 | 1.16 | +1,450 |

`min_var` almost entirely eliminates WIN (high variance, negative contribution), resulting in near-WDO-only allocation and turning a losing portfolio profitable.

---

## Extension Interaction Matrix

| | Session Filter | vol_scaled | min_var Weights |
|--|----------------|-----------|-----------------|
| B3 15min | +0.33 Sharpe ↑ | — | — |
| B3 30min | +0.08 Sharpe ↑ | +0.22 Sharpe ↑ | +0.56 Sharpe ↑ |
| Forex 1D | N/A | marginal | +0.13 Sharpe ↑ |

These extensions can be combined — session filter + min_var weighting is the highest-value combination for B3 and has not been jointly tested yet.
