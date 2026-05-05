# Step 6 — Robustness Testing

**Source:** `source/robustness.py` | **Notebook section:** §5

## Purpose

Validate that strategy performance is not a statistical artifact of trade sequencing, lucky parameter selection, or a specific market regime. Tests are run on the WFO-optimized trades for the **best OOS timeframe** per group.

Best timeframes selected: **Forex → 1D**, **B3 → 30min**

---

## Test 1: Monte Carlo Trade Shuffling

**Function:** `monte_carlo_trades(trades, n_runs=1000, seed=42)`

**Method:** Randomly shuffles the sequence of trades 1,000 times. Each shuffle produces a new equity curve. Since the same trades occur but in random order, this isolates whether the *sequence* of wins/losses (streak structure) contributes to drawdown profile.

**Output summary:**

| Metric | Forex 1D | B3 30min |
|--------|----------|----------|
| Mean final PnL | 0.3961 | 9,714 |
| Median final PnL | 0.3961 | 9,714 |
| p5 final PnL | 0.3961 | 9,714 |
| p95 final PnL | 0.3961 | 9,714 |
| Mean max drawdown | -0.174 | -12,808 |
| p95 max drawdown | -0.259 | -19,027 |
| Prob profitable | 100% | 100% |

Final PnL is constant across shuffles (expected — total PnL is order-independent). Drawdown varies, giving a distribution of worst-case scenarios.

---

## Test 2: Block Bootstrap

**Function:** `block_bootstrap_trades(trades, n_runs=1000, seed=42)`

**Method:** Samples blocks of consecutive trades with replacement to generate synthetic trade sequences of the same length. Unlike pure shuffling, this preserves local autocorrelation in trade outcomes (winning/losing streaks) while testing global distributional stability.

**Output summary:**

| Metric | Forex 1D | B3 30min |
|--------|----------|----------|
| Mean final PnL | 0.3961 | 9,758 |
| Median final PnL | 0.3834 | 8,581 |
| p5 final PnL | -0.0999 | -22,657 |
| p95 final PnL | +0.9397 | +46,074 |
| Mean max drawdown | -0.200 | -18,779 |
| p95 max drawdown | -0.345 | -34,978 |
| **Prob profitable** | **89.4%** | **66.8%** |

**Key insight:** Forex 1D has 89% probability of profitability across bootstrap samples. B3 30min drops to 67% — the edge is narrower and more sensitive to trade-block composition.

---

## Test 3: Sub-Period Analysis

**Function:** `subperiod_analysis(trades, freq="YE")`

Breaks trades into annual periods and computes full metrics for each year. Reveals regime sensitivity.

### Forex 1D — Year-by-Year

| Year | Trades | Total PnL | Sharpe | PF |
|------|--------|-----------|--------|----|
| 2016 | 15 | +0.165 | 2.07 | 3.79 |
| 2017 | 23 | +0.224 | 1.84 | 3.45 |
| 2018 | 27 | +0.114 | 0.83 | 1.72 |
| 2019 | 25 | -0.048 | -0.53 | 0.70 |
| 2020 | 30 | +0.028 | 0.32 | 1.13 |
| 2021 | 31 | -0.065 | -0.69 | 0.67 |
| 2022 | 24 | -0.011 | -0.02 | 0.93 |
| 2023 | 25 | -0.035 | -0.33 | 0.78 |
| 2024 | 29 | -0.044 | -0.51 | 0.73 |
| 2025 | 26 | +0.085 | 0.81 | 1.78 |

**Pattern:** Strong edge in 2016–2018 (trending Forex environment), deteriorating from 2019 onward. Possible regime change.

### B3 30min — Year-by-Year

| Year | Trades | Total PnL | Sharpe | PF |
|------|--------|-----------|--------|----|
| 2021 | 63 | +19,926 | 2.69 | 3.54 |
| 2022 | 126 | +14,696 | 1.50 | 1.86 |
| 2023 | 134 | -10,190 | -1.20 | 0.61 |
| 2024 | 128 | -5,163 | -0.96 | 0.69 |
| 2025 | 138 | -11,685 | -2.42 | 0.37 |
| 2026 | 40 | +2,130 | 0.83 | 1.47 |

**Pattern:** Strong edge in 2021–2022, sharp deterioration from 2023. Consistent with a regime shift. B3 30min overall positive PnL is almost entirely driven by early years.

---

## Test 4: Parameter Sensitivity

**Method:** One-at-a-time sweeps — each parameter varied across a range while others stay at baseline.

**Variations tested:**

```python
variations = {
    "fast":        [10, 15, 20, 25, 30],
    "slow":        [30, 50, 80, 120],
    "sl_atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
    "tp_atr_mult": [1.5, 2.0, 3.0, 4.0, 5.0],
}
```

A robust edge degrades **gradually** as parameters move away from optimal. A cliff-edge response indicates curve fitting.

---

## Test 5: Synthetic Asset Simulation

**Notebook section:** §5.4

**Method:** Generate synthetic OHLC assets by bootstrapping first-differences from real data, then run the strategy on these fabricated assets.

**Configuration:**

| Parameter | Value |
|-----------|-------|
| `SYNTH_INIT_WINDOW` | 5 bars (seed from real data) |
| `SYNTH_BUILD_BLOCK` | 1 consecutive diff per generation step |
| `SYNTH_MAX_TRADES` | 2,500 per asset |
| `SYNTH_ASSETS_PER_GROUP` | 5 |
| `SYNTH_MAX_BARS` | 250,000 |

**Interpretation:** If metrics on synthetic assets cluster around real-data values, the edge may be explained purely by autocorrelation structure of price differences, not a genuine market inefficiency. A notable divergence (better real-data metrics) provides evidence of genuine edge.

---

## Robustness Summary

| Test | Forex 1D | B3 30min |
|------|----------|----------|
| Monte Carlo (prob profitable) | 100% | 100% |
| Block Bootstrap (prob profitable) | 89% | 67% |
| Sub-period consistency | Good 2016–2018, declining 2019+ | Good 2021–2022, declining 2023+ |
| Regime concern | Yes (post-2018) | Yes (post-2022) |

Both groups show regime-sensitive performance. The WFO-optimized strategies are not unconditionally robust across all periods.
