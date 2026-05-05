# Step 5 — Walk-Forward Optimization (WFO)

**Source:** `source/wfo.py` | **Notebook section:** §4

## Purpose

Prevent in-sample overfitting by repeatedly optimizing on a training window and then evaluating on a held-out window. The stitched out-of-sample (OOS) equity curve is an honest estimate of live performance.

## WFO Configuration

```python
walk_forward(df, param_grid=param_grid, n_splits=5, oos_ratio=0.25)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n_splits` | 5 | Number of equal temporal folds |
| `oos_ratio` | 0.25 | 25% of each fold is OOS; 75% is in-sample |
| Objective | `sharpe_daily` | Metric maximized during grid search |

Each (group, timeframe) runs a separate WFO independently — no cross-group parameter sharing.

## WFO Process per Fold

```
Full history split into 5 equal blocks:
[ IS₁  OOS₁ ][ IS₂  OOS₂ ][ IS₃  OOS₃ ][ IS₄  OOS₄ ][ IS₅  OOS₅ ]
  75%   25%    75%   25%    75%   25%    75%   25%    75%   25%
```

1. Grid-search all 625 parameter combinations on the IS window.
2. Select params with highest daily Sharpe.
3. Run backtest on the OOS window with those params.
4. Stitch OOS trades from all folds chronologically.

## Combining Multi-Asset OOS

`build_combined_wfo(wfo_per_asset_tf)` merges per-asset `WFOResult` objects:
- Combines OOS trades from all assets, sorted by `exit_time`.
- Computes a single portfolio equity series.
- Averages `degradation_ratio` across assets per fold.

## Degradation Ratio

`degradation_ratio = OOS_sharpe / IS_sharpe`

- Value near 1.0 → OOS performance matches IS (low overfitting).
- Value < 0 → OOS performance is negative despite positive IS (overfit signal).

## OOS Trade Counts

| Group | TF | OOS Trades |
|-------|----|-----------|
| Forex | 1h | 983 |
| Forex | 4h | 201 |
| Forex | 1D | 38 |
| B3 | 1min | 5,416 |
| B3 | 5min | 1,384 |
| B3 | 15min | 252 |
| B3 | 30min | 154 |

Note: 1D Forex has only 38 OOS trades — statistical conclusions are limited.

## Optimized Parameters (Most Frequent WFO Selection)

`pick_best_params(wfo_windows, fallback)` returns the parameter combination most frequently chosen across the 5 folds.

Selected parameters per asset:

| Group | TF | Asset | fast | slow | sl | tp |
|-------|----|-------|------|------|----|----|
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

## Optimized Full-History Backtest Metrics

### Forex

| TF | Trades | Sharpe | Profit Factor | Max DD |
|----|--------|--------|---------------|--------|
| 1h | 4,192 | 0.069 | 1.011 | -0.58 |
| 4h | 907 | 0.160 | 1.050 | -0.35 |
| 1D | 264 | 0.404 | 1.262 | -0.24 |

### B3

| TF | Trades | Sharpe | Profit Factor | Max DD |
|----|--------|--------|---------------|--------|
| 1min | 13,098 | 0.332 | 1.023 | -28,488 |
| 5min | 4,287 | -0.294 | 0.959 | -39,077 |
| 15min | 958 | 0.342 | 1.098 | -19,534 |
| 30min | 629 | 0.241 | 1.107 | -34,052 |

## Best OOS Timeframe by Group

Selected automatically by highest OOS daily Sharpe:

| Group | Best TF | Used for Robustness in §6 |
|-------|---------|--------------------------|
| Forex | **1D** | Yes |
| B3 | **30min** | Yes |

## Known Limitations

- WFO folds are fixed equal slices — anchored or expanding-window WFO is a natural follow-up.
- Small OOS trade counts (especially 1D Forex) limit statistical power.
- Parameter stability across folds is not validated — wide variation suggests unstable edge.
