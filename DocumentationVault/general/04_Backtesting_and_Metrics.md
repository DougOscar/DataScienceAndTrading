# Step 4 — Backtesting & Metrics

**Source:** `source/backtest.py`, `source/metrics.py` | **Notebook section:** §3.1, §3.2

## Running a Backtest

```python
result = Backtester(SMACrossoverStrategy(params), slippage_points=0.0).run(df)
```

`BacktestResult` contains:
- `result.trades` — DataFrame with one row per trade
- `result.equity` — cumulative PnL Series indexed by exit time

### Multi-Asset Portfolio

```python
per_asset, portfolio = run_baseline(asset_dfs, params)
```

`build_portfolio(per_asset_results)` merges individual `BacktestResult` objects:
1. Tags each trade with its `asset` column.
2. Sorts all trades by `exit_time` chronologically.
3. Computes a single cumulative equity series over all assets combined.

## Performance Metrics (`compute_metrics`)

| Metric | Description |
|--------|-------------|
| `num_trades` | Total trades executed |
| `total_pnl` | Sum of all trade PnL (price points) |
| `win_rate` | Fraction of trades with positive PnL |
| `profit_factor` | Gross profit / gross loss |
| `expectancy` | Average PnL per trade |
| `max_drawdown` | Peak-to-trough equity drawdown |
| `sharpe_daily` | Sharpe ratio on daily equity returns |
| `sharpe_per_trade` | Sharpe ratio on per-trade PnL |
| `p_value` | One-sided t-test: p(mean PnL > 0) |

## Baseline Results

### Forex — Portfolio Metrics by Timeframe

| TF | Trades | Total PnL | Win Rate | Profit Factor | Sharpe (daily) |
|----|--------|-----------|----------|---------------|----------------|
| 1h | 4,193 | -0.555 | 38.7% | 0.93 | -0.49 |
| 4h | 1,106 | -0.153 | 38.4% | 0.96 | -0.14 |
| 1D | 192 | -0.809 | 30.2% | 0.63 | -0.74 |

Best baseline Forex timeframe: **4h** (lowest negative Sharpe).

### B3 — Portfolio Metrics by Timeframe

| TF | Trades | Total PnL | Win Rate | Profit Factor | Sharpe (daily) |
|----|--------|-----------|----------|---------------|----------------|
| 1min | 31,989 | -31,315 | 38.7% | 0.98 | -0.36 |
| 5min | 6,247 | +14,746 | 39.7% | 1.02 | +0.16 |
| 15min | 2,100 | -3,645 | 38.5% | 0.99 | -0.04 |
| 30min | 992 | -4,467 | 40.0% | 0.99 | -0.05 |

Best baseline B3 timeframe: **5min** (only positive Sharpe at baseline params).

## Key Observations

- Forex baseline is broadly unprofitable — default params do not suit any timeframe.
- B3 5min shows marginal edge with baseline params (PF = 1.02, Sharpe = 0.16).
- Win rates are consistently ~38–40%; the strategy relies on asymmetric reward-to-risk (TP > SL multiplier), not win-rate dominance.
- B3 PnL is BRL price points; Forex PnL is dimensionless. **Do not compare raw PnL numbers across groups.**
- `p_value` on baseline Forex 1h = 0.04 and 1D = 0.003 — statistically significant but *negative* edge. High trade count inflates significance of a losing strategy.

## Comparing Timeframes

```python
metrics_comparison(portfolio_dict)
# Returns DataFrame: columns = timeframes, rows = metrics
```

Use this helper to build a side-by-side comparison table for any `{tf: portfolio}` dict.
