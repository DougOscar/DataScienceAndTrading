# Cross-Strategy Comparison Dashboard

> **Notebook:** `comparison/01_strategy_comparison_dashboard.ipynb`
> **Source module:** `source/comparison.py`
> **Plot helpers:** `source/dashboard.py` (`plot_strategy_*` functions)

A single notebook that runs **every** strategy in the registry across its
documented baseline `(group, timeframe, asset)` grid and renders six panels for
side-by-side comparison. The goal is to surface — at a glance — which approaches
win on which market, which travel between groups, and which actually diversify
each other.

This complements the per-strategy notebooks under `technical_analysis/` and
`machine_learning/`: those drill into one strategy with WFO and robustness;
this one stays shallow but spans every strategy.

---

## What the dashboard shows

| § | Panel | Purpose | What to look for |
|---|-------|---------|------------------|
| 3 | **Leaderboard** | Top-N `(strategy, group, tf, asset)` rows sorted by a chosen metric. | Cells worth a follow-up WFO. Treat anything with `num_trades < 50` as noise. |
| 4 | **Per-asset heatmap** | One `strategy × asset` heatmap per `(group, timeframe)`. Color centered at zero. | Vertical green bands = strategies that travel across assets. Lone bright cells = single-asset wins. |
| 5 | **Equity overlay** | Top-N strategies on one selected cell, plotted on the same axes. | Compounding curves vs. choppy ones. The leaderboard ranks by a number; this panel shows the *path*. |
| 6 | **Metric distribution** | Box plot of the metric across every cell, split by group. | Wide boxes = strategy lives or dies by cell choice (overfitting risk). Tight high boxes = robust outperformers. |
| 7 | **Return correlation** | Pairwise correlation of daily PnL on a benchmark asset. | Pairs near zero or negative *and* both with positive Sharpe = diversification candidates. |
| 8 | **Group breakdown** | Per-strategy mean of the metric, separated by group. | Bars positive in *both* groups = strategy that travels. Negative in one group = market-specific. |

The default metric is `sharpe_daily` because it's directly comparable across
markets with different price scales (forex 5-decimal pips vs. B3 contract
points). Switch the notebook's `METRIC` variable to `total_pnl`,
`profit_factor`, `win_rate`, `max_drawdown`, or anything else in the metrics
table to re-rank the panels.

---

## Inputs

The runner uses each strategy's **documented baseline parameters** — the same
values that the per-strategy notebook uses in its §3 "baseline backtest" cell.
These are encoded in `STRATEGY_REGISTRY` in `source/comparison.py`:

```python
StrategyRegistration(
    name="EMA Ribbon + RSI",
    notebook_id="06",
    strategy_cls=EMARibbonStrategy,
    params_cls=EMARibbonParams,
    group_timeframes={"forex": ["1h", "4h"], "b3": ["15min", "1h"]},
    base_kwargs=dict(ema_p1=5, ema_p2=8, ema_p3=13, ema_p4=21,
                     rsi_period=14, rsi_overbought=65.0, rsi_oversold=35.0,
                     atr_period=14, sl_atr_mult=2.0, tp_atr_mult=3.0),
    b3_overrides=dict(session_start=9, session_end=18),
)
```

`base_kwargs` is the forex configuration. `b3_overrides` (when present) is the
B3-only patch — typically the session window. Intraday strategies (Dual Thrust,
VWAP) bake their session window into `base_kwargs` since they're B3-only.

Walk-forward optimization is intentionally **out of scope** for this dashboard —
running WFO across every strategy × cell would explode runtime and the value
the dashboard provides comes from the *unoptimized* baseline. A WFO-comparison
dashboard is tracked as a follow-up to issue #16.

---

## Caching

Each strategy's three result tables are cached as parquet under
`comparison/.cache/`:

```
<slug>__<fingerprint>__metrics.parquet
<slug>__<fingerprint>__equity.parquet
<slug>__<fingerprint>__trades.parquet
```

The `<fingerprint>` is a 10-character SHA-1 of
`(name, base_kwargs, b3_overrides, group_timeframes)`. **Edit any of those four
fields in `source/comparison.py` and the cache for that strategy auto-invalidates**
on the next run.

* To force a rerun of a single strategy: change a registry field for that
  strategy (e.g. swap a TF), or delete its three cache files.
* To force a rerun of everything: `rm -rf comparison/.cache/` or call
  `run_all_strategies(..., force=True)`.

The cache directory is `.gitignore`d.

---

## How to add a new strategy to the registry

1. Implement the strategy following [[03_Strategy_Definition]] and the existing
   per-strategy notebooks.
2. Add a `StrategyRegistration(...)` entry to `STRATEGY_REGISTRY` in
   `source/comparison.py`. Mirror the `GROUP_TIMEFRAMES` and `baseline_params`
   from the strategy's notebook §3.
3. Re-run the dashboard notebook. Existing strategies hit cache; the new one
   runs once and is then cached.

No changes to the plot helpers are needed — they all consume the long-format
metrics / equity / trades frames produced by `run_all_strategies`.

---

## Cross-references

* Per-strategy notebooks: `technical_analysis/NN_*.ipynb`,
  `machine_learning/NN_*.ipynb`
* Strategy docs: [[../strategies/00_Index]]
* Backtest + metrics convention: [[04_Backtesting_and_Metrics]]
* Why baselines (not WFO results) here: shallow-but-wide first pass; deep
  WFO sweep happens inside each strategy's own notebook (see
  [[05_Walk_Forward_Optimization]]).
