# Quantitative Trading Framework — Overview

This vault documents the applied data science process for researching, testing, and validating algorithmic trading strategies across multiple market groups and timeframes.

## Pipeline Summary

```
Data → Resample → Strategy → Backtest → WFO → Robustness → Extensions
```

| Step | File | Purpose |
|------|------|---------|
| 1 | [[01_Data_Loading]] | Load OHLC CSVs, tag by market group |
| 2 | [[02_MultiTimeframe_Preparation]] | Resample M1 data to target timeframes |
| 3 | [[03_Strategy_Definition]] | Define entry/exit rules and parameters |
| 4 | [[04_Backtesting_and_Metrics]] | Run backtests, compute performance metrics |
| 5 | [[05_Walk_Forward_Optimization]] | In-sample optimization + OOS validation |
| 6 | [[06_Robustness_Testing]] | Monte Carlo, block bootstrap, sensitivity |
| 7 | [[07_Extensions]] | Session filter, position sizing, portfolio weighting |

## Market Groups

| Group | Assets | Timeframes Tested | Data Range |
|-------|--------|-------------------|------------|
| Forex | EURUSD, EURCAD, GBPCHF | 1h, 4h, 1D | 2016–2026 |
| B3 (Bovespa) | WDO, WIN (mini-futures) | 1min, 5min, 15min, 30min | 2021–2026 |

## Source Code Layout

```
source/
├── data_loader.py   — load_all, resample_ohlc
├── strategy.py      — SMACrossoverStrategy, StrategyParams
├── backtest.py      — Backtester
├── metrics.py       — compute_metrics, metrics_table
├── dashboard.py     — plot_backtest_dashboard, plot_wfo_dashboard, plot_robustness_dashboard
├── wfo.py           — walk_forward, WFOResult
├── robustness.py    — monte_carlo_trades, block_bootstrap_trades, subperiod_analysis, parameter_sensitivity
└── portfolio.py     — correlation_weights, weighted_portfolio
```

## Strategy Knowledge Base

See [[strategies/00_Index|Strategy Index]] for documented strategies and the evaluation checklist.

## Notebooks

- `technical_analysis/01_baseline_sma_crossover.ipynb` — full baseline pipeline (Sections 1–7)
- `machine_learning/01_classifier_signal_filter.ipynb` — ML signal overlay (planned)

## Key Design Decisions

- Groups are evaluated **independently** — Forex and B3 metrics are not compared directly because B3 PnL is BRL-denominated price points while Forex is dimensionless (pips/price).
- All source components live in `source/` so multiple notebooks share the same logic.
- Baseline parameters are fixed across all timeframes; WFO finds per-(group, timeframe) optimal values.
- Bar-close execution with pessimistic intra-bar assumption (SL checked before TP).

## Results at a Glance (Baseline Params — fast=20, slow=50, sl=2×ATR, tp=3×ATR)

| Group | Best Baseline TF | Sharpe (daily) | Profit Factor |
|-------|-----------------|---------------|---------------|
| Forex | 4h | -0.14 | 0.96 |
| B3    | 5min | +0.16 | 1.02 |

> Baseline is intentionally unoptimized. WFO-optimized results are in [[05_Walk_Forward_Optimization]].
