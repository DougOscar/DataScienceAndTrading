# Findings & Next Steps

**Notebook section:** §6

## What the Baseline Establishes

- A reproducible pipeline: load → multi-timeframe resample → baseline backtest → WFO → robustness, run independently for Forex and B3 groups.
- Cross-timeframe comparison tables for each group identifying which horizon the strategy fits best.
- A concrete baseline to beat with more sophisticated strategies.

---

## Key Findings

### Best Configuration Per Group (WFO-optimized + Extensions)

| Group | Best TF | Session Filter | Sizing | Portfolio Weights | Approx Sharpe |
|-------|---------|---------------|--------|-------------------|---------------|
| Forex | 1D | N/A | unit | min_var | ~-0.61 (baseline) / better with WFO params |
| B3 | 30min | 09:00–18:00 | vol_scaled | min_var | ~+0.89 (combined, not yet jointly tested) |

### Strategy Characteristics

- Win rate is consistently 38–40% across all configurations.
- The strategy profits through asymmetric R:R (TP > SL in ATR multiples), not hit-rate dominance.
- Optimal parameters vary significantly per asset — no single universal setting works well.

### Regime Sensitivity (Critical Finding)

**Forex 1D:** Strong edge 2016–2018 (Sharpe 0.83–2.07/year). Consistently losing 2019–2024. Minor recovery in 2025. Regime change likely tied to post-2018 shift in Forex volatility/trend structure.

**B3 30min:** Strong edge 2021–2022 (Sharpe 1.50–2.69/year). Consistent losses 2023–2025. The overall positive WFO PnL is almost entirely explained by the first two years of data.

**Implication:** Both groups show that historical profitability does not persist. A regime filter or adaptive strategy is needed before live deployment.

---

## Intentional Simplifications (Not Bugs)

| Simplification | Impact | Fix for Production |
|----------------|--------|--------------------|
| Bar-close execution | Overestimates fill quality | Add slippage model |
| No commissions | Overstates net PnL | Apply per-trade cost |
| Unit sizing | Risk not normalized | `vol_scaled` or `fixed_frac` |
| Intra-bar SL before TP | Understates TP hits | Tick-level simulation |
| Fixed WFO folds | May not adapt to regime shifts | Anchored or expanding WFO |
| No cross-group PnL normalization | B3/Forex PnL not comparable | Normalize by notional or use % returns |

---

## Recommended Next Steps

### Immediate (Strategy Improvements)

1. **Combine extensions** — test session filter + vol_scaled sizing + min_var weights jointly on B3 15min and 30min.
2. **Regime detection** — add a simple regime filter (e.g. ADX threshold, 200-day MA slope) to disable the strategy in non-trending regimes.
3. **Expanding WFO** — replace fixed-fold WFO with anchored expanding-window WFO to better simulate live re-optimization.

### Medium-Term (Strategy Development)

4. **ML signal filter** — see `machine_learning/01_classifier_signal_filter.ipynb` — overlay a classifier that filters out low-quality crossover signals.
5. **Commission modeling** — add realistic per-trade costs for both groups (B3 mini-futures have known commission schedules).
6. **Multi-strategy portfolio** — extend `weighted_portfolio` to combine different strategies, not just different assets.

### Infrastructure

7. **Data quality checks** — automated detection of gaps, price spikes, and weekend bars.
8. **Live signal generation** — adapt `SMACrossoverStrategy` to produce signals from streaming bar data.
9. **Reporting** — scheduled notebook execution to update metrics as new data arrives.

---

## Baseline Numbers for Reference (Summary)

| Metric | Forex 1D (WFO-opt) | B3 30min (WFO-opt) |
|--------|-------------------|-------------------|
| Trades (full history) | 264 | 629 |
| Total PnL | +0.396 pts | +9,714 pts |
| Sharpe (daily) | +0.40 | +0.24 |
| Profit Factor | 1.26 | 1.11 |
| Max Drawdown | -0.24 | -34,052 |
| Block Bootstrap P(profitable) | 89% | 67% |
| Consistent years | 2016–2018 | 2021–2022 |
