# Strategy Knowledge Base

Each strategy document defines an entry signal, exit signal, and risk management profile in enough detail to re-implement and test independently.

Use [[_Template]] when documenting a new strategy.

## Strategies

| # | Name | Type | Markets | Timeframes | Status |
|---|------|------|---------|------------|--------|
| 01 | [[01_SMA_Crossover_ATR_Risk\|SMA Crossover + ATR Risk]] | Trend-following | Forex, B3 | 1h–1D / 1min–30min | Backtested |

## Strategy Evaluation Checklist

Before a strategy moves from **Backtested** to **Live**, it should pass:

- [ ] Walk-forward OOS Sharpe > 0 on best timeframe
- [ ] Block bootstrap P(profitable) ≥ 70%
- [ ] Sub-period analysis: profitable in at least 60% of years/periods
- [ ] Parameter sensitivity: gradual degradation (no cliff-edge response)
- [ ] Synthetic asset test: real data outperforms synthetic average
- [ ] Commissions modeled and net PnL still positive
- [ ] Slippage sensitivity tested
- [ ] Regime filter or explicit regime scope defined

## Pipeline Reference

- [[00_Overview]] — full pipeline summary and source layout
- [[04_Backtesting_and_Metrics]] — how to run backtests and interpret metrics
- [[05_Walk_Forward_Optimization]] — how to run and read WFO results
- [[06_Robustness_Testing]] — robustness test suite
- [[07_Extensions]] — session filter, position sizing, portfolio weighting
