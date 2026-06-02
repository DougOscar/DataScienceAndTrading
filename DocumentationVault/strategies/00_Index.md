# Strategy Knowledge Base

Each strategy document defines an entry signal, exit signal, and risk management profile in enough detail to re-implement and test independently.

Use [[_Template]] when documenting a new strategy.

## Strategies

| # | Name | Type | Markets | Timeframes | Status |
|---|------|------|---------|------------|--------|
| 01 | [[01_SMA_Crossover_ATR_Risk\|SMA Crossover + ATR Risk]] | Trend-following | Forex, B3 | 1h–1D / 1min–30min | Backtested |
| 02 | [[02_RSI_Mean_Reversion\|RSI Mean Reversion + ATR Stops]] | Mean-reversion | Forex, Crypto | 15min–4h | Idea |
| 03 | [[03_Bollinger_Band_Squeeze_Breakout\|Bollinger Band Squeeze Breakout]] | Breakout / Volatility expansion | Forex, B3, Crypto | 1h–1D | Idea |
| 04 | [[04_MACD_Histogram_Momentum\|MACD Histogram Momentum]] | Trend-following / Momentum | Forex, B3, Crypto | 1h–1D | Idea |
| 05 | [[05_Donchian_Channel_Breakout\|Donchian Channel Trend Following]] | Trend-following / Breakout | Forex, B3, Crypto | 4h–1D | Idea |
| 06 | [[06_EMA_Ribbon_RSI_Filter\|EMA Ribbon with RSI Filter]] | Trend-following | Forex, B3, Crypto | 15min–4h | Idea |
| 07 | [[07_Keltner_Channel_Mean_Reversion\|Keltner Channel Mean Reversion]] | Mean-reversion | Forex, Crypto | 15min–1h | Idea |
| 08 | [[08_Hurst_Exponent_Regime_Switcher\|Hurst Exponent Regime Switcher]] | Statistical / Adaptive | Forex, B3, Crypto | 1h–1D | Idea |
| 09 | [[09_Dual_Thrust_Intraday\|Dual Thrust Intraday Breakout]] | Breakout / Intraday | B3, Crypto | 1min–15min | Idea |
| 10 | [[10_HMM_Regime_Filter\|HMM Regime Filter]] | Machine Learning / Statistical | Forex, B3, Crypto | 1h–1D | Idea |
| 11 | [[11_VWAP_Intraday_Reversion\|VWAP Intraday Mean Reversion]] | Mean-reversion / Intraday | B3, Crypto | 1min–15min | Idea |
| 12 | [[12_Stochastic_Oscillator_Divergence\|Stochastic Oscillator + Divergence]] | Mean-reversion / Momentum | Forex, B3, Crypto | 15min–4h | Idea |
| 13 | [[13_Linear_Regression_Channel\|Linear Regression Channel]] | Statistical / Dual-mode | Forex, B3, Crypto | 1h–1D | Idea |
| 14 | [[14_FFT_Cycle_Filter\|FFT Cycle Filter & Forward Projection]] | Statistical / Cycle-following | Forex, B3, Crypto | 1h–1D | Idea |
| 15 | [[15_Multi_Filter_Portfolio_System\|Multi-Filter Portfolio System]] | Trend-following / Multi-confirmation portfolio | Forex, Crypto, B3 | 15min–1D | Backtested |

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
