# Strategy Name

> **Type:** (Trend-following / Mean-reversion / Breakout / etc.)
> **Markets:** (Forex, B3, Equities, etc.)
> **Timeframes:** (1min, 5min, 1h, 1D, etc.)
> **Direction:** (Long-only / Short-only / Long & Short)
> **Status:** (Idea / In Development / Backtested / Live)

---

## Overview

_One paragraph describing the market hypothesis behind the strategy. Why should this work? What market inefficiency or structural feature is it exploiting?_

---

## Indicators

> Define every indicator used — precise enough to reproduce from raw OHLC data with no ambiguity.

### Indicator 1 — Name

- **Input:** (e.g. Close prices)
- **Formula:** (exact formula or algorithm)
- **Lookback:** (minimum bars required before first valid value)
- **Parameters:** (list all tunable params with default values)

### Indicator 2 — Name

_(repeat for each indicator)_

---

## Entry Signal

### Long Entry

Condition (all must be true):
1. ...
2. ...

Execution:
- **Price:** (at bar close / at open of next bar / at limit X / etc.)
- **Bar:** (the signal bar / the next bar)

### Short Entry

_(same structure — or "Mirror of Long Entry" if fully symmetric)_

### Entry Filters

_Any additional conditions that block an otherwise valid signal:_
- Session filter: ...
- Volatility filter: ...
- Trend filter: ...

---

## Exit Signal

### Primary Exits (Price-based)

| Exit Type | Long Condition | Short Condition | Exit Price |
|-----------|---------------|-----------------|-----------|
| Stop Loss | ... | ... | ... |
| Take Profit | ... | ... | ... |

### Secondary Exits (Signal-based)

| Exit Type | Condition | Exit Price |
|-----------|-----------|-----------|
| Signal reversal | Opposite entry signal fires | Bar close |
| Session end | ... | ... |

### Exit Priority

When multiple exits are valid on the same bar:
1. (highest priority) ...
2. ...
3. ...

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions | | Per asset / per portfolio |
| Risk per trade | | % of equity or fixed units |
| Stop type | Fixed / Dynamic / Trailing | |
| Stop loss | | Formula |
| Take profit | | Formula |
| Trailing stop | Yes / No | If yes, describe update logic |
| Position sizing mode | unit / vol_scaled / fixed_frac | |
| Pyramiding | Yes / No | |

### Markets

| Group | Assets | Conditions |
|-------|--------|-----------|
| | | |

### Time Restrictions

| Rule | Description |
|------|-------------|
| Session filter | |
| Days of week | |
| News/event blackout | |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| | | | |

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | Param1 | Param2 | ... |
|-------|----|-------|--------|--------|-----|
| | | | | | |

---

## Performance Summary

> Link to backtest results and robustness analysis.

| Metric | Baseline | WFO-Optimized | Notes |
|--------|----------|--------------|-------|
| Sharpe (daily) | | | |
| Profit Factor | | | |
| Win Rate | | | |
| Max Drawdown | | | |
| P(profitable) block bootstrap | | | |

**Best configuration:** ...
**Regime sensitivity:** ...

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- ...

---

## Implementation

**Notebook:** `technical_analysis/XX_name.ipynb`
**Source module:** `source/strategy.py` — `ClassName`
**Parameters class:** `StrategyParams`
