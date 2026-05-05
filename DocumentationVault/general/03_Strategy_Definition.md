# Step 3 — Strategy Definition

**Source:** `source/strategy.py` | **Notebook section:** §3

## Strategy: SMA Crossover with ATR-Based Risk

A fully-specified trend-following strategy using two Simple Moving Averages for signals and Average True Range for dynamic stop/target placement.

## Entry Rules

| Signal | Condition |
|--------|-----------|
| Long entry | Fast SMA crosses **above** Slow SMA (detected at bar close) |
| Short entry | Fast SMA crosses **below** Slow SMA (detected at bar close) |

Positions are entered at the **closing price** of the signal bar (no look-ahead).

## Exit Rules

| Exit type | Formula |
|-----------|---------|
| Stop loss | `entry_price − direction × sl_atr_mult × ATR` |
| Take profit | `entry_price + direction × tp_atr_mult × ATR` |
| Reversal | Triggered by an opposite crossover signal |

Intra-bar priority: **stop loss is checked before take profit** (pessimistic assumption — understates TP hits in reality).

## Position Sizing

No pyramiding. Always flat between signals (one position at a time). Default mode: 1 unit per trade; PnL reported in price points.

See [[07_Extensions]] for `vol_scaled` and `fixed_frac` sizing modes.

## Parameters (`StrategyParams`)

```python
baseline_params = StrategyParams(
    fast=20,          # fast SMA period
    slow=50,          # slow SMA period
    atr_period=14,    # ATR lookback period
    sl_atr_mult=2.0,  # stop loss = entry ± 2 × ATR
    tp_atr_mult=3.0,  # take profit = entry ± 3 × ATR
)
```

Baseline parameters are the same across all timeframes and groups. [[05_Walk_Forward_Optimization]] finds per-(group, timeframe) optimal values.

## WFO Parameter Search Grid

```python
param_grid = {
    "fast":        [5, 10, 20, 30, 50],
    "slow":        [40, 60, 100, 150, 200],
    "sl_atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
    "tp_atr_mult": [1.5, 2.0, 3.0, 4.0, 5.0],
}
```

Total combinations: 5 × 5 × 5 × 5 = 625. In-sample optimization objective: **maximize daily Sharpe**.

## Optional Parameters (Extensions)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `session_start` | None | Hour to begin accepting signals (e.g. 9 for B3) |
| `session_end` | None | Hour to close all positions (e.g. 18 for B3) |
| `sizing_mode` | `"unit"` | `"unit"` / `"vol_scaled"` / `"fixed_frac"` |
| `risk_fraction` | 0.01 | Fraction of equity risked per trade (fixed_frac mode) |

## Known Simplifications

- No commissions or realistic fill modeling.
- Bar-close execution assumes fills at the close price.
- Intra-bar SL-before-TP may understate TP hits vs. true tick-level simulation.
- No slippage by default (configurable via `Backtester(slippage_points=...)`).
