# Position Sizing — Formulas Reference

Complete mathematical derivations for all sizing methods with worked examples.

---

## Fixed Fractional Sizing

### Derivation

The goal: lose at most `R%` of account value if the stop loss is hit.

```
risk_amount = account_value × risk_percentage
price_risk_per_unit = |entry_price - stop_loss_price|
position_size_units = risk_amount / price_risk_per_unit
position_value = position_size_units × entry_price
leverage_ratio = position_value / account_value
```

### Worked Example

| Parameter | Value |
|-----------|-------|
| Account value | $10,000 |
| Risk per trade | 2% |
| Entry price | $1.50 |
| Stop loss | $1.30 |

```
risk_amount = $10,000 × 0.02 = $200
price_risk  = $1.50 - $1.30 = $0.20
units        = $200 / $0.20 = 1,000 units
position_value = 1,000 × $1.50 = $1,500 (15% of account)
```

If the stop is hit: loss = 1,000 × $0.20 = $200 = 2% of account. In MT5,
`units` must still be rounded to the symbol's `volume_step` — see the
"MT5 Lot Constraints" section below for the resulting risk-realisation error.

### Cost-Adjusted Version

In practice, include spread and commission in the risk (for MT5 FX/metals,
this is the actual `spread` column in points × point value, plus commission
— not a flat percentage-of-price fee):

```
effective_risk = price_risk + (spread_points × point_value) + commission_per_unit
position_size = risk_amount / effective_risk
```

Example with a 2-point spread and a small commission on a EURUSD-style
instrument (point value and commission expressed per unit, illustrative):

```
spread_cost = 2 points × $0.0001/point = $0.0002
commission_cost = $0.00005 per unit (round-trip)
effective_risk = $0.20 + $0.0002 + $0.00005 = $0.20025
units = $200 / $0.20025 = 999 units (vs 1,000 without costs)
```

The relative impact of spread/commission is small for a 20-pip-plus stop on
a liquid FX major, but grows materially for tight intraday stops (e.g. B3
scalps) — always check it rather than assuming it is negligible.

---

## Volatility-Adjusted Sizing

### Derivation

Normalize position sizes so each position contributes roughly the same dollar volatility to the portfolio.

```
target_daily_pnl_vol = account_value × target_vol_pct
instrument_daily_vol = price × daily_return_std
position_size_units = target_daily_pnl_vol / instrument_daily_vol
```

### Using ATR (Average True Range)

ATR is a smoothed measure of daily price range:

```
daily_vol_pct = ATR(14) / close_price
position_units = (account × target_vol_pct) / ATR(14)
```

### Worked Example

| Parameter | Value |
|-----------|-------|
| Account | $10,000 |
| Target daily vol | 2% of account ($200) |
| Instrument price | $1.50 |
| ATR(14) | $0.12 |

```
daily_vol_pct = $0.12 / $1.50 = 8%
position_units = $200 / $0.12 = 1,667 units
position_value = 1,667 × $1.50 = $2,500
expected_daily_pnl_range = 1,667 × $0.12 = $200 (2% of account)
```

### Comparing Across Assets

This method lets you hold equal-risk positions across assets with very different volatilities:

| Instrument | Price | ATR(14) | Vol % | Units | Value |
|-------|-------|---------|-------|-------|-------|
| EURUSD | 1.0850 | 0.0065 | 0.6% | 30,769 | $33,384 |
| XAUUSD | 2,650 | 28.00 | 1.1% | 7 | $18,550 |
| WIN (B3 mini-index) | 128,000 pts | 1,450 pts | 1.1% | 0.14 contracts | — |

Each contributes ~$200 daily PnL volatility despite very different
position sizes and contract structures — this is illustrative math, not a
claim about actual current prices; measure ATR on your own data. In MT5,
`Units` still round to `volume_step`, and B3 futures size in whole or
fractional contracts per the exchange's minimum.

---

## Kelly Criterion

### Full Derivation

The Kelly criterion maximizes the expected logarithm of wealth (geometric growth rate).

Given a binary bet with:
- Probability `p` of winning `b` units per unit risked
- Probability `q = 1 - p` of losing 1 unit per unit risked

Maximize: `E[ln(W)] = p × ln(1 + f×b) + q × ln(1 - f)`

Taking the derivative and setting to zero:

```
dE/df = p×b/(1+f×b) - q/(1-f) = 0
p×b×(1-f) = q×(1+f×b)
p×b - p×b×f = q + q×f×b
p×b - q = f×b×(p + q) = f×b
f* = (p×b - q) / b
```

Equivalent forms:
```
f* = (p×b - q) / b
f* = p - q/b
f* = (p×(b+1) - 1) / b
```

### Fractional Kelly

Full Kelly maximizes growth but produces large drawdowns. The variance of returns under Kelly is:

```
Var = p×q×(b+1)² × f²
```

Fractional Kelly reduces variance quadratically while reducing growth only linearly:

| Fraction | Growth Rate | Drawdown Risk | Recommended? |
|----------|-------------|---------------|-------------|
| 1.0x | 100% | Very high | No |
| 0.5x | 75% | Moderate | For well-measured edges |
| 0.25x | 44% | Low | Default recommendation |
| 0.1x | 19% | Very low | Ultra-conservative |

Growth rate at fraction `g`: `G(g) = g × (2 - g) × G(1)` approximately.

### Worked Examples

**Example 1: Moderate edge**
```
Win rate = 55%, avg win = 1.5x avg loss
p = 0.55, q = 0.45, b = 1.5
f* = (0.55 × 1.5 - 0.45) / 1.5 = 0.25 / 1.5 = 0.167 (16.7%)
Quarter Kelly: 4.2% risk per trade
Half Kelly: 8.3% risk per trade
```

**Example 2: High win rate, small payoff**
```
Win rate = 70%, avg win = 0.8x avg loss
p = 0.70, q = 0.30, b = 0.8
f* = (0.70 × 0.8 - 0.30) / 0.8 = 0.26 / 0.8 = 0.325 (32.5%)
Quarter Kelly: 8.1%
```

**Example 3: Low win rate, large payoff (trend following)**
```
Win rate = 35%, avg win = 4x avg loss
p = 0.35, q = 0.65, b = 4.0
f* = (0.35 × 4.0 - 0.65) / 4.0 = 0.75 / 4.0 = 0.1875 (18.75%)
Quarter Kelly: 4.7%
```

**Example 4: No edge**
```
Win rate = 45%, avg win = 1.0x avg loss
f* = (0.45 × 1.0 - 0.55) / 1.0 = -0.10
Negative Kelly → NO EDGE → do not trade
```

---

## MT5 Lot Constraints

### Lot Rounding

Every MT5 symbol has a minimum lot (`volume_min`), a lot step
(`volume_step`), and a maximum lot (`volume_max`). A computed "raw" size
from any formula above must be snapped to this grid:

```
lots = clamp(round_to_step(raw_lots, volume_step), volume_min, volume_max)
```

### Worked Example

| Parameter | Value |
|-----------|-------|
| Target risk | $200 |
| Stop distance | 30 points |
| Point value per lot | $10 |
| `volume_step` | 0.01 |

```
raw_lots = 200 / (30 × 10) = 0.667
lots = round(0.667 / 0.01) × 0.01 = 0.67
actual_risk = 0.67 × 30 × 10 = $201.00   (0.5% over target — acceptable)
```

### Risk-Realisation Error Near `volume_min`

| Parameter | Value |
|-----------|-------|
| Target risk | $8 (small/micro account) |
| Stop distance | 30 points |
| Point value per lot | $10 |
| `volume_min` | 0.01 |

```
raw_lots = 8 / (30 × 10) = 0.0267
lots = max(0.01, round(0.0267 / 0.01) × 0.01) = 0.03
actual_risk = 0.03 × 30 × 10 = $9.00   (12.5% over target)
```

Near `volume_min`, the rounding grid is coarse relative to the target risk,
so the realized risk can differ from the target by a large relative amount.
This is a structural constraint of small accounts, not a bug — size the
account, the risk-per-trade %, or the stop distance so that `raw_lots`
lands comfortably above a few multiples of `volume_step`.

### Two Ways to Hit a Target Risk

**Dynamic lot, fixed stop** (standard fixed-fractional, shown above):
solve for lots given a setup-derived stop distance, then round. Lot
rounding is the only source of error.

**Fixed lot, dynamic stop**: fix `lots` (e.g. always 0.10), then solve for
the stop distance:

```
stop_distance_points = risk_amount / (lots × point_value_per_lot)
```

Price/tick granularity is far finer than lot granularity, so this achieves
risk much closer to the target — but the stop no longer comes from the
setup itself, so it must still be checked against the setup's actual
invalidation level (too tight = noise stop-out; too wide = risking more
than the setup justifies even though the dollar amount matches).

---

## Combined Sizing Ladder

Calculate the three risk-based methods, take the minimum, then round to the
broker's lot grid:

```python
def sizing_ladder(
    account: float,
    risk_pct: float,
    entry: float,
    stop: float,
    atr: float,
    target_vol: float,
    win_rate: float,
    payoff_ratio: float,
    kelly_fraction: float,
    volume_min: float,
    volume_step: float,
    units_per_lot: float,
) -> dict:
    # 1. Fixed fractional
    ff_units = (account * risk_pct) / abs(entry - stop)

    # 2. Volatility-adjusted
    vol_units = (account * target_vol) / atr

    # 3. Kelly
    kelly_f = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
    kelly_risk = max(0, kelly_f * kelly_fraction)
    kelly_units = (account * kelly_risk) / abs(entry - stop)

    sizes = {
        "fixed_fractional": ff_units,
        "volatility_adjusted": vol_units,
        "kelly": kelly_units,
    }
    binding = min(sizes, key=sizes.get)
    raw_units = sizes[binding]

    # 4. Round to the broker's lot grid
    raw_lots = raw_units / units_per_lot
    step_count = round(raw_lots / volume_step)
    lots = max(volume_min, step_count * volume_step)

    return {
        "sizes": sizes,
        "binding": binding,
        "raw_units": raw_units,
        "lots": lots,
    }
```
