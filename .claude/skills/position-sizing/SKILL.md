---
name: position-sizing
description: Trade sizing methods including fixed fractional, volatility-adjusted, Kelly criterion, and MT5 lot-constrained sizing for FX/metals and B3 futures
---

# Position Sizing

Position sizing is the single most important risk management decision in trading. Your entry signal determines direction; your position size determines survival. A mediocre strategy with proper sizing will outperform a great strategy with reckless sizing over any meaningful time horizon.

**Core principle**: Size determines survival, not entries. Two traders with the same signals but different sizing will have wildly different outcomes. The one who sizes conservatively survives drawdowns and compounds capital; the one who oversizes blows up.

## Methods Covered

| Method | Best For | Key Input |
|--------|----------|-----------|
| Fixed Fractional | General trading, most recommended | Account risk % |
| Volatility-Adjusted | Volatile markets, multi-asset | ATR or realized vol |
| Kelly Criterion | Quantified edge with track record | Win rate + payoff ratio |
| MT5 Lot-Constrained | Any MT5-executed order | volume_min/volume_step, tick value |
| Anti-Martingale | Trend-following strategies | Recent P&L streak |

---

## 1. Fixed Fractional Sizing

The most recommended method for most traders. Risk a fixed percentage of your account on each trade.

### Formula

```
risk_amount = account_value * risk_percentage
price_risk_per_unit = entry_price - stop_loss_price
position_size_units = risk_amount / price_risk_per_unit
position_value = position_size_units * entry_price
```

### Risk Tiers

| Tier | Risk Per Trade | Use Case |
|------|---------------|----------|
| Conservative | 0.5–1% | New strategies, drawdown recovery |
| Standard | 1–2% | Most traders, proven strategies |
| Aggressive | 3–5% | High-conviction setups with strong, measured edge |

### Example

```python
account = 10_000  # $10,000
risk_pct = 0.02    # 2%
entry = 1.0850     # EURUSD
stop_loss = 1.0820  # 30-pip stop

risk_amount = account * risk_pct              # $200
price_risk = entry - stop_loss                # 0.0030 (30 pips)
position_units = risk_amount / price_risk     # 66,667 base-currency units
# In MT5 terms this raw size still needs rounding to volume_step —
# see "MT5 Lot Constraints" below.
```

With this sizing, if the stop loss is hit, you lose exactly 2% of your account regardless of the instrument's price or volatility — before lot-rounding error (see below).

---

## 2. Volatility-Adjusted Sizing

Scale position size inversely with volatility. When volatility is high, take smaller positions; when low, take larger positions. This normalizes the dollar risk across different market conditions.

### Formula

```
adjusted_size = base_size * (target_vol / current_vol)
```

Where:
- `target_vol`: your desired daily portfolio volatility (e.g., 2%)
- `current_vol`: the instrument's current daily volatility (from ATR or realized vol)

### Using ATR

```python
atr_14 = 0.0045         # 14-period ATR (EURUSD, H4)
close_price = 1.0850
daily_vol_pct = atr_14 / close_price  # ~0.4%

target_daily_vol = account * 0.02      # $200 target daily move
position_size = target_daily_vol / atr_14  # units (round to lot step for MT5 execution)
```

This automatically reduces exposure in volatile markets and increases it in calm ones.

---

## 3. Kelly Criterion

The mathematically optimal fraction of capital to risk, maximizing long-term growth rate. Derived from maximizing expected logarithmic utility.

### Formula

```
f* = (p * b - q) / b
```

Where:
- `p` = win rate (probability of winning trade)
- `q` = 1 - p (probability of losing trade)
- `b` = average win / average loss (payoff ratio)
- `f*` = optimal fraction of capital to risk

Equivalent form: `f* = (p * (b + 1) - 1) / b`

### Critical Rule: NEVER Use Full Kelly

Full Kelly assumes perfect knowledge of your edge. In practice, edge estimates are noisy. Always use fractional Kelly:

| Fraction | Use Case | Notes |
|----------|----------|-------|
| 0.25x Kelly | Conservative, recommended default | Robust to edge estimation error |
| 0.50x Kelly | Moderate, for well-measured edges | Still significant drawdown risk |
| 1.0x Kelly | Never in practice | Theoretical maximum, catastrophic if edge is overestimated |

### Example

```python
win_rate = 0.55       # 55% win rate
avg_win = 2.0         # Average win is 2x the average loss
avg_loss = 1.0
payoff_ratio = avg_win / avg_loss  # b = 2.0

kelly = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
# kelly = (0.55 * 2.0 - 0.45) / 2.0 = 0.325 = 32.5%

quarter_kelly = kelly * 0.25  # 8.1% — use this
half_kelly = kelly * 0.50     # 16.25%
```

**If Kelly is negative, you have no edge. Do not trade.**

See `references/sizing_formulas.md` for the full mathematical derivation.

---

## 4. MT5 Lot Constraints

MT5 sizing is not continuous — the broker's symbol specification constrains
the lot size you can actually submit, and the mathematically "optimal" size
from any formula above must be rounded to what the broker will accept.

### Key Symbol Spec Fields

| Field | Meaning | Typical value |
|-------|---------|---------------|
| `volume_min` | Smallest tradable lot | often 0.01 lots |
| `volume_step` | Increment between allowed lot sizes | often 0.01 lots |
| `volume_max` | Largest single order size | broker/instrument-specific |
| Contract size | Units per 1.0 lot | e.g. 100,000 base-currency units for a standard FX lot — check the actual symbol spec per instrument/broker, it varies (metals and B3 futures use different conventions) |
| Tick/point value | $ P&L per point per lot | used to convert a price-based stop distance into $ risk |

### Formula

```
raw_lots = risk_amount / (stop_distance_points * point_value_per_lot)
lots = round_to_step(raw_lots, volume_step, volume_min)
```

### Risk-Realisation Error

Because lots must round to `volume_step`, the dollar risk actually taken
on a trade almost never equals the target `risk_amount` exactly:

```python
risk_amount = 200.0
point_value_per_lot = 10.0   # $ per point per lot, e.g. a standard FX lot
stop_distance_points = 30.0

raw_lots = risk_amount / (stop_distance_points * point_value_per_lot)  # 0.667
lots = round(raw_lots / 0.01) * 0.01                                   # 0.67 (volume_step=0.01)
actual_risk = lots * stop_distance_points * point_value_per_lot        # $201 (close)
```

The relative error grows as position size approaches `volume_min` — a
strategy sized to risk $5 on a micro account with `volume_min=0.01` might
find its *only* two choices are 0.00 (skip the trade) or 0.01 lots (risk
some other amount entirely), with no way to hit the target in between.
Track realized vs. target risk per trade; don't assume the formula's output
was actually achieved.

### Two Ways to Hit a Fixed-% Risk Target

1. **Dynamic lot size, price-based (fixed) stop**: fix the stop distance
   from the setup (e.g., beyond a swing level, N×ATR), compute the lot
   size that risks `risk_pct` of equity at that distance, then round to
   `volume_step`. This is the standard fixed-fractional approach above;
   lot rounding is the only source of risk-realisation error.
2. **Fixed lots, dynamic stop**: fix the lot size (e.g., always trade 0.10
   lots), then solve for the stop distance that produces the target dollar
   risk: `stop_distance = risk_amount / (lots * point_value_per_lot)`.
   This avoids lot-rounding error on the risk side (price/tick granularity
   is typically far finer than lot granularity), but the stop distance now
   floats with account size/target risk and must still make technical
   sense — don't let the risk formula place a stop inside the setup's
   natural noise, or beyond where the setup is actually invalidated.

Neither is strictly "correct" — pick based on whether the strategy's edge
depends on a specific, setup-derived stop location (use #1) or tolerates a
sizing-derived stop (use #2).

---

## 5. Anti-Martingale Sizing

Increase size after wins, decrease after losses. This is the opposite of the gambler's fallacy (Martingale). The logic: winning streaks may indicate your strategy is in sync with the market; losing streaks may indicate regime change.

### Implementation

```python
def anti_martingale_size(
    base_size: float,
    consecutive_wins: int,
    consecutive_losses: int,
    scale_factor: float = 0.25,
    max_multiplier: float = 2.0,
    min_multiplier: float = 0.5,
) -> float:
    if consecutive_losses > 0:
        multiplier = max(min_multiplier, 1.0 - consecutive_losses * scale_factor)
    elif consecutive_wins > 0:
        multiplier = min(max_multiplier, 1.0 + consecutive_wins * scale_factor)
    else:
        multiplier = 1.0
    return base_size * multiplier
```

Use conservatively. After 3+ consecutive losses, reducing size by 50% protects capital during drawdowns.

---

## Position Sizing Ladder

Combine all methods and take the most conservative result:

```
1. Calculate Kelly size          → theoretical max based on edge
2. Calculate fixed fractional    → risk-based size
3. Calculate volatility-adjusted → vol-normalized size
4. Round to MT5 lot constraints  → what the broker will actually accept
5. Final size = min(first three), then rounded per step 4
```

The binding constraint tells you what is limiting your size:
- **Kelly-bound**: your edge is small, size accordingly
- **Risk-bound**: standard risk management is the limit
- **Volatility-bound**: market is too volatile for larger size
- **Lot-bound**: `volume_min`/`volume_step` prevents hitting the target size exactly (see risk-realisation error above)

---

## Account-Level Limits

Individual position sizing is necessary but not sufficient. You also need portfolio-level constraints:

| Limit | Guideline | Rationale |
|-------|-----------|-----------|
| Max single position | 10% of portfolio | Diversification floor |
| Max correlated exposure | 25% of portfolio | Correlated assets move together |
| Max total exposure | 50–80% of portfolio | Cash reserve for opportunities/margin |
| Max positions | 5–10 concurrent | Attention and management bandwidth |

---

## Integration with Other Skills

| Skill | Integration |
|-------|-------------|
| `volatility-modeling` | Better vol estimates for volatility-adjusted sizing |
| `regime-detection` | Adjust sizing/risk tier by regime |
| `correlation-analysis` | Correlation-adjusted sizing to avoid concentrated exposure |

---

## Files

### References
- `references/sizing_formulas.md` — Mathematical derivations for all sizing methods with worked examples
- `references/practical_guide.md` — Sizing by account size, instrument type, and common mistakes

### Scripts
- `scripts/size_calculator.py` — Calculates position size using all methods, shows binding constraint
- `scripts/portfolio_sizer.py` — Portfolio risk dashboard with per-position risk and available budget

---

## Quick Reference

```python
# Minimal fixed fractional sizing — copy-paste starter
def calc_position_size(
    account: float, risk_pct: float, entry: float, stop: float
) -> float:
    """Return number of units to buy."""
    risk_amount = account * risk_pct
    price_risk = abs(entry - stop)
    if price_risk == 0:
        return 0.0
    return risk_amount / price_risk
```
