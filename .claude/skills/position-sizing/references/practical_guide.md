# Position Sizing — Practical Guide

Actionable sizing guidelines by account size, instrument type, and situation. Covers common mistakes and adjustment rules.

---

## Sizing by Account Size

### Micro Accounts (<$1K)

- **Fixed lot sizing**: often forced to trade at or near `volume_min`
  (commonly 0.01 lots) — the lot grid, not the risk formula, frequently
  binds; expect meaningful risk-realisation error (see SKILL.md)
- **Focus**: Learning, not profit. Treat as tuition.
- **Max concurrent positions**: 2–3
- **Total exposure**: keep leverage/margin usage conservative even though the broker allows more
- **Key rule**: never deposit more to "average down"

### Small Accounts ($1K–$10K)

- **Risk per trade**: 1–2% of account
- **Max concurrent positions**: 5
- **Max single position (margin)**: 10% of account
- **Key rule**: check that `raw_lots` from the sizing formula lands comfortably above a few multiples of `volume_step`, not right at `volume_min`

### Medium Accounts ($10K–$100K)

- **Risk per trade**: 0.5–1% of account
- **Max concurrent positions**: 7–10
- **Max single position (margin)**: 8% of account
- **Key rule**: lot-rounding error becomes small relative to account size; the risk formula, not the lot grid, is now the binding constraint

### Large Accounts (>$100K)

- **Risk per trade**: 0.25–0.5% of account
- **Max concurrent positions**: 10–15
- **Max single position (margin)**: 5% of account
- **Key rule**: for B3 intraday, execution/slippage against available liquidity at entry becomes the practical constraint, not the lot grid

---

## Sizing by Instrument Type

### FX Majors (EURUSD, GBPUSD, USDJPY, ...)

- **Max position size (margin)**: up to 10% of portfolio
- **Sizing method**: fixed fractional or volatility-adjusted
- **Typical ATR daily vol**: well below 1% of price for a quiet major — measure on your own data rather than assuming
- **Lot constraint**: rarely binding above small accounts; deep liquidity
- **Stop loss distance**: set from structure (swing level, ATR multiple), not a fixed %

### FX Crosses / Minors and Metals (AUDNZD, XAUUSD, XAGUSD, ...)

- **Max position size (margin)**: up to 5–8% of portfolio
- **Sizing method**: fixed fractional with lot-rounding check
- **Typical ATR daily vol**: higher and more variable than majors — metals especially; measure per instrument
- **Lot constraint**: check for smaller accounts, where `volume_min` may force a coarser size than the risk formula wants
- **Stop loss distance**: from structure/ATR, not a fixed %

### B3 Futures (WIN, WDO) — Intraday

- **Max position size (margin)**: up to 5% of portfolio per session
- **Sizing method**: fixed fractional, sized in whole/fractional contracts per the exchange's minimum increment
- **Session-bound**: no overnight carry — size and risk reset each session
- **Execution constraint**: liquidity available at entry (auction depth, time of day) can matter more than a lot-grid rounding error
- **Stop loss**: derived from the session's structure (opening range, prior close), not a fixed %

### Crypto CFDs (minor, secondary part of the book)

- **Max total allocation**: small — this is a minor part of the FBS book, not a core strategy
- **Sizing method**: fixed fractional, same lot-rounding mechanics as FX/metals
- **Typical ATR daily vol**: materially higher than FX/metals — measure and don't reuse FX-scale assumptions
- **Trades 7 days/week**: unlike FX/metals, no weekend gap to account for in sizing/stop placement

---

## When to Adjust Sizing

### Reduce Size

| Trigger | Action | Rationale |
|---------|--------|-----------|
| 3 consecutive losses | Reduce to 50% normal size | Possible regime change |
| Drawdown > 10% from peak | Reduce to minimum size | Capital preservation mode |
| New/untested strategy | Start at 25% normal size | Earn the right to full size |
| Market volatility spike (VIX-like) | Reduce by vol ratio | Same dollar risk needs less exposure |
| Unusual correlation across positions | Cut weakest positions | Correlated risk compounds |

### Increase Size (Cautiously)

| Trigger | Action | Rationale |
|---------|--------|-----------|
| 50+ trade track record with edge | Scale from 25% to 100% over time | Statistical confidence |
| Win streak > 5 | Allow up to 1.5x normal size | Strategy may be in sync |
| Low volatility regime | Vol-adjusted increase is mechanical | Same risk, larger notional |
| Account growth milestone | Recalculate base size upward | Compound growth |

### Never Increase Size When

- Trying to recover from losses ("revenge trading")
- A single trade seems like a "sure thing" (no such thing)
- You haven't adjusted stops to match the larger size
- Your recent win streak is < 10 trades (not statistically significant)

---

## Position Sizing Mistakes

### 1. Sizing Based on Conviction

**Wrong**: "I'm really confident in this trade, so I'll 5x my normal size."

**Right**: Let the math decide. Conviction is emotional, not quantitative. If you have a measured edge that justifies larger size, the Kelly formula will tell you.

### 2. Not Accounting for Spread, Commission, and Slippage

**Wrong**: Calculating stop distance as `entry - stop` only.

**Right**: Include the actual MT5 `spread` (points × point value), commission, and expected slippage in your risk calculation — spread/commission are usually small relative to a normal swing stop, but can dominate a tight B3 intraday stop; check rather than assume.

### 3. Ignoring Correlation Between Positions

**Wrong**: "I'm risking 2% on each of 10 positions = 20% max risk."

**Right**: If 8 of those positions are USD-crosses that share a broad-USD factor, your effective risk in a single USD-driven move is much closer to a single concentrated bet than 8 independent ones. Account for correlation in portfolio limits (see `correlation-analysis`).

### 4. Increasing Size to Make Back Losses

**Wrong**: After a 10% drawdown, doubling size to "get back to even faster."

**Right**: Reduce size during drawdowns. You need to earn back losses with smaller, consistent gains. The math: a 10% loss needs 11.1% gain to recover; a 50% loss needs 100%.

### 5. Using Position Size as Stop Loss

**Wrong**: "I'll just buy a small amount so if it goes to zero it's fine."

**Right**: Always use an explicit stop loss. "Small size, no stop" leads to holding a losing position that ties up margin and account risk indefinitely.

### 6. Not Adjusting for Timeframe

**Wrong**: Using the same 2% risk for a 5-minute scalp and a multi-week swing trade.

**Right**: Shorter timeframes need tighter stops relative to volatility, which means either smaller position sizes or accepting more noise. Scale risk per trade with expected hold time.

### 7. Forgetting Portfolio-Level Limits

**Wrong**: Each trade is individually sized, but no check on total exposure.

**Right**: Before each new position, verify:
- Total portfolio exposure is within limits
- Correlated exposure is within limits
- Total risk-on (sum of all position risks) is acceptable

---

## Quick Decision Tree

```
Want to enter a trade?
│
├─ Do you have a measured edge (50+ trades)?
│  ├─ Yes → Calculate Kelly size (use 0.25x)
│  └─ No  → Use fixed fractional (1% risk)
│
├─ Calculate fixed fractional size
│
├─ Is the instrument's ATR unusually elevated vs. its own history?
│  ├─ Yes → Also calculate vol-adjusted size
│  └─ No  → Skip vol adjustment
│
├─ Round the binding size to the broker's MT5 lot grid
│  (volume_min / volume_step) — check the resulting risk-realisation
│  error, especially on small accounts near volume_min
│
├─ Check portfolio limits (margin-based, not notional):
│  ├─ Single position margin < 10% of account? ✓
│  ├─ Correlated exposure < 25%? ✓
│  └─ Total margin usage within comfortable leverage headroom? ✓
│
└─ Execute at the lot-rounded size
```

---

## Quick Reference (Fixed-Fractional, FX Example)

For a $15,000 account trading EURUSD-style pairs (illustrative point value):

| Risk Level | Risk/Trade | Typical Stop | Approx. Lots (0.01 step) |
|-----------|-----------|---------------|---------------------------|
| Conservative (0.5%) | $75 | 30 pips | ~0.25 lots |
| Standard (1%) | $150 | 30 pips | ~0.50 lots |
| Moderate (2%) | $300 | 30 pips | ~1.00 lots |
| Aggressive (3%) | $450 | 30 pips | ~1.50 lots |

Recompute for your actual instrument's point value and stop distance —
these numbers assume a $10/point/lot instrument and a fixed 30-pip stop,
purely to illustrate scale.
