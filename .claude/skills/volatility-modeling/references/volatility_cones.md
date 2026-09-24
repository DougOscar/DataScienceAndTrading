# Volatility Cones — Reference Guide

## Overview

A volatility cone is a visualization that shows the percentile distribution of
realized volatility across multiple lookback windows.  It answers the question:
"Is current volatility unusually high or low compared to history?"

The name "cone" comes from the shape — shorter lookback windows produce wider
percentile spreads, and the distribution narrows at longer windows due to mean
reversion.

---

## Construction Steps

### Step 1: Gather Data

Collect at least 1 year (ideally 2+) of daily OHLCV data.  More history
produces more reliable percentiles.  FX majors/crosses and metals typically
have years of clean MT5 history, so this is rarely the binding constraint;
a newly-listed B3 contract or a crypto CFD added to the broker's book may
have well under a year available.

### Step 2: Define Lookback Windows

Standard windows for FX/metals swing and B3 intraday:

| Window | Bars/Days | Use Case |
|---|---|---|
| Very short | 5 bars | B3 intraday / short swing |
| Short | 10 bars | Swing trading (H1/H4) |
| Medium | 20 bars | D1 swing / position sizing |
| Long | 60 bars | Portfolio management |
| Very long | 120 bars | Strategic allocation |

### Step 3: Compute Rolling Realized Volatility

For each window length `w`, compute rolling close-to-close or Parkinson vol:

```python
import pandas as pd
import numpy as np

def rolling_realized_vol(
    closes: pd.Series, window: int, periods_per_year: float = 260
) -> pd.Series:
    """Compute rolling annualized close-to-close volatility.

    periods_per_year: ~260 for FX/metals daily bars (5-day week; 252 also
        common), 365 for crypto CFDs, bars_per_day * trading_days for intraday.
    """
    log_ret = np.log(closes / closes.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(periods_per_year)
```

This produces a time series of realized vol estimates, one for each date.

### Step 4: Extract Percentiles

For each window, compute the percentile distribution of the rolling vol series:

```python
percentiles = [5, 25, 50, 75, 95]
windows = [5, 10, 20, 60, 120]

cone = {}
for w in windows:
    rv = rolling_realized_vol(closes, w).dropna()
    cone[w] = {p: np.percentile(rv, p) for p in percentiles}
```

### Step 5: Compute Current Volatility

For each window, the current realized vol is simply the most recent value:

```python
current = {}
for w in windows:
    rv = rolling_realized_vol(closes, w)
    current[w] = rv.iloc[-1]
```

### Step 6: Determine Percentile Rank

Where does current vol sit within the historical distribution?

```python
from scipy import stats

current_percentile = {}
for w in windows:
    rv = rolling_realized_vol(closes, w).dropna()
    current_percentile[w] = stats.percentileofscore(rv, rv.iloc[-1])
```

---

## Visualization

Plot percentile bands on y-axis vs window length on x-axis:

```
Annualized Vol (%)
    |
200 |  *                           ← 95th percentile
    | * *
150 |*   *      *
    |     *   * *
100 |      * *   *    *    *       ← 75th percentile
    |       *     *  * *  * *
 80 |              **   **   *     ← 50th (median)
    |                        *
 60 |                    *    *    ← 25th percentile
    |                         *
 40 |                          *   ← 5th percentile
    |________________________________
      5d   10d   20d  60d   120d
              Window Length
```

Overlay the current vol (marked with ●) at each window length.

---

## Interpretation

### Current Vol Above 75th Percentile
- Historically elevated volatility.
- Vol tends to mean-revert → expect compression ahead.
- Reduce position sizes.
- Widen stops to avoid getting shaken out.
- Consider vol-selling strategies if available.

### Current Vol Below 25th Percentile
- Historically compressed volatility.
- Breakout likely coming → expect vol expansion.
- Can take larger positions (risk per unit is lower).
- Tighten stops — when vol is low, small moves are significant.
- Watch for range breakouts.

### Current Vol Near 50th Percentile
- Normal conditions — no special adjustments needed.
- Use standard position sizes and stop distances.

### Cone Shape Analysis

**Steep cone (wide at short windows, narrow at long):**
- Normal mean-reverting vol structure.
- Short-term vol varies widely but long-term vol is stable.

**Flat cone (similar width across windows):**
- Vol regime has been persistent.
- Long-term vol structure may be shifting.

**Inverted at current (current vol higher at long windows):**
- Sustained vol regime — not just a short-term spike.
- May indicate structural market change.

---

## FX / Metals / B3 Considerations

### Weekend and Session Gaps
FX/metals close over the weekend (server time EET) — rolling windows in
*trading* bars already skip the closed period, so no special handling is
needed for standard close-to-close/Parkinson/Garman-Klass estimators; just
don't interpret a window boundary spanning a weekend as evidence of a gap
in coverage. B3 futures are session-bound (no overnight); build cones from
whole sessions and treat the session close-to-next-open gap explicitly if
using Yang-Zhang.

### Regime Changes
Rate-hike cycles, risk-on/off shifts, and (for B3) macro/election cycles
produce structurally different vol levels.  A cone built entirely within
one regime will show misleadingly low or high percentiles once the regime
shifts.

**Mitigation:** Build separate cones for different regimes, or use only
recent history (e.g., 6–12 months) to capture the current regime.

### Limited History for New/Thin Instruments
A newly-listed B3 contract, an exotic FX cross, or a crypto CFD added to
the broker's book may have well under a year of history.  Cones built from
short histories are unreliable.  Minimum recommended: 180 days of data.

### Spread and Liquidity Effects
MT5 Bid OHLC bars carry a `spread` (points) column; a spike in spread
(illiquid hours, news events) can inflate high-low-based estimators
(Parkinson, Garman-Klass) without a genuine increase in underlying vol.

**Mitigation:** Cross-check against close-to-close vol, and consider
excluding or flagging bars with abnormal spread.

### Multiple Timeframes
For B3 intraday, build cones from 1–5 minute bars using session-relative
windows (e.g., first N minutes, full session, N sessions) rather than
calendar-hour windows, since each session starts flat.

---

## Practical Decision Framework

```
current_pctile = percentile_rank(current_vol, historical_vol_series)

if current_pctile > 90:
    action = "Significantly reduce size, widen stops 2x"
elif current_pctile > 75:
    action = "Reduce size 30%, widen stops"
elif current_pctile > 25:
    action = "Normal sizing and stops"
elif current_pctile > 10:
    action = "Can increase size, tighten stops"
else:
    action = "Low vol — watch for breakout, keep tight stops"
```

---

## Combining with Vol Forecast

Volatility cones tell you where current vol sits historically.  Combine with
GARCH forecasts to get forward-looking information:

- **Cone says high + GARCH forecast declining:** Vol likely to compress.
- **Cone says low + GARCH forecast rising:** Breakout underway.
- **Cone says high + GARCH forecast still rising:** Crisis conditions.
- **Cone says low + GARCH forecast flat:** Extended low-vol regime.

See `scripts/estimate_volatility.py` for a complete implementation that
computes cones and overlays current vol with percentile ranks.
