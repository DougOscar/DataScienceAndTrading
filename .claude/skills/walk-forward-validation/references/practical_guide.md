# Walk-Forward Validation — Practical Guide for FX / Metals Swing and B3 Intraday

## Window Size Selection

### Principles

1. **Train window must span at least one full market cycle** — or else the model only learns one regime. For FX/metals swing (H1/H4/D1) a "cycle" (trend/range alternation, a rate-hike cycle, a risk-on/off regime) typically runs months, not weeks.
2. **Train window should not be so long that stale data dilutes signal** — broker conditions (spread, swap, execution) and cross-asset correlations do drift over years; weight or window accordingly rather than assuming multi-decade stationarity.
3. **Test window must be long enough for statistical significance** — a 1-day test window produces noisy estimates. Aim for at least 30 independent observations (bars or sessions, depending on timeframe).
4. **Step size determines compute cost** — smaller steps = more folds = better estimates but longer runtime.
5. **B3 futures (WIN/WDO) are session-bound with no overnight exposure** — build folds from whole sessions so no fold starts mid-session, and don't let a window imply carrying a position across a session boundary.

### Recommended Window Sizes

| Strategy Type | Train | Test | Step | Embargo | Rationale |
|---|---|---|---|---|---|
| B3 intraday (WIN/WDO, 1-5min bars) | 20-40 sessions | 5-10 sessions | 5 sessions | 1 session | Session-bound; liquidity/microstructure drifts across contract rolls |
| FX/metals H1 swing | 6-12 months | 1-2 months | 1 month | 6-24 hours | Needs to span both trending and ranging stretches |
| FX/metals H4 swing | 12-24 months | 2-3 months | 1 month | 1-2 days | Must span multiple regime transitions |
| FX/metals D1 swing | 2-4 years | 3-6 months | 1-2 months | 2-5 days | Longer horizon, needs full cycles across rate/macro regimes |

### FX/Metals vs. B3 Intraday

| Aspect | FX/Metals Swing (H1/H4/D1) | B3 Intraday (WIN/WDO) |
|---|---|---|
| Typical train window | Months to years (long, continuous MT5 history) | 20-40 sessions |
| Market hours | Closed weekends/holidays (server time EET); gaps at week open | Session-bound, no overnight |
| Regime change frequency | Weeks to quarters | Can shift within a session (news, auction imbalance) |
| Data source | MT5 Bid OHLC + `spread` (points) | B3 exchange data, contract rolls (WINFUT/WDOFUT) |
| Recommended approach | Rolling or expanding — both viable given long history | Rolling, aligned to whole sessions |

## Regime-Aware Validation

### Why Regimes Matter

A strategy that works in high-volatility trending markets may lose money in low-volatility mean-reverting markets. If your train and test windows both fall within the same regime, validation results are misleading.

### Simple Regime Detection for Splits

Use volatility and trend to classify market regime before splitting:

```python
import numpy as np

def classify_regime(
    returns: np.ndarray,
    lookback: int = 20,
    periods_per_year: float = 252.0,
    high_vol_threshold: float = 0.12,
) -> np.ndarray:
    """Classify each bar as trending-volatile, trending-quiet, etc.

    periods_per_year: ~260 for FX/metals daily bars (5-day week; 252 also
        common), 365 for crypto CFDs, bars_per_day * trading_days for intraday.
    high_vol_threshold: measure on your own instrument/timeframe rather than
        assuming a level — FX/metals daily annualized vol is typically far
        below crypto's.
    """
    n = len(returns)
    regimes = np.empty(n, dtype="U20")
    for i in range(lookback, n):
        window = returns[i - lookback : i]
        vol = np.std(window) * np.sqrt(periods_per_year)  # Annualized
        trend = np.sum(window)  # Cumulative return
        if vol > high_vol_threshold:
            regimes[i] = "trending-up" if trend > 0 else "trending-down"
        else:
            regimes[i] = "quiet-up" if trend > 0 else "quiet-down"
    regimes[:lookback] = "unknown"
    return regimes
```

### Regime-Aware Split Strategy

1. **Classify** each bar into a regime
2. **Verify** that each train window spans at least 2 different regimes
3. **Track** the regime composition of each test window
4. **Report** performance broken down by regime
5. **Flag** folds where train and test are the same regime

If most folds have matching train/test regimes, the validation is unreliable for out-of-regime performance.

## Common Validation Mistakes

### 1. Not Purging Labels

**Mistake:** Using 5-day forward returns as labels without purging the 5-day overlap between train and test.

**Result:** Information leakage inflates accuracy by 10–30%.

**Fix:** Always purge `label_horizon` bars from the end of training data.

### 2. Using Future Information in Features

**Mistake:** Normalizing features using the full dataset's mean/std before splitting.

**Result:** Test set statistics leak into training features.

**Fix:** Fit scalers/normalizers on training data only. Transform test data using training statistics.

```python
# WRONG
scaler.fit(all_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

# RIGHT
scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)
```

### 3. Optimizing Hyperparameters on Test Data

**Mistake:** Using the test fold to tune hyperparameters (learning rate, lookback periods, thresholds).

**Result:** Hyperparameters are fit to the test set, destroying its validity.

**Fix:** Use a three-way split: train / validation / test. Tune on validation, evaluate on test. Or use nested cross-validation.

### 4. Ignoring Transaction Costs

**Mistake:** Computing walk-forward returns without accounting for spreads, slippage, and fees.

**Result:** A strategy with 200 trades/day and 0.5 bps edge appears profitable, but 5 bps per round-trip cost makes it a loser.

**Fix:** Always include realistic transaction costs. For MT5 FX/metals, use the actual `spread` column (points × point value) plus commission and a slippage estimate, not a flat assumption; for B3 futures, exchange fees plus slippage on the auction/liquidity available at entry.

### 5. Too Few Folds

**Mistake:** Using 3 walk-forward folds and reporting the average.

**Result:** High variance in the estimate. One good or bad fold dominates.

**Fix:** Use at least 8–10 folds, or CPCV to generate 15+ paths.

### 6. Reporting In-Sample Metrics

**Mistake:** Reporting training set performance alongside (or instead of) test set performance.

**Result:** Misleading impression of strategy quality.

**Fix:** Only report out-of-sample metrics. Track the train/test ratio as an overfitting diagnostic.

### 7. Not Accounting for Multiple Testing

**Mistake:** Testing 50 parameter combinations, selecting the best, and reporting its backtest result.

**Result:** The best of 50 random strategies will look profitable by chance alone.

**Fix:** Use the Deflated Sharpe Ratio to adjust for the number of trials.

### 8. Contract Roll / Continuation Handling (B3 Futures)

**Mistake:** Backtesting WIN/WDO on a naively concatenated series across contract expiries without adjusting for the roll gap, or silently dropping the low-liquidity tail of an expiring contract.

**Result:** Phantom returns/losses at roll dates, or optimistic fills against unrealistic end-of-contract liquidity.

**Fix:** Use a proper continuous/back-adjusted series (or roll on volume/open-interest crossover), and exclude/flag the illiquid tail of each contract near expiry.

## Validation Checklist

Before trusting any backtest result, verify:

- [ ] Time series ordering is respected (no future data in training)
- [ ] Labels are purged at train/test boundaries
- [ ] Embargo period is applied after purging
- [ ] Features are normalized using training data only
- [ ] Transaction costs are included (realistic for the venue)
- [ ] At least 8 walk-forward folds or 15 CPCV paths
- [ ] Deflated Sharpe Ratio > 0.95 (accounting for all trials)
- [ ] PBO < 0.30 (if using strategy selection)
- [ ] Train/test Sharpe ratio < 2.0
- [ ] Results reported per-regime if possible
- [ ] Hyperparameters tuned on validation set, not test set
- [ ] B3 contract rolls handled with a proper continuous series (not a naive concatenation)

## Reporting Template

When presenting walk-forward results, include:

```
Walk-Forward Validation Report
==============================
Window type:        Rolling (90-day train, 14-day test)
Number of folds:    12
Embargo:            3 days
Purge horizon:      1 day
Date range:         2025-01-01 to 2025-12-31
Strategies tested:  25

Out-of-Sample Results:
  Mean Sharpe:        1.42
  Sharpe Std Dev:     0.38
  Mean Return:        +3.2% per fold
  Win Rate:           58.3%
  Max Drawdown:       -8.7%

Overfitting Metrics:
  Train/Test SR:      1.65
  Deflated SR:        0.87
  PBO:                0.22

Regime Breakdown:
  Trending-Up:    SR 2.1 (4 folds)
  Trending-Down:  SR 0.8 (3 folds)
  Quiet:          SR 1.1 (5 folds)
```
