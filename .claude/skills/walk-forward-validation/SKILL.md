---
name: walk-forward-validation
description: Walk-forward validation framework for trading strategies and ML models with time-series-aware splits, overfit detection, and regime-aware validation
---

# Walk-Forward Validation

Walk-forward validation framework for trading strategies and ML models. Standard cross-validation (k-fold, random splits) fails catastrophically for financial time series because it introduces lookahead bias and ignores autocorrelation. This skill covers proper time-series validation techniques including rolling and expanding windows, purged cross-validation, combinatorial purged cross-validation (CPCV), and overfit detection metrics.

## Why Standard Cross-Validation Fails

Standard k-fold CV assumes data points are independent and identically distributed (IID). Financial time series violate both assumptions:

1. **Lookahead bias** — Random splits let the model train on future data and predict past data, artificially inflating performance.
2. **Autocorrelation** — Adjacent observations are correlated. A random split that puts Monday in test and Tuesday in train leaks information.
3. **Regime dependence** — Markets shift between regimes. A model trained on a bull market and tested on a bull market tells you nothing about bear market performance.
4. **Label overlap** — If labels are computed over windows (e.g., 24h forward return), adjacent train/test samples share label computation periods, leaking information.

## Walk-Forward Framework

### Rolling Window (Fixed Train Size)

The train window has a fixed size and slides forward in time. Preferred when you believe older data is less relevant (e.g. after a broker/spread regime change or a shift in swap conditions).

```
Window 1: [===TRAIN===][=TEST=]
Window 2:    [===TRAIN===][=TEST=]
Window 3:       [===TRAIN===][=TEST=]
```

**Parameters:**
- `train_size`: Number of bars/days in the training window
- `test_size`: Number of bars/days in the test window
- `step_size`: How far to advance between folds (often equals `test_size`)

### Expanding Window (Growing Train)

The train window starts at the beginning and expands forward. This uses all available historical data, which helps when data is scarce.

```
Window 1: [==TRAIN==][=TEST=]
Window 2: [====TRAIN====][=TEST=]
Window 3: [======TRAIN======][=TEST=]
```

**Parameters:**
- `min_train_size`: Minimum training samples before first fold
- `test_size`: Fixed test window size
- `step_size`: How far to advance between folds

### Choosing Between Them

| Factor | Rolling | Expanding |
|---|---|---|
| Data recency | Prioritizes recent data | Uses all history |
| Regime changes | Better adapts to new regimes | May dilute recent regime |
| Sample size | Fixed, may be small | Grows over time |
| FX/metals swing (H1/H4/D1) | Fine given years of clean MT5 history | Also reasonable; history is long and continuous |
| B3 intraday (WIN/WDO) | Preferred — microstructure/liquidity drifts across contract rolls | Risks mixing pre/post liquidity regimes |

## Purging and Embargo

### Purging

Remove training samples whose labels overlap with the test set's time range. If a label is computed as the 24h forward return starting at time `t`, any training sample where `t + 24h` extends into the test period must be purged.

```python
def purge_train_indices(
    train_idx: list[int],
    test_start: int,
    label_horizon: int,
    timestamps: list[int],
) -> list[int]:
    """Remove train samples whose label windows overlap test period."""
    test_start_time = timestamps[test_start]
    return [
        i for i in train_idx
        if timestamps[i] + label_horizon < test_start_time
    ]
```

### Embargo

Add a buffer gap between the end of training and start of testing to account for serial correlation that purging alone does not eliminate.

```
[===TRAIN===][--EMBARGO--][=TEST=]
```

Typical embargo sizes:
- **B3 intraday (1–5min bars, WIN/WDO)**: 12–60 bars (session-bound — cannot carry past the day's close)
- **H1 (FX/metals)**: 6–24 bars (6–24 hours; skip embargo bars that fall in the weekend gap)
- **H4 (FX/metals)**: 6–12 bars (1–2 days)
- **D1 (FX/metals)**: 2–5 bars (2–5 trading days)
- **Rule of thumb**: embargo >= the max holding period (in bars) of the strategy being validated, so no test-fold label window can share information with an adjacent training bar

## Combinatorial Purged Cross-Validation (CPCV)

CPCV (Lopez de Prado, 2018) generates all possible train/test combinations from `N` groups while maintaining temporal ordering. This produces far more test paths than standard walk-forward, enabling statistical tests for overfitting.

**Key properties:**
- Splits data into `N` contiguous groups
- For each combination of `k` test groups, the remaining `N-k` groups form the training set
- Applies purging and embargo at each train/test boundary
- Produces `C(N, k)` backtest paths (e.g., N=6, k=2 gives 15 paths)

See `references/methodology.md` for the full CPCV algorithm and formulas.

## Overfit Detection

### Deflated Sharpe Ratio (DSR)

The observed Sharpe ratio must be adjusted for:
- Number of strategies tested (multiple testing)
- Non-normality of returns (skewness, kurtosis)
- Length of the backtest

```python
import numpy as np
from scipy.stats import norm

def deflated_sharpe_ratio(
    observed_sr: float,
    num_trials: int,
    backtest_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute the probability that observed SR > 0 after deflation.

    Args:
        observed_sr: Annualized Sharpe ratio of the selected strategy.
        num_trials: Number of strategies tested (including discarded ones).
        backtest_length: Number of return observations.
        skewness: Skewness of returns.
        kurtosis: Kurtosis of returns (not excess; normal = 3.0).

    Returns:
        p-value (probability SR is genuinely > 0).
    """
    sr_std = np.sqrt(
        (1 - skewness * observed_sr + (kurtosis - 1) / 4 * observed_sr**2)
        / (backtest_length - 1)
    )
    # Expected max SR under null (Euler-Mascheroni approximation)
    euler_mascheroni = 0.5772156649
    expected_max_sr = norm.ppf(1 - 1 / num_trials) * (
        1 - euler_mascheroni
    ) + euler_mascheroni * norm.ppf(1 - 1 / (num_trials * np.e))
    dsr = norm.cdf((observed_sr - expected_max_sr) / sr_std)
    return dsr
```

A DSR below 0.95 suggests the observed performance is likely due to overfitting across the trials tested.

### Probability of Backtest Overfitting (PBO)

PBO uses CPCV to measure the fraction of backtest paths where the in-sample optimal strategy underperforms the median out-of-sample. A PBO above 0.50 indicates more-likely-than-not overfitting.

See `references/overfit_detection.md` for complete derivations and implementation details.

## FX / Metals / B3 Considerations

1. **Weekends and holidays close FX/metals markets.** MT5 bar timestamps (server time EET) simply skip the closed period — there is no bar to purge or embargo over the weekend gap, but a rolling/expanding window boundary that lands on Friday close vs. Sunday open is not a "gap" in the data, so don't over-embargo trying to cover it; size the embargo in *trading* bars, not calendar time.
2. **Server time / spread column**: MT5 OHLC is Bid-only with a `spread` (points) column. Point-in-time embargo/purge logic should key off the bar's server timestamp, consistently with how features/labels were computed.
3. **B3 futures (WIN/WDO) are session-bound, no overnight.** Each session starts flat — folds should be built from whole sessions so a test fold never starts mid-session with warm-up state from a different session's train fold.
4. **Swing (H1/H4/D1) history is long and continuous** for FX majors/crosses and metals (years of clean MT5 history), so data scarcity is rarely the binding constraint — window size is set by how fast the tradable relationship (spread regime, swap costs, broker conditions) is believed to drift.
5. **Crypto CFDs** (a minor part of the FBS book) trade 7 days/week; when validating those, embargoes and window boundaries don't need to account for the weekend gap the way FX/metals folds do.

## Practical Window Sizes — FX / Metals Swing and B3 Intraday

| Strategy Timeframe | Train Window | Test Window | Embargo |
|---|---|---|---|
| B3 intraday (WIN/WDO, 1-5min) | 20-40 sessions | 5-10 sessions | 1 session |
| H1 swing | 6-12 months (~1,000-2,500 bars) | 1-2 months | 6-24 hours |
| H4 swing | 12-24 months (~500-1,000 bars) | 2-3 months | 1-2 days |
| D1 swing | 2-4 years (~500-1,000 bars) | 3-6 months | 2-5 days |

## Quick Start

```python
from walk_forward import WalkForwardValidator, WalkForwardConfig

config = WalkForwardConfig(
    train_size=90,
    test_size=14,
    step_size=14,
    window_type="rolling",
    embargo_size=3,
    purge_horizon=1,
)

validator = WalkForwardValidator(config)
for fold in validator.split(price_data):
    model.fit(fold.train_X, fold.train_y)
    predictions = model.predict(fold.test_X)
    fold.record_performance(predictions, fold.test_y)

results = validator.aggregate_results()
print(f"OOS Sharpe: {results.oos_sharpe:.3f}")
print(f"Train/Test Sharpe ratio: {results.sharpe_ratio_ratio:.2f}")
```

## Files

### References
- `references/methodology.md` — Walk-forward theory, window types, purging, embargo, CPCV algorithm with formulas
- `references/overfit_detection.md` — Deflated Sharpe ratio, probability of backtest overfitting, multiple testing corrections
- `references/practical_guide.md` — Window size selection for FX/metals swing and B3 intraday, regime considerations, common validation mistakes

### Scripts
- `scripts/walk_forward.py` — Walk-forward validation engine with rolling and expanding windows; `--demo` mode with synthetic data
- `scripts/overfit_detector.py` — Deflated Sharpe ratio and PBO computation; `--demo` mode with synthetic backtest results
