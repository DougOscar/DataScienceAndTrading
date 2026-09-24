---
name: correlation-analysis
description: Cross-asset correlation analysis including rolling correlation, hierarchical clustering, tail dependence, and regime-dependent correlation
---

# Correlation Analysis

Cross-asset correlation analysis for diversification assessment, risk management, pairs trading signal generation, and portfolio construction.

## Why Correlation Matters

Correlation measures how assets move together. Across FX majors/crosses, metals, and B3 futures this is critical for:

- **Diversification**: holding correlated assets provides no diversification benefit — you are effectively holding one concentrated position
- **Risk management**: portfolio risk depends on the correlation structure, not just individual asset volatility
- **Pairs trading**: highly correlated assets that temporarily diverge create mean-reversion opportunities
- **Portfolio construction**: optimal allocation requires accurate correlation estimates
- **Crash protection**: understanding tail dependence reveals whether assets crash together

## Correlation Methods

### Pearson Correlation

Linear correlation assuming normality. Most common but least robust under fat tails.

```python
import pandas as pd
import numpy as np

# Always compute on returns, never on prices
returns_a = prices_a.pct_change().dropna()
returns_b = prices_b.pct_change().dropna()

pearson_corr = returns_a.corr(returns_b)  # default is Pearson
```

- **Range**: -1 (perfect inverse) to +1 (perfect co-movement)
- **Assumes**: linear relationship, normally distributed returns, no outliers
- **Limitation**: returns are heavy-tailed, especially during risk-off/crisis periods and for crypto CFDs — Pearson underestimates extreme co-movement

### Spearman Rank Correlation

Converts values to ranks, then computes Pearson on ranks. Captures monotonic (not just linear) relationships.

```python
spearman_corr = returns_a.corr(returns_b, method='spearman')
```

- More robust to outliers and non-linear relationships
- Better when return distributions are heavy-tailed (common around news/events, and for crypto CFDs)
- Slightly lower power than Pearson when normality holds

### Kendall Tau Correlation

Counts concordant vs discordant pairs. Most robust to outliers.

```python
kendall_corr = returns_a.corr(returns_b, method='kendall')
```

- Most robust to outliers of the three methods
- Computationally slower on large datasets
- Best for small samples or heavily skewed data

## Rolling Correlation

Static correlation hides regime changes. Rolling correlation reveals how relationships evolve.

### Window-Based Rolling Correlation

```python
# Rolling Pearson correlation
rolling_corr = returns_a.rolling(window=60).corr(returns_b)

# Multiple windows for different time horizons
windows = {
    'short': 20,    # ~1 month of trading days
    'medium': 60,   # ~3 months
    'long': 120,    # ~6 months
}
for label, w in windows.items():
    df[f'corr_{label}'] = returns_a.rolling(w).corr(returns_b)
```

### EWMA Correlation

Exponentially weighted — more responsive to recent changes.

```python
def ewma_correlation(x: pd.Series, y: pd.Series, span: int = 60) -> pd.Series:
    """Compute EWMA correlation between two return series."""
    cov_xy = x.mul(y).ewm(span=span).mean() - x.ewm(span=span).mean() * y.ewm(span=span).mean()
    std_x = x.ewm(span=span).std()
    std_y = y.ewm(span=span).std()
    return cov_xy / (std_x * std_y)
```

### Typical Windows

| Window | Days | Use Case |
|--------|------|----------|
| Short  | 20   | Tactical trading, pairs entry/exit |
| Medium | 60   | Strategy allocation, regime detection |
| Long   | 120  | Portfolio construction, strategic allocation |

## Correlation Matrix Analysis

### Computing the Full Matrix

```python
# Build return matrix for multiple instruments
returns = pd.DataFrame({
    'EURUSD': eurusd_returns,
    'GBPUSD': gbpusd_returns,
    'XAUUSD': xauusd_returns,
    'USDJPY': usdjpy_returns,
})

# Correlation matrix (Pearson)
corr_matrix = returns.corr()

# Spearman (more robust under heavy tails)
spearman_matrix = returns.corr(method='spearman')
```

### Eigenvalue Decomposition

Decompose the correlation matrix to identify driving factors.

```python
eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix.values)

# Sort descending
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# First eigenvalue = market factor (explains most variance)
# Subsequent eigenvalues = sector/style factors
market_factor_pct = eigenvalues[0] / eigenvalues.sum() * 100
```

- **First eigenvector**: the market factor — when this dominates (>60% variance), everything moves together
- **Subsequent eigenvectors**: sector or style factors
- **Small eigenvalues**: noise / idiosyncratic risk

### Minimum Variance Portfolio

```python
from numpy.linalg import inv

cov_matrix = returns.cov()
ones = np.ones(len(cov_matrix))
inv_cov = inv(cov_matrix.values)

# Minimum variance weights
weights = inv_cov @ ones / (ones @ inv_cov @ ones)
```

## Hierarchical Clustering

Group assets by correlation similarity to identify natural clusters.

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Convert correlation to distance
dist_matrix = np.sqrt(2 * (1 - corr_matrix.values))
np.fill_diagonal(dist_matrix, 0)

# Hierarchical clustering
condensed = squareform(dist_matrix)
linkage_matrix = linkage(condensed, method='ward')

# Cut at threshold to get clusters
clusters = fcluster(linkage_matrix, t=1.0, criterion='distance')
```

**Applications**:
- **Sector detection**: assets in the same cluster behave similarly
- **Diversification**: select one asset per cluster for maximum diversification
- **Risk allocation**: allocate risk budget across clusters, not individual assets

## Tail Dependence

Normal correlation understates co-movement during crashes. Tail dependence measures how often assets experience extreme returns simultaneously.

### Lower Tail Dependence

```python
def tail_dependence(x: pd.Series, y: pd.Series, quantile: float = 0.05) -> float:
    """Estimate lower tail dependence coefficient.

    Measures P(Y < q | X < q) for quantile q.
    Higher values mean assets crash together more often.
    """
    threshold_x = x.quantile(quantile)
    threshold_y = y.quantile(quantile)
    joint_extreme = ((x < threshold_x) & (y < threshold_y)).sum()
    marginal_extreme = (x < threshold_x).sum()
    return joint_extreme / marginal_extreme if marginal_extreme > 0 else 0.0
```

### Tail Behavior vs. Normal Correlation

Tail dependence typically exceeds normal-times correlation — measure both on
your own instruments rather than assuming a fixed gap:
- During risk-off events, correlations across risk assets (and across FX
  crosses sharing a funding/carry currency) tend to spike toward 1.0
- This means diversification benefits can disappear exactly when needed most
- Crypto CFDs, where held, typically show this effect more sharply than
  FX/metals

## Regime-Dependent Correlation

Correlation is not constant — it changes with market regime. The qualitative
pattern below is common across asset classes, but the numeric levels are
illustrative only — measure actual regime-conditional correlation on your
own instruments rather than assuming these ranges.

| Regime | Typical Pattern | Implication |
|--------|-----------------|-------------|
| Trending | Moderate | Some diversification works |
| Range-bound | Lower | Often the best diversification environment |
| Risk-off / crisis | Elevated, can approach 1.0 | Diversification can fail when needed most |
| Recovery | Declining from crisis highs | Gradual return toward normal levels |

### Detecting Correlation Regime Shifts

```python
def correlation_zscore(rolling_corr: pd.Series, lookback: int = 252) -> pd.Series:
    """Z-score of rolling correlation vs its own history."""
    mean = rolling_corr.rolling(lookback).mean()
    std = rolling_corr.rolling(lookback).std()
    return (rolling_corr - mean) / std

# Flag regime shift when z-score exceeds threshold
zscore = correlation_zscore(rolling_corr_60d)
regime_shift = zscore.abs() > 2.0
```

## Correlation Patterns on Our Instruments

Don't assume correlation levels from asset-class folklore — measure them
directly on the FX majors/crosses, metals, and B3 futures actually traded,
using the rolling/EWMA methods above, and re-measure periodically since
these relationships drift with the macro regime (rate differentials,
risk-on/off cycles, commodity cycles). A few structural patterns worth
checking for, rather than assuming:

- **USD-cross co-movement**: pairs quoting against USD (EURUSD, GBPUSD,
  AUDUSD, ...) often share a broad-USD factor — expect elevated correlation
  that can invert sign depending on which currency is being quoted as base
  vs. quote.
- **Metals vs. USD-crosses**: gold/silver often (not always) show negative
  correlation to a broad USD move; measure it, don't assume a fixed level.
- **Carry/risk-currency clustering**: currencies sensitive to risk sentiment
  (e.g. AUD, NZD) can cluster together and de-correlate from safe-haven
  currencies (e.g. JPY, CHF) during risk-off moves.
- **B3 futures vs. FX/metals**: WIN (index) and WDO (USD/BRL) futures have
  their own drivers (local rates, Brazil risk) — don't assume they inherit
  G10 FX correlation structure.
- **Crypto CFDs**, where held, are a minor part of the book and tend to
  correlate more with each other and with broad risk sentiment than with
  FX/metals specifically.

## Integration with Other Skills

- **regime-detection**: correlation regime shifts are an input to regime classification
- **position-sizing**: correlation-adjusted sizing prevents correlated concentration

## Files

### References
- `references/methodology.md` — Correlation formulas, statistical tests, estimation methods
- `references/portfolio_applications.md` — Diversification metrics, pairs trading, risk decomposition

### Scripts
- `scripts/correlation_matrix.py` — Multi-asset correlation matrix, clustering, diversification metrics
- `scripts/rolling_correlation.py` — Rolling correlation, regime detection, tail dependence analysis
