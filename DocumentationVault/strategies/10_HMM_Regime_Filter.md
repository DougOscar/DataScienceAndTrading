# Hidden Markov Model Regime Filter

> **Type:** Machine Learning / Statistical
> **Markets:** Forex, B3, Crypto
> **Timeframes:** 1h, 4h, 1D
> **Direction:** Long & Short
> **Status:** Idea

---

## Overview

Financial markets exhibit distinct latent regimes — persistent states characterized by different return distributions, volatility levels, and trend behaviors. A Hidden Markov Model (HMM) treats the observed market features (returns, volatility, volume) as noisy emissions from a small number of unobservable hidden states. The Viterbi algorithm (or forward filtering) estimates the most probable current state given the observation history.

This strategy overlays an HMM regime filter onto a base trend-following signal (SMA crossover). Trades are only taken when the HMM assigns high probability to a "trending" or "directionally favorable" state, suppressing entries during high-uncertainty or ranging regimes. The model is retrained periodically on a rolling window to adapt to structural regime shifts.

The HMM framework is motivated by the observation that asset return distributions are not stationary — they exhibit volatility clustering, skewness shifts, and correlation breaks that correspond to identifiable market phases. Hamilton (1989) demonstrated that a 2-state HMM applied to GDP growth could identify recessions; the same machinery applies to financial return series.

---

## Indicators

### Indicator 1 — Observation Features (HMM Input)

Each bar `t` is represented by a 3-dimensional observation vector:

```
feature_1(t) = log_return(t)          = log(close(t) / close(t−1))
feature_2(t) = realized_vol(t)        = rolling std of log_returns over rv_window bars
feature_3(t) = normalized_volume(t)   = tick_vol(t) / SMA(tick_vol, vol_period)(t)
```

- **Parameters:**
  - `rv_window` (int, default `20`) — rolling window for realized volatility
  - `vol_period` (int, default `20`) — SMA period for volume normalization

### Indicator 2 — HMM State Probability

- **Model:** Gaussian HMM with `n_states` hidden states
  - Each state emits observations drawn from a multivariate Gaussian: `x ~ N(μ_k, Σ_k)`
  - Transition matrix `A` (n_states × n_states): `A[i,j] = P(state_j | state_i)`
  - Initial state distribution `π`
- **Inference:** Forward algorithm yields `P(state_k | observations_1..t)` for each bar `t`
- **Training:** Expectation-Maximization (Baum-Welch algorithm) on a rolling window of `hmm_train_window` bars; retrained every `hmm_retrain_period` bars
- **Parameters:**
  - `n_states` (int, default `3`) — number of hidden states (bull trend, bear trend, ranging)
  - `hmm_train_window` (int, default `504`) — training window in bars (~3 months on 1h)
  - `hmm_retrain_period` (int, default `21`) — bars between model retraining (~1 month on 1D)

### Indicator 3 — State Labeling

After fitting, states must be assigned semantic labels (trending-up, trending-down, ranging). The assignment is automatic and based on properties of the fitted Gaussians:

```
state with highest mean log-return    → "bull_trend"
state with lowest mean log-return     → "bear_trend"
remaining state(s)                    → "ranging"
```

This labeling is recomputed after each retraining.

### Indicator 4 — SMA Crossover Base Signal

- **Formula:** Identical to Strategy 01 (SMA_fast > SMA_slow crossover)
- **Parameters:**
  - `sma_fast` (int, default `20`)
  - `sma_slow` (int, default `50`)

### Indicator 5 — Average True Range (ATR)

- **Formula:** Simple rolling mean of True Range
- **Parameters:**
  - `atr_period` (int, default `14`)

---

## Entry Signal

### Long Entry

1. `SMA_fast(t) > SMA_slow(t)` AND `SMA_fast(t−1) ≤ SMA_slow(t−1)` — SMA golden cross
2. `P(state = "bull_trend" | obs_1..t) ≥ hmm_prob_threshold` — HMM assigns high probability to bullish state
3. `ATR(t)` is not NaN
4. HMM model has been trained at least once (requires `hmm_train_window` bars of history)

### Short Entry

Mirror of Long Entry:

1. `SMA_fast(t) < SMA_slow(t)` AND `SMA_fast(t−1) ≥ SMA_slow(t−1)` — SMA death cross
2. `P(state = "bear_trend" | obs_1..t) ≥ hmm_prob_threshold`
3. `ATR(t)` is not NaN

**Execution (both):** Bar close of bar `t`

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| HMM probability gate | Always active when model trained | Only enter if state probability ≥ `hmm_prob_threshold` (default 0.70) |
| Training warm-up | Always active | No entries until first HMM training completes |
| SMA warm-up | Always active | Requires `sma_slow` bars before SMA is valid |
| Session filter | Off | B3: 09:00–18:00 BRT |

---

## Exit Signal

### Primary Exits — Price-Based

| Exit Type | Long | Short | Exit Price |
|-----------|------|-------|-----------|
| Stop Loss | `bar_low ≤ entry − sl_atr_mult × ATR_entry` | `bar_high ≥ entry + sl_atr_mult × ATR_entry` | SL level |
| Take Profit | `bar_high ≥ entry + tp_atr_mult × ATR_entry` | `bar_low ≤ entry − tp_atr_mult × ATR_entry` | TP level |

### Secondary Exits — Signal-Based

| Exit Type | Condition | Exit Price |
|-----------|-----------|-----------|
| HMM regime exit | `P(entry_state | obs_1..t) < hmm_exit_threshold` — HMM is no longer confident in the entry state | Bar close |
| Signal reversal | Opposite SMA crossover fires | Bar close |
| Session-end | Last in-session bar | Bar close |
| End of data | Dataset ends with open position | Last close |

### Exit Priority

1. Stop Loss
2. Take Profit
3. HMM regime exit (if `use_hmm_exit = True`)
4. Signal reversal
5. Session-end

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset | **1** | |
| Stop type | **Fixed** | ATR at entry |
| Stop loss | `entry ± sl_atr_mult × ATR_entry` | Default: 2.0 × ATR |
| Take profit | `entry ± tp_atr_mult × ATR_entry` | Default: 3.5 × ATR |
| Trailing stop | **No** | |
| Default position sizing | **1 unit** | |

### Markets

| Group | Assets | Notes |
|-------|--------|-------|
| Forex | EURUSD, EURCAD, GBPCHF | 4h and 1D; HMM stability improves with longer TF |
| B3 | WDO, WIN | 1h minimum; shorter TF makes state estimation noisier |
| Crypto | BTCUSDT, ETHUSDT | 4h and 1D; crypto has well-defined bull/bear regimes |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| `n_states` | `3` | `[2, 3]` | Number of hidden states |
| `hmm_train_window` | `504` | `[252, 504, 1008]` | Training window in bars |
| `hmm_retrain_period` | `21` | `[7, 21, 63]` | Bars between model retrains |
| `hmm_prob_threshold` | `0.70` | `[0.60, 0.70, 0.80]` | Minimum state probability for entry |
| `hmm_exit_threshold` | `0.40` | `[0.30, 0.40, 0.50]` | State probability below which exit is triggered |
| `rv_window` | `20` | fixed | Realized volatility rolling window |
| `vol_period` | `20` | fixed | Tick volume normalization SMA period |
| `sma_fast` | `20` | `[10, 20, 30]` | Fast SMA period |
| `sma_slow` | `50` | `[40, 60, 100]` | Slow SMA period |
| `atr_period` | `14` | fixed | ATR period |
| `sl_atr_mult` | `2.0` | `[1.5, 2.0, 2.5]` | Stop loss ATR multiple |
| `tp_atr_mult` | `3.5` | `[2.5, 3.5, 5.0]` | Take profit ATR multiple |
| `use_hmm_exit` | `True` | — | Exit when HMM regime probability drops |

**Important:** The HMM must be re-trained using only data available at training time (no look-ahead). In backtesting, this requires an expanding-window or rolling-window online training protocol.

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | n_states | hmm_prob_threshold | sma_fast | sma_slow |
|-------|----|-------|----------|--------------------|----------|----------|
| | | | | | | |

---

## Performance Summary

_(Fill in after backtesting)_

| Metric | Baseline | WFO-Optimized | Notes |
|--------|----------|--------------|-------|
| Sharpe (daily) | | | |
| Profit Factor | | | |
| Win Rate | | | |
| Max Drawdown | | | |
| P(profitable) block bootstrap | | | |

**Best configuration:** TBD
**Regime sensitivity:** The strategy's raison d'être is regime adaptation. Validation should include comparing against the base SMA crossover (Strategy 01) without the HMM filter to measure added value.

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- HMM state labeling (assigning "bull/bear/ranging" to fitted states) is heuristic and may assign labels inconsistently after retraining, causing abrupt strategy behavior changes
- Gaussian emission assumptions may be violated by fat-tailed financial returns; a Student-t HMM or GMM-HMM would be more appropriate but is less commonly available in standard libraries
- Retraining on each bar is computationally expensive; in production, periodic retraining (weekly or monthly) is more practical
- The model is sensitive to the number of states: 2 states is more stable but less nuanced; 3 states risks one state being rarely occupied
- Look-ahead bias in backtesting is a major risk: the entire retraining and inference pipeline must be implemented with strict temporal isolation (no future data in training window)

---

## Implementation

**Notebook:** `machine_learning/10_hmm_regime_filter.ipynb`
**Source module:** `source/strategy.py` — `HMMRegimeFilterStrategy`
**Parameters class:** `HMMRegimeFilterParams`
**Dependencies:** `scikit-learn` (already in `requirements.txt`)

### Implementation Notes

- **Substitution: GaussianMixture instead of a full Gaussian HMM.**
  `hmmlearn` is not in `requirements.txt`, and the existing `Backtester`
  doesn't support periodic mid-backtest retraining. GMM provides
  `P(state | obs)` directly (no transition matrix) which is what the doc's
  regime filter consumes. The transition-matrix smoothing a true HMM would
  add is foregone in v1; the doc's "Known Weaknesses" section notes that
  even the doc-spec'd HMM is fragile, so the difference is small in practice.
- Model is fit **once per backtest call** on the first `hmm_train_window`
  bars. Under WFO each fold gets an independent fit on its in-sample slice.
- State labeling: ascending mean log_return → bear / neutral / bull.
- `use_hmm_exit` is exposed but **disabled** — regime-probability-driven
  close needs a custom `Backtester` exit hook.
- For B3, baseline params include `session_start=9`, `session_end=18`.
- Crypto group skipped (no `data/crypto/`).

---

## References

1. Hamilton, J.D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*, 57(2), 357–384. Foundational HMM application to economics.
2. Rabiner, L.R. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition. *Proceedings of the IEEE*, 77(2), 257–286. Baum-Welch and Viterbi algorithms.
3. Ang, A., & Bekaert, G. (2002). Regime Switches in Interest Rates. *Journal of Business & Economic Statistics*, 20(2), 163–182. Application to financial markets.
4. Nystrup, P., Hansen, B.W., Madsen, H., & Lindström, E. (2017). Dynamic Portfolio Optimization Across Hidden Market Regimes. *Quantitative Finance*, 17(1), 83–95. HMM-based regime detection applied to portfolio allocation.
5. Bulla, J., & Bulla, I. (2006). Stylized Facts of Financial Time Series and Hidden Semi-Markov Models. *Computational Statistics & Data Analysis*, 51(4), 2192–2209.
