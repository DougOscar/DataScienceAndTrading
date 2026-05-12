# FFT Cycle Filter & Forward Projection

> **Type:** Statistical / Cycle-following
> **Markets:** Forex, B3, Crypto
> **Timeframes:** 1h, 4h, 1D
> **Direction:** Long & Short
> **Status:** Idea

---

## Overview

Financial prices contain a mixture of genuine cyclical components — driven by auction rhythms, institutional rebalancing periods, options expiry cycles, and macroeconomic seasonality — and high-frequency noise from tick-level randomness and microstructure. The Fast Fourier Transform (FFT) decomposes a price series into its constituent frequency components, making it possible to separate persistent cycles from transient noise.

This strategy uses a rolling FFT window to:
1. **Identify dominant cycles** — frequency components with the highest spectral power in a configurable period range.
2. **Reconstruct a filtered signal** — IFFT of only the dominant components, stripping away noise.
3. **Project forward** — since each kept harmonic is a sinusoid with a known frequency, amplitude, and phase, it can be extrapolated exactly one bar beyond the window edge. The sum of projected harmonics predicts the expected direction of the next bar's filtered signal.
4. **Trade the prediction** — enter long when the projection is above the current filtered value (expecting a rise), short when it is below (expecting a fall).

The hypothesis is that medium-term market cycles — typically between 8 and 64 bars — have sufficient autocorrelation to be exploitable for the duration of a single half-cycle. The ATR stop and a phase-reversal exit ensure the trade is closed if the cycle hypothesis is invalidated.

---

## Indicators

### Indicator 1 — Detrended Log-Price Window

- **Input:** Bar close prices
- **Formula:**
  ```
  For a rolling window of fft_window bars ending at bar t
  (indices i = 0, 1, …, N−1 where N = fft_window):

  log_close[i] = log(close[t − N + 1 + i])

  # Remove linear trend via OLS to avoid trend dominating spectral power:
  β₁ = Σᵢ(i − ī)(log_close[i] − lc̄) / Σᵢ(i − ī)²   (slope)
  β₀ = lc̄ − β₁ × ī                                    (intercept)

  trend[i]      = β₀ + β₁ × i
  x_detrended[i] = log_close[i] − trend[i]
  ```
  All spectral analysis is performed on `x_detrended`. The linear trend component is restored after reconstruction to express the filtered signal in original log-price units.
- **Lookback:** `fft_window` bars
- **Parameters:**
  - `fft_window` (int, default `128`) — must be a power of 2 for maximum FFT efficiency; common values: 64, 128, 256

### Indicator 2 — Hann-Windowed FFT

- **Input:** `x_detrended[i]` for i = 0..N−1
- **Formula:**
  ```
  # Step 1: Apply Hann window to reduce spectral leakage at window boundaries.
  # Without windowing, the hard edge of the rolling window introduces spurious
  # high-frequency components (Gibbs phenomenon).
  hann[i] = 0.5 × (1 − cos(2π × i / (N − 1)))

  x_windowed[i] = x_detrended[i] × hann[i]

  # Step 2: FFT
  X[f] = Σᵢ x_windowed[i] × exp(−2πj × f × i / N)    for f = 0, 1, …, N−1

  # Step 3: Power spectrum (one-sided, f = 0..N/2)
  Power[f] = |X[f]|²
  ```
  The FFT produces N complex coefficients. Coefficients for f = 1..(N/2 − 1) each represent a sinusoidal cycle with period `N/f` bars. The DC component (f=0) and Nyquist (f=N/2) are excluded from trading logic.
- **Lookback:** `fft_window` bars
- **Note:** `numpy.fft.rfft` computes the one-sided FFT directly and is the recommended implementation.

### Indicator 3 — Band-Pass Frequency Selection

- **Input:** `Power[f]`, FFT coefficients `X[f]`
- **Formula:**
  ```
  # Convert period bounds to frequency index bounds:
  f_min = floor(N / max_cycle_bars)      (slowest cycle kept)
  f_max = ceil(N / min_cycle_bars)       (fastest cycle kept)

  # Keep only frequency indices in [f_min, f_max]:
  X_filtered[f] = X[f]   if f_min ≤ f ≤ f_max
  X_filtered[f] = 0      otherwise

  # Additionally: within the band, optionally keep only the top n_harmonics
  # by spectral power (set n_harmonics = 0 to keep all in-band components):
  if n_harmonics > 0:
      in_band_power = {f: Power[f] for f in [f_min..f_max]}
      top_freqs = top n_harmonics keys of in_band_power by value
      X_filtered[f] = 0 for f in [f_min..f_max] if f not in top_freqs
  ```
- **Parameters:**
  - `min_cycle_bars` (int, default `8`) — shortest cycle period to keep (in bars); cycles shorter than this are treated as noise
  - `max_cycle_bars` (int, default `64`) — longest cycle period to keep (in bars); cycles longer than this are treated as trend
  - `n_harmonics` (int, default `0`) — if > 0, keep only the strongest N frequency components within the band; 0 = keep all in-band

### Indicator 4 — Filtered Signal Reconstruction

- **Input:** `X_filtered[f]`, detrend parameters
- **Formula:**
  ```
  # Inverse FFT of filtered coefficients:
  x_reconstructed[i] = real(IFFT(X_filtered)[i]) / hann[i]    (Hann compensation)

  # Note: Hann compensation near i=0 and i=N−1 is numerically unstable
  # (hann ≈ 0 at edges). Use only interior values i = N/4 .. 3N/4
  # for slope and projection calculations. The edge values are discarded.

  # Restore linear trend to get filtered log-price:
  x_filtered_log[i] = x_reconstructed[i] + trend[i]
  ```
  The reconstructed signal `x_filtered_log[N−1]` is the noise-filtered log-price estimate at the current bar.

### Indicator 5 — Forward Projection (One Bar Ahead)

- **Input:** Kept frequency components `X_filtered[f]` for in-band f
- **Formula:**
  ```
  # Each kept harmonic f represents a sinusoid with:
  #   Amplitude:  A[f] = |X_filtered[f]| × (2 / N)    (one-sided normalization)
  #   Phase:      φ[f] = angle(X_filtered[f])
  #   Frequency:  cycles_per_bar = f / N

  # Value of harmonic f at bar index i:
  #   h[f, i] = A[f] × cos(2π × f × i / N + φ[f])

  # Value of each harmonic at i = N (one bar BEYOND the window):
  #   h_projected[f] = A[f] × cos(2π × f × N / N + φ[f])
  #                  = A[f] × cos(2π × f + φ[f])

  # Total projected filtered signal one bar ahead:
  x_projected = Σ_f h_projected[f]   +   trend[N]
  #                                       (trend at N = β₀ + β₁ × N,
  #                                        extrapolating linear trend one step)
  ```
  `x_projected` is the expected log-price one bar ahead, based purely on the harmonic structure identified in the current window. The difference `Δ = x_projected − x_filtered_log[N−1]` gives the predicted direction and magnitude of the next bar's move.

### Indicator 6 — Projection Delta

- **Formula:**
  ```
  Δ(t) = x_projected(t) − x_filtered_log[N−1](t)
  ```
  - `Δ(t) > 0` → harmonic model predicts price will rise next bar
  - `Δ(t) < 0` → harmonic model predicts price will fall next bar

### Indicator 7 — Average True Range (ATR)

- **Input:** High, Low, Close
- **Formula:** Simple rolling mean of True Range
- **Lookback:** `atr_period + 1` bars
- **Parameters:**
  - `atr_period` (int, default `14`)

---

## Entry Signal

### Long Entry

All conditions at bar close `t`:

1. `Δ(t) > delta_min_atr × ATR(t)` — projection is positive and exceeds a minimum ATR-scaled threshold (avoids marginal signals near zero)
2. `Δ(t−1) ≤ 0` — projection was non-positive on the prior bar (fresh directional shift in the harmonic model; avoids entering late into an already-rising projected cycle)
3. `ATR(t)` is not NaN
4. FFT has been computed at least once (requires `fft_window` bars of history)

**Execution:**
- **Price:** bar close of bar `t`
- **Bar:** signal bar `t`

### Short Entry

Mirror of Long Entry:

1. `Δ(t) < −delta_min_atr × ATR(t)`
2. `Δ(t−1) ≥ 0`
3. `ATR(t)` is not NaN

**Execution:** bar close of bar `t`

### Entry Filters

| Filter | Default | Description |
|--------|---------|-------------|
| Delta magnitude gate | Always active | `|Δ(t)| > delta_min_atr × ATR(t)` — projection must exceed noise floor |
| Fresh shift gate | Always active | `Δ` must have just changed sign (prevents late-cycle entry) |
| Dominant cycle power filter | Optional | Require that the total spectral power of kept components exceeds `min_power_fraction` of total signal power (ensures the retained cycles genuinely dominate the signal) |
| Session filter | Off | B3: 09:00–18:00 BRT; do not enter in the last 30 min |
| Warm-up guard | Always active | Requires `fft_window + atr_period` bars minimum |

---

## Exit Signal

### Primary Exits — Price-Based

Stops are fixed at entry.

| Exit Type | Long | Short | Exit Price |
|-----------|------|-------|-----------|
| Stop Loss | `bar_low ≤ entry − sl_atr_mult × ATR_entry` | `bar_high ≥ entry + sl_atr_mult × ATR_entry` | SL level |
| Take Profit | `bar_high ≥ entry + tp_atr_mult × ATR_entry` | `bar_low ≤ entry − tp_atr_mult × ATR_entry` | TP level |

### Secondary Exits — Signal-Based

| Exit Type | Condition | Exit Price |
|-----------|-----------|-----------|
| Projection reversal | `Δ(t)` changes sign against the trade direction (model now predicts opposite movement) | Bar close |
| Phase cycle completion | The dominant-cycle half-period has elapsed since entry. If the dominant cycle has period `T_dominant` bars, exit after `T_dominant / 2` bars regardless of signal. | Bar close of the exit bar |
| Signal reversal | Opposite entry signal fires | Bar close |
| Session-end forced close | Last in-session bar | Bar close |
| End of data | Dataset ends | Last close |

### Exit Priority

1. Stop Loss
2. Take Profit
3. Projection reversal (if `use_projection_exit = True`)
4. Phase cycle completion (if `use_cycle_exit = True`)
5. Signal reversal
6. Session-end forced close

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions per asset | **1** | |
| Stop type | **Fixed** | ATR at entry bar |
| Stop loss | `entry ± sl_atr_mult × ATR_entry` | Default: 2.0 × ATR |
| Take profit | `entry ± tp_atr_mult × ATR_entry` | Default: 3.0 × ATR |
| Trailing stop | **No** | The projection reversal exit approximates a soft trailing mechanism |
| Default position sizing | **1 unit** | vol_scaled for cross-asset comparison |

### Markets

| Group | Assets | Notes |
|-------|--------|-------|
| Forex | EURUSD, EURCAD, GBPCHF | 4h and 1D; longer cycles are more stable on these TFs |
| B3 | WDO, WIN | 1h and 4h; session filter required |
| Crypto | BTCUSDT, ETHUSDT | 4h and 1D; crypto exhibits multi-day cycles correlated with derivatives expiry |

### Time Restrictions

| Rule | Forex | B3 | Crypto |
|------|-------|----|--------|
| Session filter | None (24/5) | 09:00–18:00 BRT | None (24/7) |
| Days of week | None tested | None tested | None tested |

---

## Parameters

| Parameter | Default | WFO Range | Description |
|-----------|---------|-----------|-------------|
| `fft_window` | `128` | `[64, 128, 256]` | Rolling FFT window in bars (power of 2 recommended) |
| `min_cycle_bars` | `8` | `[4, 6, 8, 10]` | Shortest cycle period to retain (high-pass cutoff, in bars) |
| `max_cycle_bars` | `64` | `[32, 48, 64, 96]` | Longest cycle period to retain (low-pass cutoff, in bars) |
| `n_harmonics` | `0` | `[0, 3, 5]` | Top-N in-band components to keep; 0 = all in-band |
| `delta_min_atr` | `0.1` | `[0.05, 0.10, 0.20]` | Minimum `\|Δ\|` in ATR units to accept an entry signal |
| `atr_period` | `14` | fixed | ATR period |
| `sl_atr_mult` | `2.0` | `[1.5, 2.0, 2.5]` | Stop loss ATR multiple |
| `tp_atr_mult` | `3.0` | `[2.0, 3.0, 4.0, 5.0]` | Take profit ATR multiple |
| `use_projection_exit` | `True` | — | Exit when Δ reverses sign against the trade |
| `use_cycle_exit` | `False` | — | Force close after T_dominant/2 bars |
| `min_power_fraction` | `0.0` | `[0.0, 0.3, 0.5]` | Min fraction of total power required in kept bands to allow entry |

**Constraint:** `min_cycle_bars < max_cycle_bars < fft_window / 2`

The upper bound on `max_cycle_bars` is `fft_window / 2` because a cycle with period `P` requires at least 2P bars to be detectable (Nyquist). In practice, reliable detection requires at least 3–4 full cycles within the window, so `max_cycle_bars ≤ fft_window / 3` is a tighter practical constraint.

### WFO-Optimized Values

_(Fill in after running walk-forward optimization)_

| Group | TF | Asset | fft_window | min_cycle | max_cycle | n_harmonics | sl_atr_mult | tp_atr_mult |
|-------|----|-------|------------|-----------|-----------|-------------|-------------|-------------|
| | | | | | | | | |

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
**Regime sensitivity:** Expected to perform best during periods where medium-term cyclicality is persistent and regular. Performance likely degrades during non-cyclic trend regimes and structural breaks. The `min_power_fraction` filter is the primary mechanism for detecting when the cycle hypothesis is weak.

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]]

---

## Known Weaknesses & Improvement Ideas

- **Stationarity assumption:** FFT implicitly assumes stationarity within the window. Financial prices are non-stationary; the detrending step removes first-order non-stationarity (linear trend), but residual non-stationarity (volatility clustering, structural breaks) will distort the spectrum. Short windows exacerbate this; long windows are slow to adapt.

- **Spectral leakage:** Even with a Hann window, significant energy from outside the target band can leak into the kept frequencies. This is particularly problematic when a strong low-frequency trend is present. Validate the detrending step carefully.

- **Edge instability:** The Hann window tapers signal magnitude near the window boundaries (i = 0, N−1). The current bar is at the window edge (i = N−1, where `hann ≈ 0`). Hann compensation (dividing by `hann[N−1]`) is numerically unstable and should be replaced by using only interior window values for signal analysis, accepting a small delay.

- **Cycle non-persistence:** A cycle identified in bars [t−128..t] may not persist into bar t+1. The one-bar projection is a minimal extrapolation, but cycles can abruptly change phase or disappear. The `min_power_fraction` threshold and strict delta-sign-change entry filter partially address this.

- **Dominant cycle detection:** When `n_harmonics = 0` (all in-band components kept), the reconstructed signal may be noisy from many weak cycles. Setting `n_harmonics = 3–5` to keep only the most powerful cycles produces a cleaner signal with fewer, higher-conviction trades.

- **Ohlers' MESA alternative:** John Ehlers' Maximum Entropy Spectral Analysis (MESA) is a parametric alternative to FFT that adaptively tracks the dominant cycle period in real time with much shorter windows (10–30 bars vs 64–256 for FFT). MESA may be more suitable for intraday data and should be considered as a comparison baseline.

---

## Implementation Notes

The FFT pipeline requires careful attention to avoid look-ahead bias:

1. At each bar `t`, the window `[t − N + 1 .. t]` must use only closed bars.
2. The forward projection at step 5 uses only the coefficients derived from bars up to and including `t` — it is a model prediction, not a look into future data.
3. Retraining is implicit (rolling window): the FFT is recomputed every bar, so no explicit retraining protocol is needed. However, computational cost is O(N log N) per bar; for large windows on minute data, batching or sub-sampling the FFT recomputation may be necessary.

**Recommended Python implementation:**
```
numpy.fft.rfft / numpy.fft.irfft  — one-sided real FFT
```

---

## Implementation

**Notebook:** `technical_analysis/14_fft_cycle_filter.ipynb`
**Source module:** `source/strategy.py` — `FFTCycleFilterStrategy`
**Parameters class:** `FFTCycleFilterParams`
**Dependencies:** `numpy` (FFT routines are part of core NumPy; no additional dependencies required)

### Implementation Notes

- Rolling FFT projection is computed in `source.strategy._rolling_fft_delta`
  using `numpy.fft.rfft`. Per-bar cost is O(N log N); cumulative cost across
  a multi-year forex 4h series is a few seconds.
- The one-bar-ahead projection uses the analytical fact that
  `cos(2π·f·N/N + φ) = cos(φ)` for integer `f`, so the projection at `i=N`
  is a simple weighted sum of the real parts of the kept FFT coefficients.
- The in-window reconstruction at `i = N-1` is **Hann-distorted** (no `hann`
  compensation, since `hann ≈ 0` at the edge makes division unstable). The
  *delta* between projection and last value is the strategy's signal —
  both terms carry the same Hann scaling so their difference is meaningful.
- `Δ` is a log-price difference; ATR is a price difference. The strategy
  converts ATR to log-units (`ATR / close`) before applying
  `delta_min_atr × ATR_log` as the noise floor.
- `min_cycle_bars < max_cycle_bars < fft_window / 2` (Nyquist) is enforced
  inside `generate_signals` (invalid combos emit zero signals).
- `use_projection_exit` and `use_cycle_exit` are exposed but **disabled** —
  need a custom `Backtester` exit hook.
- For B3, baseline params include `session_start=9`, `session_end=18`.
- Crypto group skipped — no `data/crypto/`.

---

## References

1. Ehlers, J.F. (2002). *Cybernetic Analysis for Stocks and Futures: Cutting-Edge DSP Technology to Improve Your Trading*. Wiley. Primary practitioner reference for applying digital signal processing (DSP) and FFT to market data.
2. Ehlers, J.F. (2013). *Cycle Analytics for Traders: Advanced Technical Trading Concepts*. Wiley. Extends FFT methods to adaptive cycle detection (MESA, dominant cycle measurement).
3. Ehlers, J.F. (1992). MESA and Trading Market Cycles. *Technical Analysis of Stocks & Commodities*, 10(4). Original MESA presentation; motivates the cycle-following approach.
4. Harris, F.J. (1978). On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform. *Proceedings of the IEEE*, 66(1), 51–83. Definitive reference on windowing functions (Hann, Hamming, etc.) and spectral leakage.
5. Oppenheim, A.V., & Schafer, R.W. (2009). *Discrete-Time Signal Processing* (3rd ed.). Prentice Hall. Standard DSP textbook — Chapters 8–9 on DFT and FFT.
6. Granger, C.W.J., & Morgenstern, O. (1963). Spectral Analysis of New York Stock Market Prices. *Kyklos*, 16(1), 1–27. Early empirical evidence for cyclical structure in financial time series using spectral analysis.
7. Bloomfield, P. (2000). *Fourier Analysis of Time Series: An Introduction* (2nd ed.). Wiley. Chapter 7: Application of spectral methods to economic and financial time series.
