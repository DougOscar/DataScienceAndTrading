---
name: applied-math-quant
description: Applied-mathematics toolbox for quantitative finance — stochastic calculus & price-process models, time-series forecasting (ARIMA/GARCH/state-space/Kalman), regime-change models (HMM, Markov-switching, change-point detection), and portfolio optimization & position sizing (Markowitz, Black–Litterman, risk parity, HRP, CVaR, Kelly). Use when designing or documenting forecasting, regime, portfolio-construction, sizing, or synthetic-data-generation components, or when a task needs mathematically-grounded modeling rather than ad-hoc heuristics.
---

# Applied Mathematics for Quant Finance

Use the right model class for the structure in the data, state assumptions explicitly, and validate with the methods in `quant-trading-research` (purged CV, DSR, no OOS selection).

## Stochastic processes & calculus

Continuous-time price models, used mainly for **synthetic data generation** (robustness testing), derivative reasoning, and mean-reversion modeling.

- **Geometric Brownian Motion:** dS = μS dt + σS dW. Log-returns are iid normal — a deliberately *too-clean* null; if a strategy is profitable on pure GBM paths it's likely exploiting structure that isn't real edge.
- **Jump-diffusion (Merton):** adds a compound-Poisson jump term — fat tails and gaps.
- **Stochastic volatility (Heston):** variance follows its own mean-reverting CIR process — volatility clustering and smiles.
- **Ornstein–Uhlenbeck:** dX = θ(μ − X) dt + σ dW — the canonical mean-reverting process. Half-life of reversion = ln 2 / θ. Foundation for pairs/stat-arb: model the spread as OU, trade deviations in σ-units, size by reversion speed.
- Itô's lemma is the bridge from an SDE for S to the SDE for f(S); know it well enough to derive log-return dynamics and to simulate with Euler–Maruyama (mind discretization bias; use exact schemes for OU/GBM where available).

## Time-series forecasting

Always start with a stationarity check (ADF/KPSS) and ACF/PACF inspection. Difference or transform (log, fractional differencing to keep memory while gaining stationarity) before modeling.

- **ARIMA / SARIMA:** linear autocorrelation in the (differenced) mean. Often a weak baseline for returns (near-zero autocorrelation) but useful for volume, spreads, and other observables.
- **GARCH family (GARCH, EGARCH, GJR):** the workhorse for **volatility**, which *is* forecastable (clustering). Use for risk scaling, vol-targeting, and option-like reasoning. Fit by MLE; check standardized-residual ACF and Ljung–Box.
- **State-space + Kalman filter:** unobserved-component models (local level/trend), dynamic regressions with time-varying betas, online estimation. Naturally causal — no look-ahead. The Kalman filter is also a clean way to estimate hedge ratios that drift.
- **ML forecasters** (see `financial-ml`): tree ensembles on engineered features; sequence models (LSTM/GRU, Temporal Convolution, Transformers, N-BEATS, Temporal Fusion Transformer). These need far more data and far more discipline against leakage than they appear to.
- Evaluate forecasts with walk-forward; compare against a naive/random-walk baseline (for prices the random walk is brutally hard to beat). Use Diebold–Mariano to test whether one forecast is significantly better than another.

## Regime-change models

Markets switch between states (trending/ranging, low/high vol, risk-on/off). Modeling this explicitly beats a single global fit.

- **Hidden Markov Models (Gaussian-emission):** latent discrete state with Markov transitions; fit via Baum–Welch (EM), decode states with Viterbi. Use the **filtered** (causal) state probability for trading, never the smoothed one (uses the whole sample = look-ahead).
- **Markov-switching models (Hamilton):** regression/AR coefficients and variances switch with a latent Markov state — e.g., switching-mean or switching-volatility return models.
- **Change-point detection:** CUSUM, Bayesian Online Change-Point Detection (BOCPD), Pruned Exact Linear Time (PELT) for offline segmentation. Good for detecting structural breaks in the strategy's own performance, too.
- **Volatility regimes:** GARCH state, realized-vol clustering, or simple rolling-vol quantiles as a robust, low-parameter regime proxy.
- Discipline: regimes are only useful if the *current* regime is identifiable in real time with acceptable lag. Always report the filtered-state classification lag.

## Portfolio construction & position sizing

- **Mean-variance (Markowitz):** max wᵀμ − (λ/2) wᵀΣw. Powerful but notoriously unstable — tiny changes in estimated μ swing weights wildly. Never use raw sample μ.
- **Estimator fixes:** Ledoit–Wolf shrinkage of Σ; denoising Σ via random-matrix theory (Marčenko–Pastur); robust/Bayesian means.
- **Black–Litterman:** blend an equilibrium prior with explicit views and view-confidences — far more stable weights than raw MVO.
- **Risk parity / equal risk contribution:** weight so each asset contributes equal marginal risk; ignores μ entirely (a feature, given how bad μ estimates are).
- **Hierarchical Risk Parity (HRP, López de Prado):** cluster the correlation matrix, allocate recursively. No matrix inversion → robust to ill-conditioned Σ; usually better OOS than MVO.
- **CVaR / mean-CVaR optimization:** minimize expected shortfall (a coherent, tail-aware risk measure) via Rockafellar–Uryasev LP — preferable to variance when returns are skewed/fat-tailed.
- **Sizing — Kelly:** f* = μ/σ² (continuous) maximizes long-run log-growth but is brutal on drawdowns and hypersensitive to estimation error; **use fractional Kelly (¼–½)**. Combine with **volatility targeting** (scale exposure to hit a constant risk budget) and hard risk limits. Account for estimation uncertainty — bet smaller than the point estimate says.

## Numerical hygiene

Prefer log-returns for aggregation; use stable/online algorithms (Welford) for moments; keep covariance matrices PSD (clip eigenvalues / shrink); set and record RNG seeds; watch float vs decimal (money is decimal, statistics are float). Vectorize with NumPy; reserve loops for genuinely path-dependent simulation.
