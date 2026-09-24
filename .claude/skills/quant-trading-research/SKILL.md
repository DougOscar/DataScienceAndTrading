---
name: quant-trading-research
description: Methodology for designing, backtesting, and validating systematic trading strategies without fooling yourself. Use whenever working on backtesting, alpha research, walk-forward optimization, robustness testing, performance metrics, or any task that asks "does this strategy actually have an edge?". Encodes the cardinal sins of quant research and how to avoid them.
---

# Quantitative Trading Research

The default outcome of strategy research is **self-deception**. Almost every backtest that looks good is overfit, leaked, or survivorship-biased. Your job is to be the adversary of your own results. Assume an edge is fake until the statistics say otherwise after honest correction for how hard you looked.

## The cardinal sins (in priority order)

1. **Multiple testing / selection bias.** Trying N indicators × M thresholds × K horizons and keeping the best is guaranteed to surface false positives. A single Sharpe of 2.0 means nothing if it's the max of 5,000 trials. **Always track the number of trials** and deflate accordingly (see Deflated Sharpe Ratio, PBO, White's Reality Check / Hansen's SPA).
2. **Look-ahead leakage.** Any use of information not available at decision time. Common forms: computing a signal on bar *t*'s close and filling at that same close; normalizing features with full-sample statistics; labeling with future data that overlaps training; selecting parameters on the test set.
3. **Overfitting via optimization.** The more parameters you tune and the finer the grid, the more the "optimum" is fitting noise. In-sample Sharpe is an upward-biased estimator of out-of-sample Sharpe.
4. **Survivorship / selection bias in data.** Delisted instruments, point-in-time index membership, restated fundamentals.
5. **Ignoring costs and capacity.** Spread, commission, slippage, market impact, financing/swap. A gross edge of a few bps per trade usually dies after costs.
6. **Non-stationarity.** Markets regime-shift. A strategy fit on 2015–2019 has no guarantee in 2020+. Validate across regimes, not just across time.

## Backtesting correctly

- **Execution timing:** signals decided on bar *t* (using only data ≤ *t*) execute at bar *t+1*'s open (or with an explicit, defended latency model). Never fill at the close that generated the signal.
- **Costs:** model spread (half-spread per side), commission (per-lot or bps), and slippage (volatility- or volume-scaled, not a flat constant). For overnight positions add swap/financing.
- **Point-in-time data:** only expose `bar[i]` and earlier when processing bar `i`. Rolling statistics must be causal.
- **Position accounting:** track exposure, leverage, and margin; force-close at series end and mark the reason.
- **Reproducibility:** every run records a manifest — data hash, code version, config, RNG seed. A research result you can't reproduce is an anecdote.

## Labeling and forward-return analysis

- A raw "did price go up after N bars?" hit-rate is weak: it ignores the path (you may be stopped out before the horizon) and the **overlap problem** — signals close in time share overlapping forward windows, so observations are autocorrelated and naive standard errors are far too small.
- Prefer the **triple-barrier method** (López de Prado): label by which barrier is hit first — profit-take, stop-loss, or time horizon. Combine with **meta-labeling** (a secondary model decides whether to act on the primary signal and at what size).
- For overlapping samples, down-weight by **uniqueness** (average number of concurrent labels) or use non-overlapping windows; compute standard errors with HAC/Newey–West.

## Performance metrics — do them honestly

- **Sharpe:** annualize with √(periods/yr) *only if returns are iid*. Correct for autocorrelation (Lo, 2002). Report alongside skew and kurtosis — Sharpe assumes neither.
- **Probabilistic Sharpe Ratio (PSR):** probability the true Sharpe > a benchmark, given track-record length, skew, kurtosis.
- **Deflated Sharpe Ratio (DSR):** PSR adjusted for the number of trials and the variance of trial Sharpes — the single most important metric when you've searched a parameter space.
- Report **CIs (bootstrapped)** on headline metrics, not point estimates. Add downside/tail measures: Sortino, Calmar, Omega, Ulcer index, max drawdown + duration, VaR/CVaR (historical or Cornish–Fisher), and the **t-stat of mean return**.

## Walk-forward & cross-validation

- Optimize on **in-sample (IS)** only; evaluate on **out-of-sample (OOS)**. **Selecting the parameter set with the best OOS metric re-introduces leakage** — OOS then stops being out-of-sample. Select on IS (or on an IS-internal validation fold); use OOS purely for unbiased reporting.
- For ML/feature strategies, plain k-fold leaks because of serial correlation. Use **purged k-fold with an embargo** around each test fold, and **Combinatorial Purged Cross-Validation (CPCV)** to get a distribution of OOS paths.
- Quantify overfitting with **PBO (Probability of Backtest Overfitting)** via the combinatorially-symmetric cross-validation framework.

## Robustness testing — what each method can and cannot prove

- **Trade reshuffling / bootstrap of the realized trade list** estimates *sequence risk* (drawdown distribution, path dependence). It **cannot detect overfitting** — it reuses the same overfit trades, so an overfit strategy still looks "robust." Do not use `pct_profitable` from reshuffling as evidence the edge is real.
- To probe whether the edge is genuine, resample/regenerate the **input data or the selection process**: block/stationary bootstrap of returns (preserves autocorrelation), synthetic price paths (GBM/jump-diffusion/GARCH/regime-switching), and **re-run the entire optimization** on each resample to measure how often the discovered edge survives (this is what estimates PBO).
- **Monte Carlo permutation tests (MCPT):** shuffle the mapping between signals and forward returns to build a null distribution for your metric, then read off a p-value.
- **Sensitivity:** measure performance under parameter perturbation using **elasticity** (% change in objective per % change in parameter), not raw gradients. Prefer a broad performance *plateau* over a sharp peak.

## Red flags in a result

Sharpe > 3 on daily data; equity curve is a near-straight line; tiny number of trades carrying all the PnL; performance concentrated in one regime/year; many tuned parameters; metric collapses under next-bar execution or realistic costs; OOS was used for selection. Any one of these voids the result until explained.

## References

López de Prado, *Advances in Financial Machine Learning*; Bailey & López de Prado on DSR/PSR/PBO; Harvey, Liu & Zhu, "…and the Cross-Section of Expected Returns" (multiple testing); Lo, "The Statistics of Sharpe Ratios"; White's Reality Check; Hansen's SPA.
