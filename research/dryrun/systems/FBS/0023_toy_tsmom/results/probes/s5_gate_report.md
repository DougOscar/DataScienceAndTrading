# Validation — fbs-0023-a1

Gates: DESIGN §4.2 v1.2 (development data, after costs). Sharpe annualised from daily %-equity returns (periods/year = 260).

| Gate | Value | Threshold | Status | Interpretation |
|---|---|---|---|---|
| Deflated Sharpe probability | 0.000197 | >= 0.95 | **FAIL** | DSR 0.00: 0% probability the true Sharpe exceeds the 1.07 expected from the best of N = 840 raw null trials (840 this study + 0 from related earlier studies) over 2352 days. |
| CSCV OOS loss probability | 0.789 | < 0.1 | **FAIL** | The in-sample-best config lost money out of sample in 79% of 12870 CSCV splits (median OOS Sharpe -0.30); PBO 0.53 (diagnostic). |
| OOS Sharpe (CPCV median) | -0.683 | >= 1.0 | **FAIL** | Median OOS Sharpe -0.68 across 9 CPCV paths (IQR -0.78–-0.51); 0% of paths positive; in-sample -0.12. |
| Walk-forward procedure OOS | 0.07 / recent -0.07 | >= 0.5 and recent ⅓ > 0 | **FAIL** | WFO OOS Sharpe 0.07 over 1826 days (2018-05-02 → 2025-05-14); -0.07 over the most recent third (2023-01-10 → 2025-05-14); the procedure does not hold up out of sample. |
| Cost-stress Sharpe | -0.917 | > 0.5 | **FAIL** | Worst Sharpe -0.92 at 1.5× spread + 1 stress unit of slippage (1 pip / tick / median spread by asset class) on all market and stop fills (slippage 10 points per fill for EURUSD = 0.31× the median dev spread of 32 points), bar-extreme stops, swap ×0.5, 1, 1.5 (-0.92, -0.92, -0.92); base -0.12. |
| Parameter plateau | 0 | >= 0.6 | **FAIL** | Weakest parameter axis: lookback keeps 0/4 judge-run perturbations (±½, ±1 of each param's pre-registered plateau scale; r = 0.2, ledger study_created row; joint axis: 8 points) at ≥ 50% of the peak Sharpe -0.12; score = min over 4 axes (pooled 0%); 6 point(s) outside the search bounds evaluated; selected at search-space edge (stop_mult); 21 evaluations in 4.2 s; sharp, fragile optimum. |
| Positive years | 0.6 | >= 60% | **PASS** | 60% of 10 years positive; losing years [2016, 2018, 2019, 2025]. |
| Max single-year PnL share | inf | <= 40% | **FAIL** | Total PnL ≤ 0: no share defined; PnL is concentrated. |
| Trade count vs MinTRL | 670 trades / need inf | ≥ MinTRL (95%) | **FAIL** | 670 trades vs per-trade MinTRL inf; 2352 days vs daily MinTRL inf at claimed Sharpe -0.68 (95%). |
| Mechanism check | — | ablation matches hypothesis | **MANUAL** | Hypothesis-specific component ablation; to be judged against the card. |

## Verdict: **FAIL**

Not passed: Deflated Sharpe probability (FAIL), CSCV OOS loss probability (FAIL), OOS Sharpe (CPCV median) (FAIL), Walk-forward procedure OOS (FAIL), Cost-stress Sharpe (FAIL), Parameter plateau (FAIL), Max single-year PnL share (FAIL), Trade count vs MinTRL (FAIL), Mechanism check (MANUAL).

## Trials and deflation

- **N used = raw 840 trials** = 840 in this study + 0 from related earlier studies (ledger: no earlier related studies)
- Hurdle SR0 = √(1/(T−1)) · E[max of N] = 1.07 annualised (T = 2352 days)
- Selected Sharpe -0.12 annualised; skew 1.16, kurtosis 12.28
- DSR = 0.000; PSR(SR*=0) = 0.362
- Diagnostics only — effective N of this study: eigen 7.1 · cluster (ρ ≥ 0.5) 12.0 · Li–Ji 111.0; cross-trial Sharpe variance 0.000199 (per-period); v1.1-style DSR (N_eff + prior, V_cross) 0.069; raw N with V_cross 0.006

## Plateau perturbations (judge-run, not trials)

Radius 0.2 (ledger study_created row); peak (re-evaluated) Sharpe -0.12; 21 evaluations in 4.2 s.

Score = min over parameter axes of the per-axis pass share; weakest axis lookback ({'lookback': 0.0, 'stop_mult': 0.0, 'hold': 0.0, 'joint': 0.0}). Selected at search-space edge: stop_mult.

| Param | Offset | Value | Outside search bounds | Sharpe | Status | Keeps ≥ 50% of peak |
|---|---|---|---|---|---|---|
| lookback | x0.8 | 64 |  | -0.07 | ok | no |
| lookback | x0.9 | 72 |  | 0.09 | ok | no |
| lookback | x1.1 | 88 |  | -0.05 | ok | no |
| lookback | x1.2 | 96 |  | -0.64 | ok | no |
| stop_mult | x0.8 | 0.8 | yes | -0.23 | ok | no |
| stop_mult | x0.9 | 0.9 | yes | -0.24 | ok | no |
| stop_mult | x1.1 | 1.1 |  | -0.17 | ok | no |
| stop_mult | x1.2 | 1.2 |  | -0.14 | ok | no |
| hold | x0.8 | 14 |  | -0.02 | ok | no |
| hold | x0.9 | 16 |  | -0.05 | ok | no |
| hold | x1.1 | 20 |  | -0.14 | ok | no |
| hold | x1.2 | 22 |  | -0.16 | ok | no |
| joint | all -r | {'lookback': 64, 'stop_mult': 0.8, 'hold': 14} | yes | 0.04 | ok | no |
| joint | all -r/2 | {'lookback': 72, 'stop_mult': 0.9, 'hold': 16} | yes | 0.27 | ok | no |
| joint | all +r/2 | {'lookback': 88, 'stop_mult': 1.1, 'hold': 20} |  | 0.03 | ok | no |
| joint | all +r | {'lookback': 96, 'stop_mult': 1.2, 'hold': 22} |  | -0.46 | ok | no |
| joint | alt+ r/2 | {'lookback': 88, 'stop_mult': 0.9, 'hold': 20} | yes | 0.02 | ok | no |
| joint | alt+ r | {'lookback': 96, 'stop_mult': 0.8, 'hold': 22} | yes | -0.39 | ok | no |
| joint | alt- r/2 | {'lookback': 72, 'stop_mult': 1.1, 'hold': 16} |  | 0.43 | ok | no |
| joint | alt- r | {'lookback': 64, 'stop_mult': 1.2, 'hold': 14} |  | 0.17 | ok | no |

## Pre-registered holdout pass band (DESIGN §4.4 v1.2)

Horizon: 262 trading days, 2025-05-15 00:00:00 → 2026-05-15 10:36:00 (source: manifest; newer-than-locked data included: no). The exam uses the locked year plus all newer data available at the unlock; if more data is exported first, the band is rebuilt (gates.rebuild_holdout_band) and re-registered before unlocking.

Source: wfo_oos + stationary bootstrap (mean block 4.6 d), 2000 samples, seed 583009915. Joint band: common tail level α = 0.046 (one-sided limits at α, trade range at α/2 each side); 90% of the joint draws pass all four (target 90%).

| Criterion | Pass if |
|---|---|
| Holdout Sharpe (annualised) | ≥ -1.75 (median 0.06) |
| Mean monthly return at leverage k = 0.172 | ≥ -0.27% |
| Max drawdown (unlevered) | ≤ 20.72% |
| Trade count | within [56, 117] (source: wfo_oos.n_trades) |

Power: a zero-edge holdout (the source series de-meaned, same volatility and trade rate) passes this band with probability 89% (1000 draws).
**Not decisive at this horizon:** P(pass | zero edge) > 30%. If every criterion passes, the exam is recorded NOT_DECISIVE: the system stays `holdout_pending` and is re-examined on the full, longer holdout once more data is exported (DESIGN §4.4). A FAIL still kills it.

## Diagnostics (not gates)

- pbo_detail: pbo=0.5312, slope=-0.7413, r2=0.1419, prob_oos_loss=0.7887, median_logit=-0.1549, n_combos=12870, n_strategies=840
- cpcv_path_sharpe: median=-0.6826, p25=-0.7835, p75=-0.5139, min=-0.9444, n_paths=9
- wfo_oos: sharpe_all=0.06736, sharpe_recent=-0.07403, n_days=1826, recent_from=2023-01-10, recent_to=2025-05-14
- cost_stress_by_swap_mult: 0.5=-0.9174, 1=-0.9174, 1.5=-0.9174
- cost_stress_pip_points: pip_points=10, spec_source=load_instrument(EURUSD), slippage_points=10, added_slippage_points=10, median_spread_points=32, slippage_x_median_spread=0.3125
- cost_stress_version: fbs-v2-uncalibrated+spread_mult1.5+slippage10+stop_fill=bar_extreme
- plateau_optimizer: score=0, neighbourhood=knn=None, min_centre_quantile=0.5, min_neighbours=1, neighbourhood=grid, peak_fraction=0.5, radius=1
- plateau: kind=judge-run, plateau_score=0, score_rule=min over parameter axes, pooled_share=0, weakest_axis=lookback, n_points=20, n_evaluations=21, peak_sharpe=-0.1178, peak_sharpe_study=-0.1178, radius=0.2, radius_source=ledger study_created row, runtime_s=4.222, pass_share_by_param=lookback=0, stop_mult=0, hold=0, joint=0, n_outside_search_bounds=6, selected_at_search_space_edge=[stop_mult], unordered_not_judged=[]
- min_trl_days: inf
- return_at_10pct_dd_budget: leverage=0.1096, mean_monthly=-0.0001493, band_p5_p50_p95=[-0.0007876, -0.0001505, 0.0004754], label=unprofitable
