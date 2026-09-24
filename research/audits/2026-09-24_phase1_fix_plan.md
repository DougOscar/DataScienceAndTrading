# Phase 1 fix plan: red-team B1–B4, M1–M5 + calibration F1–F4 (2026-09-24)

Inputs: `2026-09-24_phase1_redteam.md` (B/M/m) and `../calibration/2026-09-24_phase1_calibration.md` (F, R).
User decision 2026-09-24: fix B1–B4, M1–M5, F1–F4; **adopt R1 and R2**; re-run calibration with a
correlated-grid null; red-team re-verification. Baseline results archived at
`research/calibration/output/baseline_a741fa2/`.

Decisions below were made by the lead to fill gaps the audit left open. Items marked **[user]** are
flagged for the user's confirmation in the final report; the calibration measures them.

## Gate set after the fixes (DESIGN §4.2 → v1.2)

| Key | Gate | Threshold | Change |
|---|---|---|---|
| `dsr` | Deflated Sharpe | ≥ 0.95 | **R1/B1.** SR0 = √V0 · E[max of N iid normals], V0 = 1/(T−1) (per-period null sampling variance of a Sharpe estimate, normal-moment approx.), **N = raw trials of this study + raw trials of earlier attempts of the same system** (from the ledger). Eigen / cluster / Li–Ji N_eff and the cross-sectional variance are reported as diagnostics only. |
| `cscv_oos_loss` | P(OOS Sharpe of IS-best < 0), CSCV 16 splits | < 0.10 | **R2/M1.** Replaces the PBO gate. PBO (and slope, degradation) stays as a reported diagnostic. |
| `oos_sharpe` | CPCV path-median OOS Sharpe | ≥ 1.0 | Unchanged. SKIPPED if the candidate set is data-dependent (B3) or the embargo cap binds (m3). |
| `wfo_oos` | **New (B2).** WFO procedure OOS series | whole-span Sharpe ≥ 0.5 **and** Sharpe over the most recent third of WFO-OOS days > 0 **[user]** | The procedure the holdout will run is now gated, including recency. SKIPPED if data-dependent candidate set (B3) or no WFO. |
| `cost_stress_sharpe` | Stressed cost model | > 0.5 | **M2.** Stress = 1.5× spread + **1 pip** adverse slippage on **every market fill** (entries, signal/time/session exits) and stop fills, bar-extreme stops, swap band. Limit/target fills get no slippage. |
| `plateau` | Share of neighbours with Sharpe ≥ 50 % of peak (and > 0) | ≥ 0.60 | **B4:** always recomputed by `gates` with DESIGN constants; the optimizer's score is a diagnostic only. **M5:** neighbourhood on a data-independent economic scale: numeric params within ±20 % (relative, for strictly positive params; else ±20 % of the declared space range), categoricals equal; if a param has no other level inside the radius, its adjacent levels count (coarse grids). Radius can be pre-registered per param on the hypothesis card (`plateau_radius`), default 0.20. |
| `positive_years`, `max_year_share` | Time stability | ≥ 60 %, ≤ 40 % | Unchanged (R3 still open). |
| `trade_count` | MinTRL | ≥ MinTRL | Unchanged. |
| `mechanism` | Ablation | manual | Unchanged. |

## Holdout band (DESIGN §4.4)

- **F4:** trade-count range from the WFO procedure's own daily trade counts (the active config's trades on each OOS day), not the dev-selected trial.
- **M3 / R4:** criteria are set **jointly**: one common tail level chosen so that ≈ 90 % of joint bootstrap draws of the WFO series pass all four criteria at once (instead of four marginals that jointly pass ≈ 78 %).
- **M3:** the band reports its **power against a zero-edge holdout** (bootstrap of the de-meaned series run through `holdout_check`). If P(pass | zero edge) > 0.30, the report states the holdout is not decisive on its own **[user: holdout length / renewing-holdout policy]**. Dead edges are now mostly stopped earlier by `wfo_oos` (B2).

## Other fixes

- **B3:** new `method="sobol"` (scrambled Sobol, seeded, data-independent candidate set). Auto method: grid if ≤ 3 params, else Sobol. TPE stays available for exploration, but its study is marked `candidate_set_data_dependent=True` and the gates SKIP `oos_sharpe`, `wfo_oos`, `cscv_oos_loss` (verdict can't be PASS).
- **M4:** `evaluate_gates` reads the earlier-attempt trial count from the ledger itself (study meta carries book/system/attempt). Raises if attempt > 1, no ledger entry and no explicit prior.
- **F1:** worker processes get their thread caps before Polars/Numba initialise (spawn/forkserver context + env set in the parent at pool creation).
- **m1:** the ledger logs the study's own trial count (`n_trials_study`); cumulative sums use that.
- **m2:** cross-rate lookback for conversion starts ≥ 7 days before the window.
- **m3:** `meta["embargo_capped"]`; when true the OOS gates are SKIPPED with instructions (fewer CPCV groups).
- **m4:** `stats.dsr` rejects per-period Sharpes with |SR| > 1 (annualised input).
- **m9:** `log_gates` records the gate-run index; the report flags earlier verdicts on the same study.
- Not in scope (documented, minor): m5 WFO switch cost, m6, m7, m8, m10.

## Re-verification

1. Full suite + slow suite green.
2. Calibration re-run (`research/calibration/phase1_validator_calibration.py`) with the new gate set, plus new families: **correlated-grid null** (SMA-crossover-like grid on real EURUSD H1, and a shared-entry-schedule random null where only stop/hold vary, ρ ≈ 0.8), **dead-edge** family (planted edge that dies at 45–55 % of dev), and a Sobol vs grid check.
3. Red-team re-verification of B1–B4, M1–M5 with the original probes.
