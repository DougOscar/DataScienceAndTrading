# Phase 1 red-team audit — `stats`, `opt`, `gates`, `evaluators`, `metrics`

- Date: 2026-09-24 · Branch `feat/quantlab-phase1` @ `a741fa2` (library unmodified by this audit)
- Probes: `research/audits/probes/phase1/` (`pNN_*.py` + `.out`), run with `PYTHONPATH=. .venv/bin/python`
- All studies ran in `tempfile` ledgers/studies dirs; `research/ledger/` was not touched; no data at or after 2025-05-15 was requested (spy in p08).
- Severity: **BLOCKER** = invalidates a gate or verdict / successful pipeline exploit · **MAJOR** = changes numbers materially · **MINOR**

## Summary

| # | Sev | Finding | Evidence |
|---|---|---|---|
| B1 | BLOCKER | The DSR gate lets 19–21 % of zero-edge studies through (nominal 5 %). On correlated grids the effective-N (max of eigen and cluster) drops to about 2 | p02, p06 |
| B2 | BLOCKER | An edge that is **dead for the last ~45 % of dev (≈ 2021→2025)** passes every §4.2 gate in 4–5 of 8 seeds. The walk-forward procedure series is never gated | p05b, p05c |
| B3 | BLOCKER (`method="tpe"`) | TPE picks the candidate set using the full-dev objective, so CPCV and WFO "OOS" are not out of sample: +0.17 Sharpe under the null (6σ) | p07 |
| B4 | BLOCKER | The plateau gate takes the optimizer's own `plateau_score`, computed with the optimizer's `PlateauConfig.peak_fraction`. A single-point spike moves from FAIL to PASS | p10 |
| M1 | MAJOR | PBO can be pushed to ≈ 0 by padding the grid with configs that are consistently bad (cost-bleeders) | p03, p06 |
| M2 | MAJOR | The cost-stress "+1 pt slippage" only applies to **stop fills**. 1 pt is 0.1 pip on 5-digit FX, and it does nothing for systems without a stop | p09 |
| M3 | MAJOR | The §4.4 holdout band has almost no power: a zero-edge holdout year passes 17–64 % of the time for gate-passing systems. The band is built from WFO while the gates use CPCV | p05c |
| M4 | MAJOR | Cross-attempt deflation is opt-in: `evaluate_gates` never reads the ledger, and `prior_effective_trials` defaults to 0 | p11 |
| M5 | MAJOR | The plateau gate barely separates signal from noise (passes 77–81 % of null worlds with IS Sharpe > 0) and depends on grid resolution | p06, p10 |
| m1–m10 | MINOR | See below | p01, p08b, p11, p12, reading |

Overall verdict: the Phase 1 exit test ("zero-edge fails ≥ 95 %") is met in my end-to-end null runs (p06: 0/80 PASS). But it is met **only** because of the OOS-Sharpe ≥ 1.0 gate. DSR, PBO and plateau are each lenient or gameable (B1, M1, B4/M5), and the OOS gate itself is contaminated under TPE (B3). A non-null but dead edge gets through (B2). The fixes are small; the protocol should not be used on real systems before B1–B4 are closed.

---

## BLOCKERS

### B1 — Effective-N collapse makes DSR lenient on realistic (correlated) grids
**Code:** `stats.dsr_from_matrix` → `n_eff = max(effective_n_trials(eigen), effective_n_trials(cluster))` together with `var_sr` = cross-sectional variance over **all** trials; `gates.evaluate_gates` uses it.

**Why it is wrong:** under the null with a common factor, the maximum of M equicorrelated Sharpe estimates is `√ρ·Z0 + √(1−ρ)·max_M(iid)`. The cross-sectional SD already equals `√(1−ρ)/√T`, so the correct hurdle is `√V_cross · E[max of M iid]` with **raw M**. Plugging in an n_eff of about 1/ρ² (eigen) or 1 (cluster at ρ ≥ 0.5) on top of V_cross counts the correlation twice. LdP19's ONC recipe pairs K clusters with the variance of **cluster-level** Sharpes, not of all trials.

**Probe p02** (300 null worlds each; select argmax):
```
sma_grid (90 SMA crossover configs on a random walk): n_eff eigen 2.34, cluster 3.00, Li-Ji 20
  P(DSR>=0.95): lib 0.190 | Li-Ji 0.030 | raw N 0.007 | PSR(0) 0.383
  realised max SR 0.49 ann vs lib hurdle 0.16 ann
equicorr rho=0.8, N=100: n_eff eigen 1.55, cluster 1.00
  P(DSR>=0.95): lib 0.213 | Li-Ji 0.057 | raw N 0.043
```
End to end (p06, run_study + evaluate_gates, 10×10 grid, ρ = 0.8): DSR passes 10 % of zero-edge worlds overall and **18 % of those with positive IS Sharpe**. The median n_eff used is 1.55 of 100, and the median hurdle is 0.04 annualised.

**Failure scenario:** an SMA/Donchian pilot (Phase 3) with a 100–500-config grid gets deflated as if it were about 2 trials. The DSR row then reports "95 % probability the Sharpe is real" for a noise pick about 1 time in 5. The docstring already warns that eigen "can be *lenient*", yet the gate still takes the max of two lenient estimators.

**Fix options:** (a) DSR with raw N and cross-sectional V (calibrated for equicorrelation, 4.3 %); (b) Li–Ji n_eff (3–6 % in p02); (c) proper ONC: V = variance of the cluster-representative Sharpes and N = K. Report all of them and gate on the most conservative. Also re-run the Phase 1 calibration with correlated grids, not only ρ-noise synthetic grids.

### B2 — A dead edge passes all gates; the WFO procedure series is never gated
**Probe p05b** (SyntheticEvaluator 7×7, broad bump height H, ρ = 0.5, edge exactly 0 from fraction `from` of dev onward, mechanism forced True):
```
dead_from=0.45 (≈2020-05) H=3.0: PASS 3/8
dead_from=0.55 (≈2021-04) H=2.5: PASS 4/8
dead_from=0.55 H=3.0:          PASS 5/8   (e.g. seed 4: DSR .987 PBO .07 OOS 1.11 plateau .88 posY .70 maxY .29)
```
A 2016–2018-only edge (p05, dead from 0.30) did **not** pass (0/18): max-year-share and OOS caught it. The time-stability gate catches edges that are concentrated in a few years. It does not catch an edge that was spread over 4–5 years and then died.

**p05c**, the same passing studies: the Sharpe over the last 45 % of dev (the dead span) is −0.47 to +0.79, and the WFO-OOS Sharpe over that span is −0.63 to +0.79 (true value 0). None of this reaches a gate: `evaluate_gates` reads `study.wfo_oos` only to build the holdout band. DESIGN §4.6.1 says the schedule "is validated with an anchored walk-forward", but no gate checks it.

**Failure scenario:** a regime-dependent edge (the user's own 2016–18 / 2021–22 observation, DESIGN §4.2 rationale) that died about 2021 passes S5, goes to the holdout, and the weak holdout band (M3) passes it again with 17–64 % probability.

**Fix:** add a gate on the WFO-OOS series of the procedure (e.g. Sharpe ≥ threshold, and/or a recency check such as last-3-years Sharpe > 0 or PSR over the last third), or run CUSUM/`sequential_sharpe_test` over the WFO OOS as a gate.

### B3 — TPE candidate-set leak into CPCV / walk-forward (`method="tpe"`)
**Code:** `opt.run_study(method="tpe")` → `_run_tpe` tells Optuna the **full-dev** `objective_raw`. `cpcv_paths`/`walk_forward` then select, inside each train fold or refit window, only among the trials TPE chose to evaluate. Real strategies have noise that is smooth in parameter space, so TPE concentrates its trials in the region that was lucky on the full sample, test folds and future included. The module docstring's claim that "every refit sees only rows with date < d" does not hold for the candidate set.

**Probe p07** (zero-edge smooth random field over 2 params, 30 paired seeds; grid = all 441 configs vs TPE = 60 configs):
```
grid: mean CPCV path-median OOS Sharpe +0.083 (se .076); WFO OOS -0.030 (se .058)
tpe:  mean CPCV path-median OOS Sharpe +0.251 (se .069); WFO OOS +0.134 (se .069)
paired TPE - grid: CPCV +0.168 (se .027), WFO +0.164 (se .035)
```
**Failure scenario:** any study with more than 3 parameters is forced onto TPE (`grid ≤ 3 params`). Its OOS-Sharpe gate, its holdout band (built from WFO) and its PBO-like diagnostics are then all biased upward. The OOS gate is the one gate that holds under the null (p06), so this removes the last guard. The bias grows with trial concentration (fewer trials, larger spaces).

**Fix:** for CPCV/WFO, re-run the search per train fold/refit (TPE on train rows only). Alternatively, restrict TPE studies to a pre-registered random/Sobol candidate set that does not depend on data, and use TPE only for exploration logged as trials. At minimum, the gate should refuse `method="tpe"` studies for the OOS gate and band.

### B4 — The plateau gate is decided by the optimizer's own configuration
**Code:** `gates.evaluate_gates`, "Plateau" block: `pscore = study.selection.get("plateau_score")`. This value comes from `opt.plateau_select` with the optimizer-chosen `PlateauConfig(peak_fraction, radius, knn, neighbourhood)`. The gate recomputes with DESIGN parameters only when the optimizer did **not** supply a score.

**Probe p10** (single sharp spike, same data):
```
default             gate plateau 0.00 (FAIL) | gate's own recompute 0.00
peak_fraction=0.05  gate plateau 1.00 (PASS) | gate's own recompute 0.00
```
This breaks DESIGN §0.2 (whoever tunes does not judge) and moves the §4.2 threshold "within 50 % of the peak" out of the read-only `GATE_THRESHOLDS`.

**Fix:** always recompute the plateau in `gates` with `PLATEAU_PEAK_FRACTION` and a fixed neighbourhood rule, and report the optimizer's number only as a diagnostic. Alternatively, reject a study whose `selection.neighbourhood.peak_fraction != 0.5`.

---

## MAJOR

### M1 — PBO is gameable by padding the grid with dominated configurations
PBO is a *relative* rank. Configurations that reliably lose (fast SMA pairs churning spread, inverted signals in a trending sample) always rank below the IS-best out of sample, so they push its OOS rank above the median even with zero edge.

**p03** (40 zero-edge configs, ρ = 0.5, plus K bleeders at −3 bp/day): mean PBO 0.485 (K = 0) → 0.072 (K = 40) → 0.005 (K = 160); P(PBO < 0.30) goes from 0.23 to 1.00. At −1 bp/day the effect is weaker (0.56 → 0.28).

**p06** (full pipeline, 40 % of the grid bleeding −1.0 annual Sharpe): the PBO gate's pass rate under the null rises from 0.07 to 0.40.

Realistic grids contain such configs without anyone intending it. The rank/logit implementation itself is correct (p03b).

**Fix:** compute PBO among non-dominated/competitive configs chosen *without* test data (e.g. per-split, keep configs whose IS objective ≥ the IS median), or report PBO against the OOS median of the eligible (`min_centre_quantile`) set. Also report the `slope`/`prob_oos_loss` diagnostics in the gate.

### M2 — The cost-stress "+1 pt slippage" is much weaker than DESIGN intends
`CostModel.stressed()` adds 1.0 to `slippage_points`. The engine uses `slip_price = slippage_points * point` **only in stop checks** (`engine.py:306`, `_chk_long/_chk_short`). Market entries at next-bar open and signal exits get nothing.

**p09** (EURUSD H4, point = 1e-5 = 0.1 pip):
```
with ATR stop:  base -0.4912 | slip+1pt -0.4948 | slip+10pt -0.5290 | stressed() -0.6145
no stop (type C): base -0.3774 | slip+1pt -0.3774 | slip+10pt -0.3774 | stressed() -0.4130 (= spread x1.5 only)
```
**Failure scenario:** a type-C or high-turnover signal-exit system faces only ×1.5 spread in the stress test. The +1 pt part is a no-op, and it would be tiny even if applied (0.1 pip).

**Fix:** decide the unit (DESIGN probably means 1 pip for FX / 1 tick for metals) and apply slippage to every market fill in the stressed model.

### M3 — The §4.4 holdout band has almost no power and is built from a different series than the gates
**p05c**: for studies that passed the gates, the probability that a **zero-edge** 1-year holdout passes `holdout_check` is 0.17, 0.27, 0.64, 0.37 and 0.56. `sharpe_p10` is −0.69 to +0.79, because a 1-year Sharpe has SE ≈ 1, so p10 ≈ median − 1.28. The band uses WFO OOS (`source="wfo"`), while the OOS gate uses CPCV. In p05c seed 4 the gate saw OOS 1.11, but the band median is 0.64.

**Failure scenario:** combined with B2, a dead system passes S5 and then S8 with probability 0.3–0.6. DESIGN says "Fail → killed, no retries", but the exam hardly ever fails a dead system.

**Fix:** a DESIGN decision is needed. Options: require holdout Sharpe > 0 in addition to p10; test the holdout jointly with the newer-data renewing holdout; or use a one-sided test of H0 "SR = predicted median" with stated power. At minimum, report the power of the band against SR = 0 when the band is pre-registered.

### M4 — Cross-attempt deflation depends on the caller remembering it
`evaluate_gates(prior_effective_trials=0.0)` never queries `ledger.system_effective_trials`, even though `study` carries its id. **p11** shows the same study reporting DSR with and without the prior. Nothing warns when `attempt > 1` and `prior == 0`. This undercuts DESIGN §0.3 "every trial counts".

**Fix:** derive the prior inside `evaluate_gates` from the ledger (book/system/attempt from `study.meta` or the ledger row), or raise when `attempt > 1` and no prior is given.

### M5 — The plateau gate barely separates signal from noise and depends on grid resolution
**p06**: under the null the plateau gate passes 42 % of worlds overall and 77–81 % of those with IS Sharpe > 0. A smooth noise surface with a positive common drift looks like a plateau.

**p10**: the **same** narrow bump (width about 2 % of the range) gives plateau 0.00 at grid step 10 or 5 and **1.00 at step 2 or 1**. Finer grids pass for free, and because fine-grid neighbours are nearly collinear they add almost nothing to n_eff (B1).

**Fix:** define the neighbourhood in a data-independent economic scale that is pre-registered on the hypothesis card (e.g. ±20 % of a lookback), not in "grid steps". Measure the plateau on CPCV-OOS Sharpe rather than full-sample Sharpe.

---

## MINOR

- **m1 — Cross-attempt effective trials double-count** (conservative). `log_gates` logs `effective_trials = n_eff` *including* the prior it was given, and `system_effective_trials` sums the logged values. p11: attempt 3 gets prior 16 = 4 + (4 + 8), where 12 is correct. Fix: log the study's own n_eff separately.
- **m2 — RuleEvaluator crashes on non-USD-quoted symbols** when the first evaluation bar follows a gap longer than 5 min. That covers every Monday 00:00 start and the default `start=None` (data start). `data._asof_closes` loads the cross only from `min(ts) − 5 min`. p08b: EURJPY D1 with start 2023-01-02 or None → `ValueError`; GBPAUD D1 with start 2020-06-01 → `ValueError`; a Tuesday start works. It fails loudly, but it blocks the default dev-window evaluation of every cross. Load the cross from `lo − 7 days` or similar.
- **m3 — Embargo/purge sized from full-sample `m_hold_days_max`** over all ok trials. This is a structural statistic and cannot steer selection (negligible leak). But `_resolve_embargo` caps it at a quarter of a group with only a `warnings.warn`, so long-hold systems are silently under-purged. Record the cap in `meta`/ledger and fail the gate if the cap binds.
- **m4 — `stats.dsr()` accepts annualised inputs silently.** p01(d): passing annualised trial Sharpes gives DSR 0.000 (conservative). The opposite mix (annualised `sr`, per-period `var_sr`) would make it maximally lenient. Add a sanity check (e.g. raise if |sr| > 1 per period).
- **m5 — WFO switch model:** at each refit the new config's column is adopted as-is, including positions it opened before `d`. No transition cost is charged for closing the old book and opening the new one. Small at quarterly refits, but it biases WFO OOS (and the holdout band) upward.
- **m6 — Trade-count gate:** per-trade MinTRL is against SR* = 0 with skew/kurt estimated from the trades themselves. A config with few trades and high per-trade Sharpe passes with a handful of trades. `Objective.min_trades` defaults to 0.
- **m7 — Irrelevant parameters inflate the plateau** through exact-copy neighbours: with d_rel relevant and d_irr 3-level irrelevant params, (3^d_irr − 1)/(3^(d_rel+d_irr) − 1) of the neighbours are copies (analytic; about 8–10 % uplift, not enough alone to flip the gate). n_eff and PBO are unaffected by duplicate columns.
- **m8 — CUSUM defaults** (`k = ½·μ/σ`, ARL0 = 2600): p12 median time to alarm after an SR-1.0 edge dies completely is 554 days (about 2.1 y). This is a design note for §4.6 retirement; the ARL formula itself is right (simulated 483 vs target 500).
- **m9 — Gates can be re-run until PASS.** `log_gates` appends, and the folded `ledger.studies()` state shows the last verdict. The history is kept, but nothing flags multiple gate events per study, for example re-runs with a different `mechanism_check` or `swap_band`.
- **m10 — `time_stability`** drops years with < 20 days from both the per-year shares and the total (fine for the FBS dev window; noted for B3 or short windows).

---

## Checks run that found nothing wrong

- **PSR:** matches an independent implementation to machine precision; raw kurtosis, `(γ4−1)/4`; Monte-Carlo coverage 0.0498 vs 0.05 with t5 tails (p01).
- **`expected_max_sharpe`:** within 0.02–0.05 of Monte-Carlo E[max] for N = 2…1000 (the standard approximation error). **MinTRL:** PSR at MinTRL = 0.95 exactly (p01).
- **PBO/CSCV:** ω = rank/(N+1) with rank 1 = worst, average ties, λ ≤ 0 counted as overfit, all C(16,8) splits symmetric. The anti-persistent toy gives 100 % overfit on every unbalanced split (p03b).
- **Stationary bootstrap:** block lengths geometric with the requested mean (9.65 vs 10), uniform indices. The Politis–White block on AR(1) 0.5 gives a bootstrap var(mean) of 0.94× theory (p12).
- **`return_at_dd_budget`:** Brent root is achieved, and the exact scale property holds (k(r/2)/k(r) = 2.0000) (p12).
- **SPA_c / Reality Check:** null rejection 3.3 % / 2.7 % at 5 %, power 69 % for one strategy at 0.1 sd. Hansen's recentring rule matches the paper (p12, reading).
- **BH:** matches a brute-force step-up (p12). **CUSUM:** Siegmund ARL approximation verified by simulation (p12).
- **Evaluator** (p08):
  - Holdout spy: 4 data calls, all with `end ≤ 2025-05-15` and `include_holdout=False`.
  - Scaling all HTF/M1 prices by 1.07 and the conversion rate by 1.5 from 2024-01-02 onward leaves every daily return and trade before the cut bit-identical, and changes everything after it.
  - The engine starts flat: no entry before the first evaluation bar.
  - Sizing mode is correct by risk type (A → fixed_fraction, B/C → fixed_lots, D raises).
  - Conversion uses the close of the last *closed* M1 bar at the bar open (reading of `data._asof_closes`).
- **Selection in CPCV/WFO for grid studies:** objective, `min_centre_quantile`, plateau floor and window trade counts all use train rows only. The neighbourhood matrix depends on parameters only. The low-trades/ok status is full-sample, but both statuses are eligible, so there is no leak (reading of `_select_rows`, `plateau_select`). Grid CPCV/WFO OOS is centred on 0 under the null (p07 grid arm).
- **Gate verdict logic:** any SKIPPED/MANUAL → INCOMPLETE, never PASS (`gates.py:472`). With no trade returns, the trade-count gate is SKIPPED, not PASS. `GATE_THRESHOLDS` is a read-only `MappingProxyType` and matches §4.2.
- **Cost stress model:** `stressed()` = spread ×1.5, `stop_fill="bar_extreme"`, gate value = min over swap ×{0.5, 1, 1.5} (subject to M2 on slippage scope).
- **OOS gate:** the median of per-path annualised Sharpe over `cpcv_paths` (φ = C(9,1) = 9 paths for 10/2), and the paths come from the selection procedure, not from fixed params.
- **Trial accounting:** TPE duplicates are not new columns, so leaving them uncounted is honest for n_eff (they add no information). Invalid configs are never evaluated, so they are correctly absent from n_eff and PBO. Low-trade trials are included in n_eff, var_sr and PBO.
- **End-to-end plain null** (p06, 80 worlds including padded ones): 0 PASS verdicts. The OOS-Sharpe gate and cost-stress rejected every one.
