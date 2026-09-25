# Phase 1 exit test: "validate the validator" (2026-09-24)

**Author:** validation-statistician · **Branch:** `feat/quantlab-phase1` (uncommitted working tree) ·
**Script:** `research/calibration/phase1_validator_calibration.py` ·
**Raw results:** `research/calibration/output/phase1_results.jsonl` (1,180 studies, one JSON row each) ·
**Smoke test:** `tests/test_calibration_smoke.py` (`-m slow`)

Every study went through the real pipeline:
- `opt.run_study`: grid search over 144 configurations, CPCV(10, 2) with auto purge/embargo, anchored WFO (quarterly refit, 2-year minimum training), plateau selection;
- `gates.evaluate_gates`, using the real `RuleEvaluator` with M1 intrabar resolution and the base FBS cost model;
- the §4.2 thresholds exactly as in DESIGN. Nothing was moved.

Each study got its own temporary ledger and studies directory; `research/ledger/` was never written. Only dev data was read. The locked holdout (≥ 2025-05-15) was never touched.

> **Definition used for "PASS" in this report** (`stat_pass`): all 8 statistical gates are PASS.
> These are DSR, PBO, OOS Sharpe, cost stress, plateau, positive years, max-year share and trade count vs MinTRL.
> The mechanism gate is MANUAL by construction, so `evaluate_gates` returns `INCOMPLETE` for every system that clears the statistical gates.
> Treating MANUAL as passed is the most lenient reading, so every null false-pass rate below is an upper bound.

---

## 1. Headline numbers

| Question | Result | Requirement | Status |
|---|---|---|---|
| Null false-pass, real data (440 studies, 4 null families) | **0 / 440 = 0.0 %**, 95 % Clopper–Pearson CI [0.0 %, 0.8 %] | ≤ 5 % | **MET** |
| Null false-pass, all nulls incl. synthetic zero edge (640) | **0 / 640**, CI [0.0 %, 0.6 %] | ≤ 5 % | **MET** |
| Planted edge passes | P(PASS) = 95 % at true net SR 2.35, 90 % at 2.90 | "a planted edge passes" | **MET** |
| Power 80 % point (logistic fit, true net SR of a random config) | **SR ≈ 2.1** (50 % at SR ≈ 1.6) | — | Power is low at SR 1.0–1.5 (see R2) |
| Holdout-band coverage, true edges (pseudo-holdout 2023-05→2024-05) | Joint 4-criterion pass **47/60 = 78 %** [66 %, 88 %]; Sharpe criterion alone **52/60 = 87 %** [75 %, 94 %] | ≈ 90 % for the p10 criterion | Sharpe criterion consistent with 90 %; joint band ≈ 78 % (R4) |
| Holdout-band coverage, gate-passing true edges | 24/31 = 77 % [59 %, 90 %] | — | Winner's-curse effect is small |

**Verdict on the Phase 1 exit criterion (DESIGN §10):**
- The literal criterion is met: zero-edge systems fail at least 95 % of the time (observed 100 %), and a planted edge passes.
- The calibration also exposes two threshold/estimator problems that make the gates **over-strict for real edges**: the PBO gate on flat true plateaus, and DSR on heterogeneous grids.
- Combined with the red team's opposite-direction finding B1 (DSR is lenient on highly correlated grids), **the DSR estimator needs a decision before Phase 3**.
- My nulls use low-correlation grids, so they do **not** cover B1. The 0/640 must not be read as a clearance of B1.

---

## 2. Experimental design

### 2.1 Systems (defined in the script)

| Name | Role | Mechanism | Risk type / sizing |
|---|---|---|---|
| `RandomEntryATR` | NULL | Entry bars and sides come from a splitmix64 hash of (bar timestamp, seed): an entry probability of `rate` per bar, with non-overlapping trades. Stop 2×ATR14, target 3×ATR14, time exit after `hold` bars. It is causal and uses zero information. | A, 1 % fixed-fraction risk |
| `NoisyOracle` | PLANTED | **A calibration device that uses look-ahead on purpose.** It is never a strategy. It uses the same entry schedule, but the side is the sign of `open[i+hold+1] − open[i+1]` (the trade's own fills), correct with probability `p` and flipped otherwise. Safety stop 3×ATR, time exit. | A, 1 % |

- **Grid:** `seed ∈ {0..47}` × `hold ∈ {12, 24, 48}` bars (H1), or `{3, 6, 12}` (H4). That is 144 trials per study.
- Picking the best seed is pure data-mining.
- In the oracle, `seed` is an irrelevant parameter, but `hold` matters (shorter holds have a higher Sharpe). The search therefore does real selection over a true surface that is flat in one dimension.
- Each study has its own master seed (`salt`, fixed through `fixed_params`), so the studies are independent.

### 2.2 Study families

| Family | Symbol / TF | Window | Studies |
|---|---|---|---|
| NULL, costs on | EURUSD H1 | 2016-05-02 → 2025-05-14 | 200 |
| NULL, costs on | XAUUSD H4 | same | 100 |
| NULL, frictionless (spread × 0, swap × 0): the exact-zero-edge real-data null, since with costs on the random entries are *negative*-edge | EURUSD H1 | same | 100 |
| NULL, split window (also used for 4) | EURUSD H1 | 2016-05 → 2023-05-14 | 40 |
| PLANTED, p ∈ {0.54, 0.555, 0.575, 0.60, 0.62, 0.65, 0.68, 0.72} | EURUSD H1 | full dev | 8 × 20 = 160 |
| HOLDOUT, oracle p ∈ {0.60, 0.65, 0.72} | EURUSD H1 | calibrate → 2023-05-14; pseudo-holdout 2023-05-15 → 2024-05-14 | 3 × 20 = 60 |
| SYNTHETIC (`SyntheticEvaluator`, 12×12 grid, ρ = 0.3, 2,340 days, 150 trades/yr) | — | — | zero 200; plateau h ∈ {0.5…3.0} 6 × 40; spike 40; regime 40 |

**True Sharpe.** A planted study's true Sharpe is the mean full-dev net Sharpe across **all** its 144 trials, i.e. the edge of a randomly drawn configuration. Pooled over 20 studies per p, its standard error is ≈ 0.02.

The p → SR mapping (pilot, then confirmed by the run) is: 0.54 → 0.34, 0.555 → 0.56, 0.575 → 0.85, 0.60 → 1.21, 0.62 → 1.50, 0.65 → 1.93, 0.68 → 2.35, 0.72 → 2.90.

---

## 3. NULL: false-pass rate and which gates reject

| Null family | n | PASS | Rate | 95 % CP | Mean trial SR | Median selected SR | Median best-of-144 SR | Median N_eff |
|---|---|---|---|---|---|---|---|---|
| EURUSD H1, costs on | 200 | 0 | 0.0 % | [0.0, 1.8] | −0.44 | −0.00 | 0.44 | 78 |
| XAUUSD H4, costs on | 100 | 0 | 0.0 % | [0.0, 3.6] | −0.33 | 0.12 | 0.61 | 96 |
| EURUSD H1, frictionless | 100 | 0 | 0.0 % | [0.0, 3.6] | −0.03 | 0.37 | 0.79 | 79 |
| EURUSD H1, split window | 40 | 0 | 0.0 % | [0.0, 8.8] | −0.43 | 0.02 | 0.55 | 77 |
| Synthetic zero edge | 200 | 0 | 0.0 % | [0.0, 1.8] | 0.00 | 0.21 | 0.73 | 144 |
| **All real-data nulls** | **440** | **0** | **0.0 %** | **[0.0, 0.8]** | | | | |
| **All nulls** | **640** | **0** | **0.0 %** | **[0.0, 0.6]** | | | | |

### 3.1 Gate-by-gate rejection rate under the null

| Null family | DSR | PBO | OOS SR | Cost stress | Plateau | Pos. years | Max-year share | Trade count | Median gates failed | Min gates failed |
|---|---|---|---|---|---|---|---|---|---|---|
| EURUSD H1 costs | 100 % | 86 % | 100 % | 100 % | 92 % | 66 % | 96 % | 100 % | 8 | 5 |
| XAUUSD H4 costs | 100 % | 79 % | 100 % | 100 % | 70 % | 50 % | 88 % | 99 % | 7 | 4 |
| EURUSD frictionless | 100 % | 95 % | 100 % | 99 % | 26 % | 34 % | 74 % | 97 % | 6 | 4 |
| EURUSD split | 100 % | 88 % | 100 % | 100 % | 90 % | 60 % | 95 % | 100 % | 8 | 5 |
| Synthetic zero | 100 % | 92 % | 100 % | 80 % | 48 % | 41 % | 80 % | 98 % | 7 | 2 |

- **Leave-one-gate-out:** removing any single gate still gives 0 passes in every null family. No gate is load-bearing on these nulls; every null study fails at least 2 gates, and the typical one fails 7 to 8.
- **This differs from red-team p06** ("met only because of OOS ≥ 1.0"). The red-team null uses highly correlated synthetic grids (ρ = 0.8), where DSR collapses. Mine uses nearly independent seeds.

**Standalone null pass rate of each gate** (pooled 640 nulls; a gate's own "alpha" on these nulls):

| Gate | Standalone false-pass |
|---|---|
| DSR (library) | 0.0 % [0.0, 0.6] |
| OOS Sharpe ≥ 1 | 0.2 % |
| PBO < 0.30 | **11.9 %** [9.5, 14.6] |
| Time stability (both rows) | 13.6 % |
| Plateau ≥ 0.60 | **35.5 %** |
| *P(OOS loss of IS-best) < 0.10* (candidate, see R2) | 0.2 % |

### 3.2 Effective-trials sanity check (false-strategy theorem)

Under the null, the expected maximum Sharpe SR0 should match the observed best-of-144 minus the common mean.

| Null family | Observed (best − mean) | SR0, library (cross-sectional V, N_eff) | SR0, V0 = 1/(T−1), N_eff |
|---|---|---|---|
| EURUSD costs | 0.87 | 0.85 | 0.81 |
| EURUSD frictionless | 0.81 | 0.78 | 0.82 |
| XAUUSD | 0.93 | 0.93 | 0.84 |
| Synthetic zero | 0.73 | 0.74 | 0.89 |

- Under the null, both variance estimates agree: √(V·260) is 0.28–0.37 against √(260/T) = 0.333.
- N_eff (max of eigen and cluster: 78–96 here) predicts the data-mining inflation to within ±0.1 annualised Sharpe. The deflation machinery is well calibrated **on this grid structure**.

---

## 4. PLANTED: power curve

| p | n | True net SR (mean over trials) | True SR of selected hold | Median OOS (CPCV) SR | P(PASS) | 95 % CP | Gates rejecting (count of 20) | Only-rejecting gate | Median PBO |
|---|---|---|---|---|---|---|---|---|---|
| 0.540 | 20 | 0.34 | 0.35 | 0.36 | 0 % | [0, 17] | OOS 20, DSR 19, PBO 16, cost 16, trades 14 | — | 0.44 |
| 0.555 | 20 | 0.56 | 0.60 | 0.63 | 0 % | [0, 17] | DSR 19, PBO 17, OOS 17, cost 12, trades 8 | DSR 1 | 0.42 |
| 0.575 | 20 | 0.85 | 0.96 | 0.90 | 5 % | [0, 25] | PBO 16, DSR 12, OOS 11, cost 5 | PBO 4, DSR 1 | 0.42 |
| 0.600 | 20 | 1.21 | 1.39 | 1.39 | 35 % | [15, 59] | PBO 12, DSR 2, OOS 2 | **PBO 9** | 0.36 |
| 0.620 | 20 | 1.50 | 1.74 | 1.63 | 40 % | [19, 64] | PBO 12 | **PBO 12** | 0.32 |
| 0.650 | 20 | 1.93 | 2.25 | 2.15 | 75 % | [51, 91] | PBO 5 | **PBO 5** | 0.24 |
| 0.680 | 20 | 2.35 | 2.75 | 2.61 | 95 % | [75, 100] | PBO 1 | PBO 1 | 0.18 |
| 0.720 | 20 | 2.90 | 3.41 | 3.22 | 90 % | [68, 99] | PBO 2 | PBO 2 | 0.12 |

- Logistic fit of P(PASS) against true net SR: **80 % power at SR ≈ 2.1**, 50 % at SR ≈ 1.6.
- **For true net SR ≥ 1.2, PBO is the only rejecting gate in 29 of the 33 failures.** DSR, OOS Sharpe, cost stress, plateau, time stability and MinTRL all pass.
- The CPCV OOS Sharpe is unbiased for the selected configuration: median OOS SR tracks the true SR of the selected hold within ±0.1–0.2.

---

## 5. SYNTHETIC cross-check

| Scenario | Height | n | True SR, mean over grid | True SR of selected | P(PASS), library | P(PASS), V0-DSR (diagnostic) | Main rejecting gates | Median DSR (lib / V0) | Median SR0 (lib) |
|---|---|---|---|---|---|---|---|---|---|
| Zero | 0 | 200 | 0.00 | 0.00 | 0/200 | 0/200 | DSR 200, OOS 199, trades 196, PBO 184 | 0.05 / 0.02 | 0.74 |
| Broad plateau (8×8 box of 12×12) | 0.5 | 40 | 0.22 | 0.50 | 0/40 | 0/40 | DSR 40, OOS 37 | 0.14 / 0.25 | 1.00 |
| | 1.0 | 40 | 0.44 | 1.00 | 0/40 | 4/40 | DSR 40, OOS 20 | 0.11 / 0.79 | 1.53 |
| | 1.5 | 40 | 0.67 | 1.50 | **0/40** | 25/40 | **DSR 40** | 0.07 / 0.99 | 2.13 |
| | 2.0 | 40 | 0.89 | 2.00 | **0/40** | 40/40 | **DSR 40 (only)** | 0.03 / 1.00 | 2.76 |
| | 2.5 | 40 | 1.11 | 2.50 | **0/40** | 40/40 | **DSR 40 (only)** | 0.01 / 1.00 | 3.40 |
| | 3.0 | 40 | 1.33 | 3.00 | **0/40** | 40/40 | **DSR 40 (only)** | 0.00 / 1.00 | 4.05 |
| Isolated spike (1 grid point) | 3.0 | 40 | 0.02 | 0.30 | 0/40 | 0/40 | DSR 36, trades 35, OOS 34, max-yr 31, plateau 26; PBO 0 | 0.01 / 0.03 | 0.99 |
| Regime (broad edge only in first 30 % of sample) | 3.0 | 40 | 1.33 → 0 | 3.00 → 0 | 0/40 | 1/40 | DSR 40, OOS 30, max-yr 17, pos-yrs 6 | 0.13 / 0.65 | 1.40 |

**Readings:**
- **Spike-only surfaces are rejected (40/40).** Plateau selection centres on the isolated spike in only 4/40 studies, so the pick is noise and fails many gates. The plateau gate fires in 26/40.
- PBO never fires on the spike (0/40). The raw IS-best is the real spike and it ranks high OOS, so PBO correctly says "the argmax is not overfit". Only the plateau machinery treats the spike as fragile.
- **Broad true plateaus are *not* accepted by the library as specified: 0/240, even at a true SR of 3.0.** DSR is the only rejecting gate from height 2.0 up. Its hurdle SR0 is built from the cross-sectional variance of trial Sharpes, and that variance here is dominated by the *planted* difference between the plateau (SR h) and the rest of the grid (SR 0). SR0 grows with the edge itself (1.00 → 4.05) and always stays above the selected Sharpe. See finding F2 / proposal R1.
- **Regime-concentrated edges are rejected (40/40), but not by the time-stability gates.** Positive-years and max-year-share together reject only 17/40 (43 %). An edge alive for about 2.7 of 9 years spreads its PnL over 3 years, each below the 40 % cap (median max-year share 0.38), and 70 % of years are positive.
- DSR and OOS Sharpe carry the rejection of regime surfaces. With the V0-DSR, 1/40 regime surfaces would pass all gates (task `syn-regime-h3.0-025`: max-year share 0.26, positive years 0.8).

---

## 6. Holdout band calibration (dev split, not the real holdout)

**Protocol:**
- `run_study` + `evaluate_gates` run on 2016-05-02 → 2023-05-14. The band comes from `st.holdout_band` via the gates (`source="wfo"`, the default).
- The grid is re-evaluated on 2016-05-02 → 2024-05-14 (dev data only). The frozen WFO procedure (`opt.walk_forward`, same schedule and selection) re-fits inside 2023-05-15 → 2024-05-14 using only rows before each refit date.
- Trade count is counted from whichever configuration was live. Then `st.holdout_check` is applied.

| Group | n | Band PASS (all 4) | Sharpe ≥ p10 | Ret@budget ≥ p10 | MaxDD ≤ p95 | Trades in range | Mean realised HO SR | True HO SR | Median band p10 / median |
|---|---|---|---|---|---|---|---|---|---|
| Null | 40 | 25 (62 %) | 33 | 36 | 38 | 29 | −0.54 | −0.55 | −1.64 / −0.33 |
| p = 0.60 | 20 | 13 (65 %) | 15 | 16 | 15 | 19 | 0.96 | 1.19 | −0.04 / 1.26 |
| p = 0.65 | 20 | 17 (85 %) | 19 | 19 | 19 | 19 | 2.04 | 1.85 | 0.82 / 2.05 |
| p = 0.72 | 20 | 17 (85 %) | 18 | 19 | 19 | 19 | 3.32 | 2.78 | 2.27 / 3.45 |
| **All true edges** | **60** | **47 (78 %)** [66, 88] | **52 (87 %)** [75, 94] | 54 (90 %) | 53 (88 %) | 57 (95 %) | | | |
| True edges that passed the gates | 31 | 24 (77 %) [59, 90] | 27 (87 %) | 29 | 28 | 29 | | | |

- **Sharpe criterion (p10):** coverage for true edges is 87 % (CI [75, 94]), consistent with the design intent of 90 %. There is no measurable winner's curse for gate-passers (27/31 = 87 %).
- **MaxDD (p95):** coverage is 88 % (CI [77, 95]), below its nominal 95 %. That is a mild hint that the WFO-bootstrap drawdown band is slightly tight (the evidence is weak, n = 60).
- **Joint band:** because four criteria must hold at once, a genuine edge fails its holdout **≈ 22 %** of the time. See R4.
- **Nulls pass the band 62 % of the time.** This is expected and is not a defect. The band is a *consistency* test built from the system's own WFO series (a null's band is centred on its own negative Sharpe). Significance is the job of the §4.2 gates; no null reached the holdout (0/40 gate-passes). It should still be stated on every card: "holdout PASS ≠ evidence of edge".
- **Trade-range criterion on nulls: 11/40 fail.** In all 11 the WFO procedure switched `hold` while the band's trade range came from the *fixed* dev-selected trial. For example, dev selection had hold = 48 with range [79, 93], while WFO traded hold = 12 for 112 trades. See F4.

---

## 7. Runtime

| Family | Studies | Median wall per study (1 core, 8-way parallel) | of which run_study | of which gates |
|---|---|---|---|---|
| Real null | 400 | 26 s | 21 s | 5.5 s |
| Real planted | 160 | 28 s | 23 s | 5.0 s |
| Holdout (study + re-evaluation + WFO exam) | 100 | 42 s | 18 s | 3.8 s |
| Synthetic | 520 | 7 s | 2 s | 5.3 s |

- **Total:** 6.8 core-hours, **51 min wall** on 8 workers for the final run of 1,180 studies. One 144-trial real study takes 12 s solo (Polars multi-threaded).
- This is well inside the DESIGN §7 budget: a 500-trial × CPCV study takes < 1 h, and the validation suite takes < 15 min.
- A first attempt was 4× slower because of thread oversubscription (F1).

---

## 8. Library findings (not fixed: reported, with probes)

**F1 (performance, `opt._worker_init`).** The Polars thread cap is set after fork and has no effect.
- `_worker_init` does `os.environ.setdefault("POLARS_MAX_THREADS", "1")` inside the pool initializer.
- With Linux `fork`, the child inherits the parent's already-initialised Polars pool, so every worker keeps 16 threads. That means 8 workers × 16 threads.
- Measured: 4× slowdown when running studies in parallel. My script sets the caps before `import polars`.

Probe (scratch):
```python
pl.DataFrame({"a": [1, 2]}).sum()
with ProcessPoolExecutor(2, initializer=opt._worker_init, initargs=(SyntheticEvaluator(bounds={"x": (0, 1)}), None)) as ex:
    print(list(ex.map(lambda _: pl.thread_pool_size(), range(2))))   # → [16, 16], expected [1, 1]
```
(A module-level function is used in the real probe, because lambdas don't pickle.)

**F2 (statistics design, `stats.dsr_from_matrix`).** `var_sr` is the cross-sectional variance of the trial Sharpes, as in BLdP14.
- Under the null this equals the sampling variance and works (§3.2). When the true Sharpe differs across the grid (any real parameter that matters), the variance contains signal, and SR0 grows with the edge.
- Evidence: the synthetic broad plateau is rejected 240/240 at true SR 0.5–3.0, with DSR as the sole rejecting gate from SR 2.0 up. Median SR0 at h = 3 is 4.05 against a selected SR of 3.0.
- On the real oracle, SR0 rises from 0.79 to 1.44 as p rises. There the `hold` dimension is the only heterogeneity, so DSR still passes.
- Probe: `task syn-plateau-h3.0-000` gives DSR 0.000, PSR(0) ≈ 1, SR 2.63, SR0 ≈ 4.
- This is the mirror image of red-team **B1** (N_eff collapse on ρ ≈ 0.8 grids makes DSR lenient). Both come from estimating (V, N) from a trial matrix whose structure is arbitrary.

**F3 (gate semantics, PBO).** PBO < 0.30 rejects genuine edges whose true surface is flat.
- CSCV-PBO measures whether *selection within the grid* adds OOS value. When every configuration carries the same edge, the IS-best's OOS rank is uniform and PBO ≈ 0.5, whatever the edge's size.
- On the planted oracle, PBO is the **only** rejecting gate in 29/33 failures at true SR ≥ 1.2.
- This conflicts with the plateau gate, which rewards flat surfaces. Together with red-team **M1** (PBO gameable towards 0 by dominated "bleeder" configurations), PBO is miscalibrated in both directions as a hard gate.
- Standalone null alpha of PBO is only 11.9 %.

**F4 (holdout band, `stats.holdout_band` via `gates`).** The trade-count range is built from the fixed dev-selected trial (`selected_trades`), while Sharpe, return at budget and DD come from the WFO series (`source="wfo"`, the procedure that actually runs in the holdout).
- When WFO switches configurations with different trade rates, the trade criterion fails on a procedure that behaves exactly as validated. That happened in 11/40 null split studies, e.g. range [79, 93] against 112 trades after WFO moved to hold = 12.
- It should use the WFO procedure's own daily trade counts (the `trade_counts` matrix gathered along `wfo_params`).

No correctness bugs were found in `opt`, `evaluators`, `gates` or `stats` along the paths exercised: 1,180 studies with 0 errors. The CPCV OOS median is unbiased against the planted truth, and the effective-trials prediction matches the observed null maxima (§3.2).

---

## 9. Recalibration proposals (NOT applied; DESIGN §11 item 2, the user decides)

All counterfactuals below are recomputed from the same 1,180 stored studies. The thresholds were not changed in code.

**R1: DSR hurdle variance.** Build SR0 from the null sampling variance V0 = 1/(T−1) instead of the cross-sectional trial variance. Because of B1, also stop taking N from estimators that can collapse: use the raw trial count, Li–Ji, or ONC representatives, whichever is most conservative.
- Evidence: under the null V0 ≈ V_cross (§3.2), so the test keeps its size (null standalone 0/640).
- On heterogeneous grids it restores power: synthetic plateau P(PASS) goes 0 % → 63 % at SR 1.5 and → 100 % at SR ≥ 2.0.
- V0 alone does *not* fix B1. With N_eff collapsed to ~1.5 on a ρ = 0.8 grid, SR0 → 0 under either V. The N choice is the part that closes B1.

**R2: PBO.** Demote PBO < 0.30 from a hard gate to a reported diagnostic, and replace it with an absolute CSCV criterion: **P(OOS Sharpe of the IS-best < 0) < 0.10** (already computed as `PBOResult.prob_oos_loss`).
- It is absolute, not a relative rank, so padding with bleeders (M1) cannot game it.
- Its standalone null alpha is 0.2 %, against 11.9 % for PBO.
- Counterfactual, with everything else unchanged: null 0/640, and the power 80 % point moves from **SR 2.1 to SR ≈ 1.2**. P(PASS) is 80 % at SR 1.21 and 100 % at SR ≥ 1.5.

**R1 + R2 combined** (V0-DSR with raw N = 144, normal-moment approximation, plus P(OOS loss) < 0.10, all other gates as is):

| Metric | Result |
|---|---|
| Null false-pass | 0/640, CI [0, 0.6 %] |
| Planted P(PASS) at true SR 0.34 / 0.56 / 0.85 / 1.21 / 1.50 / 1.93+ | 0 / 0 / 15 / 70 / 100 / 100 % |
| Power 80 % point | SR ≈ 1.24 |
| Synthetic plateau P(PASS) at SR 1.0 / 1.5 / ≥ 2.0 | 10 / 63 / 100 % |
| Spike | 0/40 |
| Regime | 1/40 |
| Holdout-band joint coverage among the (larger) set of passers | 40/49 = 82 % |

**R3: time stability vs regime concentration.** The current pair of rows (≥ 60 % positive years, no year > 40 %) misses 57 % of 3-year regime edges. I have no calibrated replacement yet. A multi-year concentration measure would need its own null and power study, for example "no 3-consecutive-year window > X % of PnL", or a CUSUM / structural-break test on the dev equity. I flag this as untested rather than propose a number.

**R4: holdout band joint coverage.**
- Each marginal is near nominal: Sharpe 87 % against 90 %; DD 88 % against a nominal 95 %.
- But requiring all four makes a real edge fail its one-shot holdout ≈ 22 % of the time (47/60).
- If a ≈ 10 % false-kill rate is wanted, the marginals should be widened jointly (e.g. Sharpe and return at budget at p5, DD at p97.5), or the band should be defined on the joint bootstrap distribution. That needs a follow-up with more holdout studies (n = 60 is thin).
- Independently, fix F4 so that trade-count failures reflect the procedure.

---

## 10. Limitations

- **Null grid structure.** The nulls are low- to moderate-correlation grids (independent seeds; N_eff 78–144 out of 144). **Highly correlated grids (SMA / Donchian-like, ρ ≈ 0.8) were not calibrated here.** Red-team B1/p06 shows DSR is lenient there, and there the exit test rests on the OOS-Sharpe gate alone.
- Before Phase 3, the calibration should be re-run with a correlated-grid null, e.g. a random-entry null whose entry schedule is shared across configurations and only the stops or holds vary.
- **Planted-edge scope.** The planted edge is uniform across seeds and lives on one symbol (EURUSD H1). The power curve assumes that surface shape.
- **Synthetic cost stress.** `SyntheticEvaluator` has no cost model, so the synthetic cost-stress gate equals the base Sharpe (it was run with a neutral `CostModel`).
- **Mechanism gate** was treated as passed throughout. **Swap** is the exported current rate, with the ×0.5/×1.5 band applied by the gate.
- **Holdout calibration** uses a dev split (2023-05 → 2024-05), one year per study, and 20 studies per edge level.

## 11. Reproduce

```bash
PYTHONPATH=. .venv/bin/python research/calibration/phase1_validator_calibration.py --workers 8      # resumes from the checkpoint
PYTHONPATH=. .venv/bin/python research/calibration/phase1_validator_calibration.py --summarise
.venv/bin/python -m pytest tests/test_calibration_smoke.py -m slow                                 # ~2 min
```
The run is seeded: each study's randomness comes from its `salt` and the bar timestamps, and the bootstrap seed is the salt. Rerunning a task id reproduces its row.
