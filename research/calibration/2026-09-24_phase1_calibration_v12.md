# Phase 1 exit test, v1.2 re-run: "validate the validator" after the fix round (2026-09-24)

**Author:** validation-statistician · **Branch:** `feat/quantlab-phase1` @ `e64efcf` (library unchanged; only the calibration script and the smoke test were edited) ·
**Gate set:** DESIGN §4.2 / §4.4 v1.2 (`quantlab/gates.py`, thresholds not touched) ·
**Script:** `research/calibration/phase1_validator_calibration.py` ·
**Raw results:** `research/calibration/output/phase1_v12_results.jsonl` (1,806 studies, one JSON row each; 2 errors, see L1) ·
**Baseline (v1.1, read-only):** `research/calibration/output/baseline_a741fa2/phase1_results.jsonl` ·
**Previous report:** `2026-09-24_phase1_calibration.md` (kept) ·
**Smoke test:** `tests/test_calibration_smoke.py -m slow` passes (6 tests, including a new correlated-grid null test).

Every study went through the real pipeline:
- `opt.run_study`: grid, or Sobol for the 4-parameter family, CPCV(10, 2) with auto purge/embargo, anchored WFO (quarterly refit, 2-year minimum training);
- `gates.evaluate_gates`, with the real `RuleEvaluator` (M1 intrabar) and the base FBS cost model.

Each study got its own temporary ledger. `research/ledger/` was never touched, and no data at or after 2025-05-15 was read: `RuleEvaluator` refuses `end` past the holdout start, and the 2-year pseudo-holdout ends 2025-05-15, exclusive.

The baseline families keep their task ids and salts, so every v1.2 row is paired with its v1.1 row.

> **`stat_pass`** means all 9 statistical gates are PASS: `dsr`, `cscv_oos_loss`, `oos_sharpe`, `wfo_oos`, `cost_stress_sharpe`, `plateau`, `positive_years`, `max_year_share`, `trade_count`.
> SKIPPED counts as not passed. The mechanism gate is MANUAL and is treated as passed, so the null rates are upper bounds.

---

## 1. Headline

| Question | v1.2 result | v1.1 baseline | Requirement | Status |
|---|---|---|---|---|
| Null false-pass, all real-data nulls (870 studies, 10 families incl. correlated grids and Sobol) | **0 / 870**, 95 % CP [0.0 %, 0.4 %] | 0/440 | ≤ 5 % | **MET** |
| Null false-pass, all nulls incl. synthetic (1,070) | **0 / 1,070** [0.0 %, 0.3 %] | 0/640 | ≤ 5 % | **MET** |
| Correlated-grid null, ρ = 0.82 (shared schedule) / 0.60 (sign-random SMA) | **0/200, 0/150** | not run | ≤ 5 % | **MET: B1 closed empirically** |
| Planted edge passes | 100 % (60/60) at true net SR ≥ 1.93 | 90–95 % at SR ≥ 2.35 | "a planted edge passes" | **MET** |
| Power, 80 % point (logistic, true net SR of a random config) | **SR ≈ 1.57** (50 % at 1.36) | 2.08 (same study ids) | — | Better than v1.1, worse than the R1+R2 prediction of 1.24 (all of the gap is the M2 cost stress, §4) |
| Dead edge (dies at 45 / 55 % of dev) passes | **4 / 120 = 3.3 %** [0.9, 8.3] | without `wfo_oos`: 19/120 = 15.8 % [9.8, 23.6] | — | `wfo_oos` removes about 80 % of dead-edge passes |
| Sobol (4-param) null false-pass | 0 / 80 [0, 4.5] | grid: 0/400 | same as grid | **MET** |
| Holdout band joint coverage, true edges, 1-year | **49/60 = 82 %** [70, 90] | 47/60 = 78 % (v1.1 marginal band) | ≈ 90 % | Below target; 90 % is at the edge of the CI |
| Holdout band zero-edge pass (bootstrap), median, 1 y → 2 y | 0.27 → 0.07 (0.58 → 0.38 at SR 1.2) | not reported | ≤ 0.30 = "decisive" | Decisive in 33/60 → 49/60 studies |
| Runtime | 90.6 min wall on 8 workers; 12.1 core-h | 51 min for 1,180 studies | ≤ 2 h | OK |

**Verdict on the Phase 1 exit criterion (DESIGN §10): MET.**
- Zero-edge systems fail at least 95 % of the time: observed 100 %, 0/1,070, upper CI 0.3 %. This now includes highly correlated grids (red-team B1) and data-independent Sobol candidate sets.
- A planted edge passes: 100 % at true net SR ≈ 1.9 and above.
- Leave-one-gate-out: in every null family, removing any single gate still gives 0 passes. No single gate carries the null result.

Four findings qualify how the gates will behave on real systems (§8). None is a false-pass risk; all of them cost power:
- the M2 cost stress is now the binding gate for high-turnover H1 systems (L2);
- the plateau gate on Sobol sets is SKIPPED half the time, which blocks PASS (L3);
- a data gap in XAUUSD makes every XAUUSD study from 2016-05 skip its OOS gates (L4);
- USDCHF cannot be evaluated from the dev start at all (L1).

---

## 2. Families run (v1.2)

| Family | System / grid | Studies | Trials / study | Salt range |
|---|---|---|---|---|
| NULL EURUSD H1, costs / frictionless | random entries, seed (48) × hold (12/24/48) | 200 / 100 | 144 | as v1.1 |
| NULL XAUUSD H4, costs | same, holds 3/6/12 | 100 | 144 | as v1.1 |
| NULL split window (holdout family) | EURUSD H1 to 2023-05-15 | 40 | 144 | as v1.1 |
| **CORR shared schedule** (new) | random entries and sides **shared by every config** (one draw per study, schedule spaced 48 bars); grid over stop 2–4 ATR (0.25 step) × target 3–6 ATR (0.5 step) × hold 36/48 | 100 frictionless + 100 costs | 126 | 11000+, 12000+ |
| **CORR sign-random SMA** (new) | SMA crossover, fast 10–40 × slow 120–240 × stop 2/3/4 ATR; direction × a random ±1 per calendar month shared by all configs, so the gross edge is exactly 0 in expectation | 100 frictionless + 50 costs | 147 | 13000+, 14000+ |
| **SMA real** (new, descriptive) | same grid, no sign flip, 8 H1 symbols × {costs, frictionless} | 16 (2 errors) | 147 | — |
| PLANTED oracle | p ∈ {0.54 … 0.72}; **0.59 and 0.61 added** | 10 × 20 | 144 | as v1.1 (3000+) |
| **DEAD edge** (new) | same oracle and salts as PLANTED; accuracy drops to 0.5 from 2020-05-26 (45 %) or 2021-04-21 (55 %); p ∈ {0.62, 0.68, 0.72} | 6 × 20 | 144 | 3000+ (paired with the live planted studies) |
| **SOBOL** (new) | 4 params (seed, hold 12–48, stop 1.5–4, target 1.5–6); `method="sobol"`, 144 points; null frictionless / costs; oracle p = 0.65 (stop 2–5, target 2–8) | 40 + 40 + 20 | 144 | 16000+ … 18000+ |
| HOLDOUT | oracle p ∈ {0.60, 0.65, 0.72} × 20 + null × 40; calibrate to 2023-05-15; exam 1 y (→ 2024-05-15) and 2 y (→ 2025-05-15); each oracle also examined with its edge **killed at 2023-05-15** | 100 | 144 | as v1.1 |
| SYNTHETIC | as v1.1 (zero 200, plateau 6 × 40, spike 40, regime 40) | 520 | 144 | as v1.1 |

---

## 3. NULL: false-pass per family and the effective-trials calculation

| Family | n | PASS | 95 % CP | Mean pairwise ρ | N_eff eigen / cluster / Li–Ji (median) | Raw N (gate) | Median SR0, v1.2 (V0, raw N) | Median best − mean trial SR | DSR alone passes: v1.2 | DSR alone passes: v1.1 (N_eff, V_cross) |
|---|---|---|---|---|---|---|---|---|---|---|
| EURUSD H1 costs | 200 | 0 | [0, 1.8] | 0.02 | 78 / 48 / 110 | 144 | 0.88 | 0.87 | 0 | 0 |
| XAUUSD H4 costs | 100 | 0 | [0, 3.6] | 0.02 | 89 / 96 / 120 | 144 | 0.91 | 0.93 | 0 | 0 |
| EURUSD H1 frictionless | 100 | 0 | [0, 3.6] | 0.02 | 79 / 48 / 110 | 144 | 0.88 | 0.82 | 0 | 0 |
| EURUSD split window | 40 | 0 | [0, 8.8] | 0.02 | 77 / 48 / 110 | 144 | 1.00 | 0.99 | 0 | 0 |
| **CORR shared, frictionless** | 100 | 0 | [0, 3.6] | **0.82** | **1.5 / 1.0 / 15** | 126 | 0.87 | 0.29 | **0** | **10 (10 %)** [4.9, 17.6] |
| **CORR shared, costs** | 100 | 0 | [0, 3.6] | **0.82** | 1.5 / 1.0 / 15 | 126 | 0.87 | 0.29 | 0 | 2 |
| **CORR sign-random SMA, frictionless** | 100 | 0 | [0, 3.6] | **0.60** | 2.7 / 1.0 / 41 | 147 | 0.89 | 0.46 | 0 | 4 |
| **CORR sign-random SMA, costs** | 50 | 0 | [0, 7.1] | 0.59 | 2.8 / 1.0 / 42 | 147 | 0.89 | 0.47 | 0 | 0 |
| **SOBOL null, frictionless** | 40 | 0 | [0, 8.8] | 0.01 | 86 / 90 / 118 | 144 | 0.88 | 0.85 | 0 | 0 |
| **SOBOL null, costs** | 40 | 0 | [0, 8.8] | 0.01 | 85 / 90 / 118 | 144 | 0.88 | 0.87 | 0 | 0 |
| Synthetic zero | 200 | 0 | [0, 1.8] | 0.30 | 10 / 144 / 102 | 144 | 0.89 | 0.73 | 0 | 0 |
| **All real-data nulls** | **870** | **0** | **[0, 0.4]** | | | | | | | |

**Effective-trials reading (B1).**
- On the ρ = 0.82 grid, the eigen and cluster N_eff collapse to 1–1.5. The v1.1 hurdle then drops to a median SR0 of **0.02**, and the v1.1 DSR alone passes **10 %** of exact-zero-edge studies [4.9, 17.6]. That is B1, reproduced empirically on real data.
- The v1.2 hurdle (V0 = 1/(T−1), raw N = 126) is 0.87, and DSR alone passes 0/350 correlated nulls.
- **The v1.2 hurdle is conservative on correlated grids.** The observed data-mining inflation at ρ = 0.82 is only 0.29 (best − mean), against a hurdle of 0.87. On independent real-data grids the two agree to within ±0.07 (synthetic zero: 0.89 against 0.73). On the ρ = 0.6 SMA grid the inflation is 0.46.
- The cost is power on genuinely correlated grids with a real edge. That was not measured here: no planted edge on a correlated grid (limitation).
- The plain SMA grid on 8 real symbols (ρ 0.64–0.71, N_eff ≈ 2) fails everywhere (14/14). DSR, OOS Sharpe, cost stress and trade count fail in all 14. The best is GBPJPY frictionless: selected SR 0.71, failing 4 gates.

### 3.1 Gate-by-gate rejection under the null (reject = FAIL or SKIPPED)

| Family | dsr | cscv_oos_loss | oos_sharpe | wfo_oos | cost_stress | plateau | pos_years | max_year | trade_count | Min / median gates failed |
|---|---|---|---|---|---|---|---|---|---|---|
| EURUSD costs | 100 % | 100 % | 100 % | 97 % | 100 % | 100 % | 66 % | 96 % | 100 % | 6 / 9 |
| XAUUSD H4 | 100 % | 100 %\* | 100 %\* | 96 % | 100 % | 100 % | 50 % | 88 % | 99 % | 5 / 8 |
| EURUSD frictionless | 100 % | 100 % | 100 % | 95 % | 100 % | 99 % | 34 % | 74 % | 97 % | 5 / 8 |
| CORR shared, frictionless | 100 % | 91 % | 100 % | 89 % | 100 % | 47 % | 41 % | 83 % | 93 % | 3 / 8 |
| CORR shared, costs | 100 % | 99 % | 99 % | 98 % | 100 % | 77 % | 78 % | 96 % | 99 % | 2 / 9 |
| CORR SMA, frictionless | 100 % | 100 % | 100 % | 97 % | 100 % | 53 % | 45 % | 85 % | 99 % | 5 / 8 |
| CORR SMA, costs | 100 % | 100 % | 100 % | 100 % | 100 % | 76 % | 60 % | 94 % | 100 % | 6 / 8.5 |
| SOBOL null, frictionless | 100 % | 100 % | 100 % | 98 % | 100 % | 65 % (15 SKIPPED) | 48 % | 82 % | 98 % | 6 / 8 |
| SOBOL null, costs | 100 % | 100 % | 100 % | 100 % | 100 % | 95 % (11 SKIPPED) | 75 % | 100 % | 100 % | 7 / 9 |
| Synthetic zero | 100 % | 100 % | 100 % | 92 % | 80 % | 84 % | 41 % | 80 % | 98 % | 3 / 8 |

\* All 100 XAUUSD H4 studies have these two gates **SKIPPED**, because the embargo cap binds (L4).

- New gates, standalone null alpha (pooled 1,070 nulls): `cscv_oos_loss` 11/1,070 pass (1.0 %); `wfo_oos` 50/1,070 (4.7 %).
- On the correlated nulls, plateau alone passes about 50 % (a correlated grid is a plateau by construction). Five gates (DSR, CSCV, OOS, cost stress, trade count) each still reject ≥ 91 %.

---

## 4. PLANTED: power curve (same 20 salts per p as v1.1)

| p | True net SR | Median OOS SR | v1.1 PASS | **v1.2 PASS** [95 % CP] | Without `wfo_oos` | Recent-third ≥ 0.5 | Without cost stress | Failing gates (count / 20), v1.2 |
|---|---|---|---|---|---|---|---|---|
| 0.54 | 0.34 | 0.36 | 0 | 0 [0, 17] | 0 | 0 | 0 | dsr 20, oos 20, cost 20, cscv 19, wfo 15, … |
| 0.555 | 0.56 | 0.63 | 0 | 0 [0, 17] | 0 | 0 | 0 | dsr 20, cost 19, oos 17, … |
| 0.575 | 0.85 | 0.90 | 1 | 0 [0, 17] | 0 | 0 | 3 | cost 19, dsr 13, oos 11 |
| 0.59 | 1.07 | 1.19 | — | 3 (15 %) [3, 38] | 3 | 3 | 13 | cost 17 (only gate: 10), dsr 6, oos 4 |
| 0.60 | 1.21 | 1.39 | 7 | 9 (45 %) [23, 68] | 9 | 9 | 15 | cost 11 (only: 6), dsr 2, oos 2 |
| 0.61 | 1.35 | 1.53 | — | 10 (50 %) [27, 73] | 10 | 10 | 20 | **cost 10 (only gate)** |
| 0.62 | 1.50 | 1.63 | 8 | 11 (55 %) [32, 77] | 11 | 11 | 20 | **cost 9 (only gate)** |
| 0.65 | 1.93 | 2.15 | 15 | **20 (100 %)** [83, 100] | 20 | 18 | 20 | — |
| 0.68 | 2.35 | 2.61 | 19 | 20 (100 %) | 20 | 20 | 20 | — |
| 0.72 | 2.90 | 3.22 | 18 | 20 (100 %) | 20 | 20 | 20 | — |

Logistic fits of P(PASS) on true net SR (200 studies):

| Gate set | 50 % point | **80 % point** |
|---|---|---|
| v1.1 (baseline, same 160 study ids) | 1.61 | **2.08** |
| R1+R2 counterfactual predicted in the v1.1 report | — | **1.24** |
| **v1.2 as built** | 1.36 | **1.57** |
| v1.2 without `cost_stress_sharpe` | 1.03 | **1.16** |
| v1.2 without `wfo_oos` | 1.36 | 1.57 |
| v1.2 with recent-third ≥ 0.5 | 1.39 | 1.65 |
| v1.2 without plateau | 1.35 | 1.59 |

**Readings:**
- R1 + R2 work as predicted. Remove the one gate that changed independently (M2 stress) and the 80 % point is 1.16, against the 1.24 predicted. PBO no longer rejects flat true surfaces, and DSR no longer grows with the edge.
- **The whole gap to the prediction is the M2 cost stress.** The oracle trades ≈ 173 times a year on EURUSD H1, with 1 % risk on a 3-ATR stop.
  - The stress (1.5× spread + 1 pip on every market and stop fill, bar-extreme stops) costs a median **1.34 Sharpe**, flat across p (1.33–1.35).
  - Probe: slippage alone removes 21.9 points per trade, i.e. 2 fills × 10 points plus stop fills. That is correct per the spec; I found no double charge.
  - So the gate demands a base Sharpe of about 1.85 for this turnover. Between SR 1.07 and 1.50, cost stress is the only rejecting gate in 35 of the 47 failures.
  - This is the gate working as specified, not a miscalibration of the statistics. How high the bar sits for high-turnover systems is a design choice that should be made knowingly (see §7).
- `wfo_oos` with recent > 0 costs **no measurable power**: 0 studies change. Standalone, it rejects 2/140 live edges with SR ≥ 1.
- The plateau gate (judge-computed, ±20 %) costs no power on the grid oracle (1 study).
- Synthetic plateau (heterogeneous grid, R1's target case): P(PASS) at true SR 0.5 / 1.0 / 1.5 / ≥ 2.0 = 0 / 2 / 62 / 100 %. The v1.1 library gave 0 % throughout, and the R1 prediction was 0 / 10 / 63 / 100 %.
- Spike 0/40; regime (edge only in the first 30 % of the sample) 0/40. In v1.1 the V0 diagnostic let 1/40 regime studies through.

---

## 5. DEAD edge (B2)

Same oracle, same salts as PLANTED: identical to the live study before the death date. After it, the side accuracy is 0.5 and the net SR ≈ −0.23.

| p | Death | SR alive / dead / full dev | **v1.2 PASS** | Without `wfo_oos` | Recent > 0 (= v1.2) | Recent ≥ 0.5 | `wfo_oos` rejects | Median WFO SR all / recent third |
|---|---|---|---|---|---|---|---|---|
| 0.62 | 45 % | 1.47 / −0.24 / 0.54 | 0/20 | 0 | 0 | 0 | 19 | 0.23 / −0.31 |
| 0.62 | 55 % | 1.48 / −0.22 / 0.71 | 0/20 | 0 | 0 | 0 | 17 | 0.40 / −0.34 |
| 0.68 | 45 % | 2.32 / −0.24 / 0.93 | 0/20 | 0 | 0 | 0 | 16 | 0.45 / −0.29 |
| 0.68 | 55 % | 2.34 / −0.22 / 1.19 | 2/20 | 4 | 2 | 0 | 17 | 0.72 / −0.59 |
| 0.72 | 45 % | 2.87 / −0.24 / 1.18 | 1/20 | 4 | 1 | 0 | 17 | 0.66 / −0.25 |
| 0.72 | 55 % | 2.90 / −0.22 / 1.50 | 1/20 | **11** | 1 | 0 | 16 | 1.01 / −0.45 |
| **Pooled** | | | **4/120 = 3.3 %** [0.9, 8.3] | **19/120 = 15.8 %** [9.8, 23.6] | 4/120 | **0/120** [0, 3.0] | 102/120 | |

- **Without the WFO gate, a dead strong edge (SR 2.9, dead for the last 45 % of dev) passes 55 % of the time.** With `wfo_oos` it passes 5 %.
- The whole-span condition (≥ 0.5) alone stops none of them (19/120 pass without recency). The recency condition does all the work.
- The 4 remaining passers had a positive recent-third WFO Sharpe by chance. SE ≈ 0.66 on about 2.3 years, against a true −0.23.
- Without both `wfo_oos` and cost stress, 48/120 = 40 % would pass. Cost stress also rejects dead edges, because their full-dev Sharpe is diluted.

---

## 6. SOBOL candidate set (B3)

| Family | n | PASS | Plateau SKIPPED (0 neighbours) | Median neighbours in the ±20 % box |
|---|---|---|---|---|
| Sobol null, frictionless | 40 | 0 [0, 8.8] | 15 | 1 |
| Sobol null, costs | 40 | 0 [0, 8.8] | 11 | 2 |
| Sobol oracle p = 0.65 (true SR 1.95) | 20 | **10 (50 %)** [27, 73] | **10: plateau is the only failing gate in all 10** | 0.5 |
| Grid oracle p = 0.65 (for comparison) | 20 | 20 (100 %) | 0 | 19–37 |

- The null false-pass on Sobol sets (0/80) is at the level of the grid (0/400).
- But on a 4-parameter Sobol set of 144 points, the ±20 % box around the selected point holds 0–3 other points. Expected: about 143 × 0.4 × 0.33 × 0.44 × 0.33 ≈ 2.8 in the interior, fewer at the edges.
  - Half the time it is empty → plateau SKIPPED → the verdict cannot be PASS.
  - When it is not empty, the score rests on 1–3 neighbours.
- See L3.

---

## 7. Holdout band (§4.4 v1.2), dev split: calibrate to 2023-05-15, examine 1 y and 2 y

Band: WFO series + stationary bootstrap, common tail level α (median 0.037–0.045), trade range from the WFO procedure's own entry counts (F4).

| Group | Horizon | Joint coverage, live edge | S / R / D / T criteria passed | Zero-edge pass, band bootstrap (median) | Decisive (≤ 0.30) | Edge killed at 2023-05-15: empirical pass |
|---|---|---|---|---|---|---|
| p = 0.60 (HO SR ≈ 1.2) | 1 y | 16/20 | 18 / 18 / 17 / 19 | **0.58** | 0/20 | 6/20 |
| | 2 y | 15/20 | 15 / 15 / 17 / 19 | **0.38** | 9/20 | 6/20 |
| p = 0.65 (≈ 1.9) | 1 y | 17/20 | 19 / 19 / 19 / 19 | **0.27** | 13/20 | 1/20 |
| | 2 y | 18/20 | 19 / 19 / 20 / 19 | **0.07** | 20/20 | 1/20 |
| p = 0.72 (≈ 2.8) | 1 y | 16/20 | 19 / 20 / 19 / 18 | 0.03 | 20/20 | 1/20 |
| | 2 y | 17/20 | 19 / 19 / 19 / 19 | 0.00 | 20/20 | 0/20 |
| **All true edges** | **1 y** | **49/60 = 82 %** [70, 90] | | median 0.27 (mean 0.28) | **33/60** | **8/60 = 13 %** [6, 25] |
| | **2 y** | **50/60 = 83 %** [72, 92] | | median 0.07 (mean 0.15) | **49/60** | **7/60 = 12 %** [5, 23] |
| True edges that passed the gates | 1 y / 2 y | 38/46 = 83 % / 38/46 = 83 % | | 0.15 / 0.02 | | 4/46 / 4/46 |
| Null (costs) | 1 y / 2 y | 20/40 / 14/40 | trades 22/40 / 16/40 | 0.93 / 0.93 | 0/40 | — |

**Readings:**
- **Joint coverage is 82–83 %, not the 90 % the band is built for.** Each marginal is 92–95 %; the upper CI (90–92 %) just includes the target.
  - The joint construction improved things over v1.1 (78 %), but the bootstrap of the WFO series is still slightly under-dispersed relative to a real next period.
  - A true edge fails a one-shot holdout ≈ 17 % of the time.
- **Zero-edge power depends on edge size far more than on horizon choice alone:**
  - at SR ≈ 1.2 a 1-year holdout is never decisive (P(pass | zero edge) = 0.58), and even 2 years leaves 0.38;
  - at SR ≈ 1.9, 1 y gives 0.27 and 2 y gives 0.07;
  - at SR ≥ 2.8 both are decisive.
- The empirical dead-edge check (the same study, edge killed at the holdout start, net SR ≈ −0.3 to −0.5 over the holdout) agrees in ranking: 30 % pass at SR 1.2, 5 % at 1.9, 0–5 % at 2.8.
  - With n = 20 per level, the 1 y vs 2 y difference in this empirical check is not resolved.
- Null bands: the trade-range criterion fails 18/40 (1 y). The null's WFO switches between holds with 2–4× different trade rates quarter by quarter. The bootstrap block (median 7.8 days) does not carry that regime, so the range is too narrow.
  - This is irrelevant for true edges with stable selection (4/60 trade failures), and no null reaches the holdout.

---

## 8. Decision support (DESIGN §11 #8), no decisions taken, thresholds unchanged in code

### 8.1 `wfo_oos` recent-third threshold: pre-declared candidates, same 200 planted + 120 dead studies

| Candidate | Power 80 % point | P(PASS) at SR 1.93 | `wfo_oos` alone rejects live edges (SR ≥ 1) | Dead-edge pass (full gate set) |
|---|---|---|---|---|
| (no recency condition / no WFO gate) | 1.57 | 20/20 | — | 19/120 = 15.8 % [9.8, 23.6] |
| **> 0** (current v1.2) | **1.57** | 20/20 | 2/140 = 1.4 % [0.2, 5.1] | **4/120 = 3.3 %** [0.9, 8.3] |
| **≥ 0.5** | **1.65** | 18/20 | 9/140 = 6.4 % [3.0, 11.9] | **0/120 = 0 %** [0, 3.0] |

**Trade-off in one line:** moving from > 0 to ≥ 0.5 removes the remaining ~3 % dead-edge passes. The price is ~0.08 on the power 80 % point, and about 5 % of live edges rejected by this gate alone.

### 8.2 Holdout horizon: zero-edge pass probability of the band (dev split)

| True edge (HO SR) | 1-year horizon | 2-year horizon |
|---|---|---|
| ≈ 1.2 | 0.58 (decisive 0/20) | 0.38 (decisive 9/20) |
| ≈ 1.9 | 0.27 (13/20) | 0.07 (20/20) |
| ≈ 2.8 | 0.03 (20/20) | 0.00 (20/20) |
| All 60 true edges | median 0.27, **decisive 33/60** | median 0.07, **decisive 49/60** |
| Joint coverage of live edges | 82 % | 83 % |

- The FBS holdout as defined is 1 year (2025-05-15 → 2026-05-15).
- For systems at the power frontier (SR 1.2–1.9), a 1-year holdout passes a zero-edge continuation 27–58 % of the time. A 2-year horizon brings that to 7–38 % at unchanged coverage.
- Getting a second year requires the renewing holdout (post-2026-05-15 data).

---

## 9. Library / data findings (reported with probes; `quantlab/` not edited)

**L1 (bug, `data.conversion_rate` via `RuleEvaluator._data`): USDCHF cannot be evaluated from the dev start.**
- The first USDCHF H1 bar is 2016-05-02 00:00, but the first M1 bar (the conversion source for CHF→USD) is 00:01. The conversion lookup raises for the first bar, so every trial errors and `run_study` aborts.
- The same applies to any symbol whose quote or swap currency converts via a series starting after the first bar: AUDCHF, AUDNZD, NZDCHF and NZDUSD also start at 00:01.
- Each failed study also takes ≈ 8 min, because `_data()` reloads M1 per trial after the exception; nothing is cached on failure.
```python
b = data.load_bars('USDCHF', 'H1', book='FBS', start=datetime(2016,5,2), end=datetime(2016,5,3))
data.conversion_rate('CHF', 'USD', b['ts_utc'].head(3), book='FBS')        # ValueError: no M1 bar had opened yet …
data.conversion_rate('CHF', 'USD', b['ts_utc'].slice(1, 3), book='FBS')    # works
```

**L2 (design consequence, M2 stress, not a code bug): the cost-stress gate is now the binding gate for high-turnover H1 systems.**
- Median haircut is 1.34 Sharpe at 173 trades/yr, 1 % risk, 3-ATR stops.
- 35 of the 47 planted-edge failures between SR 1.07 and 1.50 are cost-stress-only.
- Probe: `planted p=0.62 salt 3000, {seed:5, hold:24}` gives base SR 1.27, slippage-only stress 0.64 (−21.9 points/trade: 2 × 10 plus stop fills), full stress 0.23.

**L3 (design, plateau × Sobol): the ±20 % box neighbourhood is empty in 36/100 Sobol studies** → plateau SKIPPED → the verdict can never be PASS.
- It halves the power of a p = 0.65 Sobol oracle (10/20 against 20/20 on grid).
- When it is not empty, 1–3 neighbours make the score coarse.
- Probe: any `sobol-oracle-p0.650-*` row with `plateau_diag.n_neighbours == 0` (e.g. 10 of the 20).

**L4 (data + `opt._resolve_embargo`): a 133-day XAUUSD M1 gap (2016-06-24 09:48 → 2016-11-04 18:19) makes every XAUUSD study from the dev start skip its CPCV gates.**
- One trade straddling the gap sets `m_hold_days_max` ≈ 55 weekdays for the whole study. The auto embargo takes the max over trials, the cap binds (m3), and `oos_sharpe` and `cscv_oos_loss` are SKIPPED in 100/100 XAUUSD H4 studies.
- v1.1 had the same 55-day embargo, but no skip logic.
- The embargo is driven by calendar holding time, so data gaps inflate it. A gap-aware holding measure (bars held × bar length), or excluding gap-straddling trades, would avoid this.
- Probe: `null-xau-000`: `embargo_days = 55`, `embargo_capped = True`, while the trial's `hold_days_max` is 3.

No other correctness issues found in 1,804 completed studies.

---

## 10. Limitations

- **No planted edge on a correlated grid.** The raw-N DSR is shown to be safe on ρ = 0.6–0.8 nulls. Its conservatism is not quantified for real edges on such grids: hurdle 0.87 against an actual inflation of 0.29.
- The planted edge is one system shape: EURUSD H1, ≈ 173 trades/yr, a flat surface. The power curve, and especially the cost-stress effect, depends on turnover and stop distance.
- The dead-edge families use a single death shape (an abrupt drop to 0.5 accuracy). Gradual decay was not tested.
- Holdout: n = 20 per edge level. The "killed at holdout start" check has net SR ≈ −0.3 to −0.5 (costs), so it is slightly easier to reject than an exact zero edge. The band's bootstrap zero-edge probability is the exact-zero version.
- R3 (time stability vs regime concentration) is still open. Regime surfaces are rejected (40/40), but by DSR, OOS and WFO rather than by the time-stability rows.

## 11. Reproduce

```bash
PYTHONPATH=. .venv/bin/python research/calibration/phase1_validator_calibration.py --workers 8   # resumes from output/phase1_v12_results.jsonl
PYTHONPATH=. .venv/bin/python research/calibration/phase1_validator_calibration.py --summarise
.venv/bin/python -m pytest tests/test_calibration_smoke.py -m slow
```
Every study is seeded (salt + bar timestamps); re-running a task id reproduces its row. The script refuses to append to the archived v1.1 baseline file.
