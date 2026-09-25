# Phase 1 exit test, final confirmation run after round 3 (2026-09-25)

**Author:** validation-statistician · **Branch:** `feat/quantlab-phase1` @ `30b23b6`. The library includes red-team round 2 (R2-1..R2-4, `8356343`): the judge plateau with pre-registered scales, a joint axis and ±1/±2 ordered levels; fixed band seed and n_boot 2000/1000; trial-store and evaluator identity checks; holdout family keys. `quantlab/` was not edited.

**Gate set:** DESIGN §4.2 / §4.4 v1.2. Thresholds were not touched.

**Script:** `research/calibration/phase1_validator_calibration.py --final` (`build_final_tasks`, unchanged).

**Raw results:** `research/calibration/output/phase1_final_r3_results.jsonl` (1,245 studies, **0 errors**).

**Comparisons, paired on task id and salt:**
- `output/phase1_final_results.jsonl`: the final run on `48e696f`, before round 3. Kept as is.
- `output/phase1_v12_results.jsonl`: the v1.2 run.

Earlier reports: `2026-09-24_phase1_calibration.md` (v1.1 baseline, `a741fa2`) and `2026-09-24_phase1_calibration_v12.md`.

Every study ran through the real pipeline: `opt.run_study`, then `gates.evaluate_gates` with the real `RuleEvaluator` and the base FBS cost model. Each study had its own temporary ledger. `research/ledger/` was not touched. No holdout bars were read: the band horizon comes from the data manifest, which is metadata only.

> **`stat_pass`** = all 9 statistical gates PASS. SKIPPED counts as not passed. The mechanism gate is MANUAL and treated as passed, so the null rates are upper bounds.

---

## 0. What changed since the baseline (a741fa2 → 30b23b6)

**v1.1 baseline (`a741fa2`, first report).** Nulls passed 0/640, but the power 80 % point was SR 2.08. PBO and a DSR whose hurdle grew with the edge rejected flat true surfaces.

**Phase 1 fix rounds (`e64efcf`, `02eb9dd`, `7e5ec63`, `dc69e8c`; v1.2 report).** These replaced:
- the DSR hurdle, now from the null variance and the raw trial count (R1);
- PBO, replaced by CSCV P(OOS loss) (R2).

They added:
- the WFO procedure gate with a recency condition;
- a judge-run plateau, scored as the minimum over axes;
- per-asset 1-pip stress slippage;
- the ledger as the source of truth.

Result: 0/1,070 nulls, power 80 % point 1.57, dead-edge pass 3.3 %.

**Later fixes before this run.**
- `29324b1` fixed L1 (the USDCHF evaluation-start clamp) and L4 (holding time measured in rows, so the XAUUSD embargo no longer blows up).
- `48e696f` made the holdout exam "locked year + newer data" with a NOT_DECISIVE verdict.
- The final run on `48e696f` gave a power 80 % point of 1.58. It has no separate report; its checkpoint is the comparison here.

**Red-team round 2 (`8356343`, DESIGN text `30b23b6`).**
- Numeric parameters must pre-register a plateau scale.
- A joint axis was added, and ordered categoricals are judged at ±1/±2 levels.
- Holdout state is keyed on the system family.
- The band is frozen (seed = hash(study_id), n_boot 2000, n_power 1000).
- The gates check the study against the trial store and check the evaluator identity.

**This run measures round 2 only.** Across all 1,245 paired studies, **every non-plateau gate status is identical to the `48e696f` run** (0 differences). Every change below comes from the plateau gate.

---

## 1. Headline

| Question | Final r3 (`30b23b6`) | 48e696f (paired) | v1.2 (paired) | Requirement | Status |
|---|---|---|---|---|---|
| Null false-pass, real-data nulls (660; 8 families incl. correlated grids, Sobol, USDCHF) | **0 / 660**, 95 % CP [0.0, 0.6 %] | 0/660 | 0/540 | ≤ 5 % | **MET** |
| Null false-pass, all nulls incl. synthetic (760) | **0 / 760** [0.0, 0.5 %] | 0/760 | 0/640 | ≤ 5 % | **MET** |
| Leave-one-gate-out, all nulls | 0 passes with any single gate removed | 0 | 0 | — | No single gate carries the null result |
| Planted edge passes (grid, true net SR ≥ 1.93) | 38/40 = 95 % [83, 99] | 40/40 | 40/40 | "a planted edge passes" | **MET** |
| Planted edge passes (Sobol, true SR 1.95 / 1.51) | 20/20 / 14/15 | 20/20 / 14/15 | 10/20 / — | — | MET |
| **Power, 80 % point** (logistic on true net SR, 160 paired grid studies) | **SR 1.76** (50 %: 1.45) | **1.58** (50 %: 1.37) | **1.58** (1.57 on v1.2's full 200) | — | **Worse by 0.18**, paired bootstrap 95 % [+0.01, +0.45]. All of it comes from the hold axis's ±2 level (§4) |
| Dead-edge pass (dies at 45/55 % of dev) | **3/90 = 3.3 %** [0.7, 9.4] | 3/90 | 3/90 | — | Unchanged |
| Holdout band decisive at the manifest horizon (262 d), gate-passing real studies | **48/104 = 46 %** [36, 56] | 49/104 | — | — | Decisive only for true SR ≳ 1.9 (§8) |
| Runtime | 58.5 min wall for 1,144 studies + 6 min pilot for 101; **8.6 core-h**; gates median 7.7 s per real study | 9.4 core-h; gates 7.6 s | 12.1 core-h | ≤ 2 h | OK |

**Verdict on the DESIGN §10 Phase 1 exit criterion: MET.**
- Zero-edge systems fail ≥ 95 % of the time: observed 760/760, upper 95 % CI of the false-pass rate 0.5 %.
- A planted edge passes: 95–100 % at true net SR ≥ 1.9, on grid and on Sobol candidate sets.

The round 2 plateau rule costs measurable power: the 80 % point goes from 1.58 to 1.76. On these families, the measured cost is all from the ±2-level perturbation of a coarse ordered categorical. It buys no measurable null protection. That is a design trade-off to be decided knowingly (§4.3). It does not affect the exit criterion.

---

## 2. Scales pre-registered by each family (check requested)

| Space (families) | Parameters and declared scale | What the judge does | Sensible? |
|---|---|---|---|
| `seedhold` (NULL EUR / EUR0 / XAU / CHF, PLANTED, DEAD) | `seed` int 0–47, `plateau_step = 9.4` (0.2 × range); `hold` ordered categorical (12, 24, 48) H1 / (3, 6, 12) H4 | `seed`: ±5, ±9 seeds. On the oracle the seed is irrelevant by construction, so the axis acts as a replicate: another random schedule with the same edge. `hold`: levels ±1 and ±2. **±1 = ×2 or ×½ the hold, ±2 = ×4 or ×¼.** No joint axis: only one numeric parameter | `seed`: yes. It equals the pre-round-3 fallback (r × range), so the seed-axis shares match `48e696f` in 610/610 studies. **`hold`: no.** The ±2 level is a ×4 change against a ×1.2 change on numeric axes. In log units that is 1.39 against 0.18, ~8× larger. See §4.3 |
| `exits` (CORR shared) | `stop_mult` 2–4, `target_mult` 3–6, `hold` 36/48, all `relative` (r = 0.20) | stop ±0.3/±0.6; hold at 48 → 38, 43, 53, 58, which are off-grid, outside the search bounds and still evaluated. Joint axis over 3 parameters | Yes |
| `sma` (CORR sign-random SMA) | `fast` 10–40, `slow` 120–240, `stop_mult` 2–4, all `relative` | fast = 10 → 8, 9, 11, 12; slow = 200 → 160, 180, 220, 240. Joint axis over 3 | Yes |
| `sobol4` / `sobol4_oracle` | `seed` step 9.4; `hold` 12–48, `stop_mult`, `target_mult` `relative` | 4 axes plus a joint axis that also moves the seed | Yes. The joint point is "another replicate with perturbed exits" |
| Synthetic | `x`, `y` int 0–11, `plateau_step = 2.2` | ±1 and ±2 grid cells; joint axis over 2 | Yes, the same cells as before |

The identity and trust checks (R2-4) raised no false refusal in 1,245 studies. The holdout family keys (R2-2) are not exercised by this run: it has no unlock.

---

## 3. NULL: false-pass per family (Clopper–Pearson 95 %)

| Family | n | PASS, r3 | 95 % CP | 48e696f (paired) | v1.2 (paired) |
|---|---|---|---|---|---|
| EURUSD H1, costs (seed × hold) | 100 | 0 | [0, 3.6] | 0/100 | 0/100 |
| EURUSD H1, frictionless | 100 | 0 | [0, 3.6] | 0/100 | 0/100 |
| XAUUSD H4, costs | 100 | 0 | [0, 3.6] | 0/100 | 0/100 |
| **USDCHF H1, costs** (L1 check) | 60 | 0 | [0, 6.0] | 0/60 | not runnable (L1) |
| CORR shared schedule, frictionless (ρ ≈ 0.82) | 100 | 0 | [0, 3.6] | 0/100 | 0/100 |
| CORR sign-random SMA, frictionless (ρ ≈ 0.60) | 100 | 0 | [0, 3.6] | 0/100 | 0/100 |
| SOBOL null, frictionless (4 params) | 60 | 0 | [0, 6.0] | 0/60 | 0/40 |
| SOBOL null, costs | 40 | 0 | [0, 8.8] | 0/40 | 0/40 |
| Synthetic zero | 100 | 0 | [0, 3.6] | 0/100 | 0/100 |
| **All real-data nulls** | **660** | **0** | **[0, 0.6]** | 0/660 | |
| **All nulls** | **760** | **0** | **[0, 0.5]** | 0/760 | |

Standalone null pass rate per gate (760 nulls; each gate alone):

| Gate | Pass rate |
|---|---|
| dsr | 0.0 % |
| oos_sharpe | 0.1 % |
| cscv_oos_loss | 1.2 % |
| trade_count | 2.4 % |
| cost_stress | 3.3 % |
| wfo_oos | 4.3 % |
| **plateau** | **10.8 %** (48e696f: 12.8 %) |
| max_year_share | 14.5 % |
| positive_years | 49.5 % |

### 3.1 Gate-by-gate rejection (reject = FAIL or SKIPPED)

| Family | n | dsr | cscv | oos | wfo_oos | cost | plateau | pos_years | max_year | trades | Min / median gates failed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EURUSD costs | 100 | 100 % | 100 % | 100 % | 97 % | 100 % | 100 % | 67 % | 97 % | 99 % | 6 / 9 |
| XAUUSD H4 | 100 | 100 % | 100 % | 100 % | 96 % | 100 % | 100 % | 50 % | 88 % | 98 % | 5 / 8 |
| EURUSD frictionless | 100 | 100 % | 100 % | 100 % | 95 % | 100 % | 99 % | 34 % | 74 % | 97 % | 5 / 8 |
| USDCHF costs | 60 | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 90 % | 100 % | 100 % | 8 / 9 |
| CORR shared, frictionless | 100 | 100 % | 91 % | 100 % | 89 % | 100 % | **58 %** (was 53 %) | 41 % | 83 % | 93 % | 3 / 8 |
| CORR SMA, frictionless | 100 | 100 % | 100 % | 100 % | 97 % | 100 % | **72 %** (was 70 %) | 45 % | 85 % | 99 % | 5 / 8 |
| SOBOL null, frictionless | 60 | 100 % | 100 % | 100 % | 97 % | 100 % | 93 % | 43 % | 80 % | 97 % | 6 / 8 |
| SOBOL null, costs | 40 | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 75 % | 100 % | 100 % | 8 / 9 |
| Synthetic zero | 100 | 100 % | 100 % | 99 % | 95 % | 75 % | **93 %** (was 90 %) | 37 % | 75 % | 98 % | 3 / 8 |
| *PLANTED grid, all p* | 160 | 26 % | 5 % | 22 % | 8 % | 53 % | **21 %** (was 8 %) | 0 % | 1 % | 5 % | 0 / 1 |
| *DEAD, all* | 90 | 44 % | 12 % | 43 % | 89 % | 79 % | 24 % (was 17 %) | 1 % | 3 % | 12 % | 0 / 3 |

- XAUUSD: `oos_sharpe` and `cscv_oos_loss` are now real FAILs, with **0/100 SKIPPED**. In v1.2 all 100 were SKIPPED (L4 closed, §7).
- In every null family, five gates each reject ≥ 89 %: dsr, cscv, oos, wfo_oos (≥ 89 %) and cost stress (≥ 75 % on synthetic, 100 % on real data). No null has fewer than 3 failing gates.

---

## 4. PLANTED: power (same 20 salts per p as 48e696f and v1.2)

| p | True net SR | r3 PASS [95 % CP] | 48e696f | v1.2 | Changed (48e696f → r3) | Failing gates r3 (count; only-gate in brackets) |
|---|---|---|---|---|---|---|
| 0.555 | 0.56 | 0/20 [0, 17] | 0 | 0 | — | dsr 20, cost 19, oos 18, … |
| 0.575 | 0.85 | 0/20 [0, 17] | 0 | 0 | — | cost 19, dsr 13, oos 11, plateau 7 (1) |
| 0.59 | 1.07 | 3/20 [3, 38] | 3 | 3 | — | cost 17 (6), plateau 7, dsr 6, oos 5 |
| 0.60 | 1.21 | 7/20 [15, 59] | 8 | 9 | salt 012: 1 → 0 | cost 11 (5), plateau 5 (2), … |
| 0.61 | 1.35 | 9/20 [23, 69] | 10 | 10 | salt 012: 1 → 0 | cost 10 (8), plateau 3 (1) |
| 0.62 | 1.50 | 10/20 [27, 73] | 11 | 11 | salt 012: 1 → 0 | cost 9 (8), plateau 2 (1) |
| 0.65 | 1.93 | **19/20** [75, 100] | 20 | 20 | salt 012: 1 → 0 | plateau 1 (1) |
| 0.72 | 2.90 | **19/20** [75, 100] | 20 | 20 | salt 012: 1 → 0 | plateau 1 (1) |

| Gate set (logistic fit, 160 planted grid studies) | 50 % point | **80 % point** |
|---|---|---|
| **r3 as built** | 1.45 | **1.76** |
| 48e696f, same ids | 1.37 | **1.58** |
| v1.2, same ids | 1.36 | **1.58** (1.57 on v1.2's 200) |
| r3 without cost stress | 1.14 | 1.44 |
| r3 without plateau | 1.35 | 1.59 |
| r3, plateau scored without the `hold` axis | 1.37 | 1.58 |
| r3 with the matrix plateau instead of the judge | 1.36 | 1.58 |

- Paired bootstrap (1,000 resamples of the 160 task ids): x80(r3) − x80(48e696f) = **+0.18**, 95 % [+0.01, +0.45]. Against v1.2: +0.19 [+0.02, +0.46].
- The 5 planted studies that flipped PASS → FAIL all share salt 3012, where the oracle schedule for (seed 0, hold 12) is selected.

### 4.1 Where the loss comes from

The loss is entirely the `hold` axis:
- The run has 46 plateau PASS → FAIL transitions, and each one comes from a newly scored point.
- 33 are the ±2 `hold` level on seed × hold studies: planted 21, dead 7, null 5. The hold share goes 1.0 → 0.5, with 24 passing and 48 (or 12) failing. The seed-axis shares are identical to `48e696f` in 610/610 studies.
- The other 13 are the joint axis: correlated nulls 7, synthetic zero 3, synthetic plateau 2 (true SR 0.5–1.0), spike 1.
- With the `hold` axis removed, the power curve is back to 1.58 exactly.

### 4.2 Why a genuine flat-accuracy edge fails the ±2 level

The oracle's accuracy does not depend on `hold`, but its Sharpe does, through trade frequency. The mean true net SR by hold (12 / 24 / 48) is, for example:
- p = 0.60: 1.39 / 1.28 / 0.96;
- p = 0.72: 3.41 / 3.05 / 2.23.

So the ×4 level keeps only 0.65–0.75 of the peak's *true* Sharpe. The judge compares it with the *re-evaluated selected* configuration, which carries the winner's-curse bias of a max over 144 trials. For example, the peak is 4.14 against a true 3.41 at hold 12 (p 0.72, salt 012). The hold-48 point then falls below 50 % of the peak by sampling noise. That happens to 2–6 of 20 studies per p at SR ≤ 1.2, and to 1/20 at SR ≥ 1.9. Selected hold is 12 in 235/250 planted and dead studies.

### 4.3 Design consequence (for the main session; no threshold moved)

For ordered categoricals, the ±2-level rule makes the perturbation size whatever the level spacing happens to be. Here a 3-level geometric grid turns it into a ×4 move, while numeric parameters move ±20 %.
- **Cost:** +0.18 on the power 80 % point, and 1/20 true edges rejected at SR 1.9–2.9 by plateau alone.
- **Benefit on the nulls:** none. Null false-pass is 0/760 with or without it, and leave-one-gate-out gives 0 either way.
- **Anti-gaming benefit:** the part R2-1 targets (an author who hides fragility behind a coarse categorical) is not measured by these families.

DESIGN §11 #8 quotes "power 80 % point SR 1.57". Under the round 2 rule and this family, the measured value is **1.76**.

---

## 5. Judge plateau vs matrix plateau, and the joint axis

| Family | n | Judge PASS, r3 | Judge PASS, 48e696f | Judge without joint axis | Joint axis < 0.6 | Joint = weakest | Matrix PASS (SKIPPED) | Median judge evals / s |
|---|---|---|---|---|---|---|---|---|
| NULL seed × hold (EUR, EUR0, XAU, CHF) | 360 | **1 (0.3 %)** [0.0, 1.5] | 6 | 1 (no joint axis) | — | — | 1 (0) | 7 / 0.9 |
| CORR shared, frictionless | 100 | **42 %** [32, 52] | 47 % | 47 % | 55/100 | 37 | 53 % (0) | 21 / 3.2 |
| CORR SMA, frictionless | 100 | **28 %** [20, 38] | 30 % | 30 % | 49/100 | 18 | 47 % (0) | 21 / 3.0 |
| SOBOL null | 100 | **4 %** [1, 10] | 4 % | 4 % | 85/100 | 17 | 24 % (32) | 25 / 3.9 |
| SYN zero | 100 | **7 %** [3, 14] | 10 % | 10 % | 74/100 | 15 | 16 % (0) | 17 / 0.1 |
| **All nulls** | 760 | **82 = 10.8 %** [8.7, 13.2] | 97 = 12.8 % | 92 = 12.1 % | 263/400 | 87 | 141 = 18.6 % (32) | 17 / 1.0 |
| PLANTED grid, all p | 160 | **78.8 %** [72, 85] | 91.9 % | 78.8 % (no joint axis) | — | — | 95.6 % (0) | 7 / 1.1 |
| PLANTED grid, p ≥ 0.60 | 100 | **88 %** [80, 94] | 99 % | 88 % | — | — | 100 % | 7 / 1.1 |
| SOBOL oracle p = 0.65 | 20 | **100 %** | 100 % | 100 % | 0/20 | 0 | 50 % (10) | 25 / 4.1 |
| SOBOL oracle p = 0.62 | 15 | **100 %** | 100 % | 100 % | 0/15 | 1 | 47 % (8) | 25 / 4.2 |
| DEAD, all | 90 | 75.6 % | 83.3 % | 75.6 % | — | — | 87.8 % | 7 / 1.1 |
| SYN plateau, h ≥ 1.5 | 80 | 100 % | 100 % | 100 % | 0/80 | 15 | 100 % | 17 / 0.1 |
| SYN spike | 40 | **0 %** [0, 8.8] | 1/40 | 1/40 | 32/40 | 5 | 12.5 % | 17 / 0.1 |
| SYN regime | 40 | 72.5 % | 72.5 % | 72.5 % | 6/40 | 10 | 77.5 % | 17 / 0.1 |

**Joint-axis effect:**
- **Null side:** the joint axis is below 0.6 in 263 of the 400 null studies that have one. It is the sole reason the plateau fails in 13 studies, and it removes 10 of the 97 null plateau passes (the ±2 hold levels remove 5 more). The correlated-grid nulls, the case the joint axis was meant for, lose 5 + 2 plateau passes. On the ρ = 0.82 grid the plateau still passes 42 %: a correlated grid is a plateau by construction, and five other gates reject it.
- **True-edge side:** it costs almost nothing. The joint axis failed in 0/35 Sobol oracles and 0/80 synthetic plateaus with h ≥ 1.5. It blocked one overall PASS in the whole run: `syn-plateau-h1.0-003`, true SR 1.0.
- **Spike surface:** 1/40 → 0/40.

**Judge vs matrix:**
- On the nulls the judge is stricter: 10.8 % against 18.6 %.
- On Sobol sets the judge never SKIPs, while the matrix box is empty in 32/100 nulls and 18/35 oracles.
- On the grid oracle the judge is now less permissive than the matrix (78.8 % against 95.6 %), because of the ±2 hold level (§4). The matrix only looks at adjacent levels.

---

## 6. DEAD edge

| p | Death | r3 PASS | 48e696f | v1.2 | Without `wfo_oos` |
|---|---|---|---|---|---|
| 0.62 | 45 % / 55 % | 0/15 / 0/15 | 0 / 0 | 0 / 0 | 0 / 0 |
| 0.68 | 45 % / 55 % | 0/15 / 2/15 | 0 / 2 | 0 / 2 | 0 / 3 |
| 0.72 | 45 % / 55 % | 1/15 / 0/15 | 1 / 0 | 1 / 0 | 3 / 9 |
| **Pooled** | | **3/90 = 3.3 %** [0.7, 9.4] | 3/90 | 3/90 | 15/90 = 16.7 % [9.6, 26.0] |

- The same 3 studies pass in all three runs. Their recent-third WFO Sharpe is small and positive by chance: 0.08, 0.34, 0.12.
- The round 2 plateau changes did not touch them. It made 7 dead studies fail the plateau, but none of those passed the other gates.

---

## 7. Sobol vs grid; USDCHF and XAUUSD sanity

| Family | n | True SR | PASS r3 | 48e696f | Plateau PASS / SKIPPED | Failing gates |
|---|---|---|---|---|---|---|
| Sobol oracle p = 0.65 | 20 | 1.95 | **20/20** [83, 100] | 20/20 | 20 / 0 | — |
| Grid planted p = 0.65 | 20 | 1.93 | 19/20 [75, 100] | 20/20 | 19 / 0 | plateau 1 |
| Sobol oracle p = 0.62 | 15 | 1.51 | **14/15** [68, 100] | 14/15 | 15 / 0 | cost stress 1 |
| Grid planted p = 0.62 | 20 | 1.50 | 10/20 [27, 73] | 11/20 | 18 / 0 | cost stress 9, plateau 2 |

- **L3 is closed.** On a Sobol set, a true edge now passes at least as often as on a grid (v1.2: 10/20, all plateau SKIPPED). The Sobol null is still 0/100.
- The Sobol oracle at p = 0.62 passes more often than the grid at the same true SR. Its exit space differs: stop 2–5, target 2–8, continuous hold, so there is lower turnover in the selected configuration and cost stress bites less. It is not a like-for-like power comparison.
- **USDCHF H1 (L1):** 60/60 studies ran, 0 errors, median 32 s per study (v1.2: every study aborted after ≈ 8 min). Embargo 3 days, OOS gates evaluated (0 SKIPPED), 0/60 pass. The selected Sharpe is negative after costs (median −0.33), as expected for random entries.
- **XAUUSD H4 (L4):** embargo median 3 days (v1.2: 55), capped 0/100, `oos_sharpe` and `cscv_oos_loss` SKIPPED 0/100 (v1.2: 100/100), 0/100 pass.

---

## 8. Holdout band at the manifest horizon (DESIGN §4.4)

- **Horizon:** from the manifest in 945/945 real studies. Holdout start 2025-05-15, horizon end 2026-05-15 10:36, 262 trading days. No newer data is in the manifest, so this is the locked year only.
- **Band settings:** n_boot 2000 and n_power 1000 in every study. The band seed is now hash(study_id): 945 distinct seeds, against 715 in `48e696f`, where the salt was shared across families.

| Group | n | Median P(zero-edge holdout passes the band) | Decisive (≤ 0.30) | 48e696f decisive |
|---|---|---|---|---|
| Planted p = 0.555 … 0.61 (SR 0.6–1.35) | 100 | 0.79 → 0.46 | 0/100 | 1/100 |
| Planted p = 0.62 (SR 1.50) | 20 | 0.47 | 4/20 | 4/20 |
| Planted p = 0.65 (SR 1.93) | 20 | 0.28 | 11/20 | 12/20 |
| Planted p = 0.72 (SR 2.90) | 20 | 0.03 | 20/20 | 20/20 |
| **Gate-passing real studies** | **104** | **0.32** | **48/104 = 46 %** [36, 56] | 49/104 |
| … true SR < 1.5 | 31 | 0.48 | 0/31 [0, 11] | 0/31 |
| … true SR 1.5–2.0 | 50 | 0.28 | 26/50 = 52 % [37, 66] | 27/50 |
| … true SR ≥ 2.5 | 19 | 0.03 | 19/19 | 19/19 |
| Dead edges that passed the gates | 3 | 0.69 | 0/3 | 0/3 |

**Readings:**
- With one locked year, the exam is decisive for about half of the systems that pass the gates. That matches the §4.4 note ("~55 % for one year").
- Below true SR 1.5 it is never decisive: those systems would wait for newer data as `holdout_pending`.
- The three gate-passing dead edges are all NOT_DECISIVE, so they cannot be promoted on a holdout PASS.
- The seed change moves the zero-edge pass probabilities by ≤ 0.04. That is Monte Carlo noise: the SE at n_power = 1000 is ≈ 0.015.

---

## 9. Runtime

- **Wall time:** 58.5 min for 1,144 studies plus ≈ 6 min for the 101-study pilot (both into the same checkpoint), on 8 workers. A red-team agent ran in parallel on ≤ 4 workers.
- **Core time:** 8.6 core-h in total, of which the gates took 2.6 core-h (`48e696f`: 9.4 and 2.56).
- **Gate run time per study (median):**
  - real studies 7.7 s (p90 10.6 s), 48e696f: 7.6 s;
  - synthetic 5.4 s (5.1 s).

  The band was already 2000/1000 in the `48e696f` run, so fixing it adds nothing.

- **Judge plateau per study (median):**

  | Space | Time | Evaluations |
  |---|---|---|
  | seed × hold | 0.9–1.1 s | 7 |
  | correlated grids | 3.0–3.2 s | 21 |
  | Sobol 4-param | 3.9–4.2 s | 25 |
  | synthetic | 0.1 s | 17 |

- **Whole study, gates included (median):**

  | Family | Time |
  |---|---|
  | null | 28 s |
  | planted | 32 s |
  | corr | 32 s |
  | sobol | 35 s |
  | synthetic | 7.5 s |

---

## 10. Library findings

No correctness bug was found:
- 1,245/1,245 studies completed;
- the identity checks raised no false refusal;
- all non-plateau gate statuses reproduce the `48e696f` run exactly.

The one material finding is a design consequence, not a code defect: the ±2-level ordered-categorical perturbation and its power cost (§4.2–4.3). Probe:

```bash
# any of these rows: plateau_diag.pass_share_by_param == {'seed': ≥0.75, 'hold': 0.5}, selected hold 12;
# true_sharpe_by_hold shows the ×4 level keeping ~0.67 of the true Sharpe
grep '"planted-eur-p0.720-012"' research/calibration/output/phase1_final_r3_results.jsonl
```

## 11. Limitations

- **No planted edge on a correlated grid.** The joint axis's power cost is measured only on Sobol, synthetic and the grid oracle, where the seed × hold space has no joint axis.
- **One planted system shape** (EURUSD H1, ≈ 173 trades/yr). The hold-axis cost depends on how Sharpe scales across the categorical's levels. For a system whose edge per trade grows with holding time, the ±2 level could cost nothing, or more.
- **The R2-1 anti-gaming benefit** (an author choosing a coarse parametrisation) is a red-team matter and is not measured here.
- **Holdout decisiveness is the band's own bootstrap estimate** at the locked-year horizon. No empirical dead-at-holdout exam was run in this set; see the v1.2 report §7.

## 12. Reproduce

```bash
.venv/bin/python research/calibration/phase1_validator_calibration.py --final --workers 8 \
    --checkpoint research/calibration/output/phase1_final_r3_results.jsonl   # resumes; prints summaries
```

Every study is seeded (salt, and hash(study_id) for the band), so re-running a task id reproduces its row. The paired comparison and the bootstrap of the power difference were computed from the three checkpoints; seed 12345, 1,000 resamples.
