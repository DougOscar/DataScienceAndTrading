**DRY RUN: toy system** (DESIGN §10 Phase 2 exit test; sandbox `research/dryrun/`). None of the numbers below is evidence about H4 TSMOM. S1 already returned KILL-EARLY.

# S5 validation: `toy_tsmom`, study `fbs-0023-a1` (issue #23)

Validation statistician, 2026-09-30. Gates: DESIGN §4.2 v1.2, computed by `quantlab.gates.evaluate_gates`. It ran **once** with `log=True`, and that run is ledger event seq 3, `gates`, `gate_run` 1. Evaluator: `RuleEvaluator(ToyTsmom, "EURUSD", "H4", book="FBS")`, with M1 intrabar resolution and the default FBS cost model `fbs-v2-uncalibrated`, the same as S3/S4. Periods/year = 260, and Sharpe is annualised from the daily %-equity returns. `mechanism_check=None`, so the mechanism gate is MANUAL and the evidence is given in §3. No mechanism review was recorded.

## 1. How the StudyResult was obtained (read-only)

There is no read-only loader (FINDINGS #34). I did **not** use `run_study(resume=True)`, for two reasons:
- It appends `resumed`, `trials` and `selection` rows to the ledger.
- It recomputes CPCV/WFO and rewrites the artifact parquet files.

Instead, `results/probes/s5_load_study.py` rebuilds the StudyResult from the public store readers (`ledger.load_trials`, `load_trial_returns`, `load_study_artifact`) and from the ledger's `study_created` and `selection` rows. It writes nothing.

The gates' own guards were then run read-only as acceptance tests (`probes/s5_precheck.py`):
- `ledger_context` passed.
- `verify_trial_store` passed. It checks the 840 trial ids, status and source, the return content, the selection trial 492 and params, and the sha256 of all 5 artifacts.
- `check_evaluator_identity` passed. It checks the describe(), the cost version and the strategy source sha `edddb832…`.

No trials were re-run: n_store = n_ledger = 840. The ledger held 3 rows before the gate run and 4 after, and the only new row is `gates`.

## 2. Gate table (GateReport, verbatim)

| Gate | Value | Threshold | Status | Interpretation |
|---|---|---|---|---|
| Deflated Sharpe probability | 0.000197 | >= 0.95 | **FAIL** | DSR 0.00: 0% probability the true Sharpe exceeds the 1.07 expected from the best of N = 840 raw null trials (840 this study + 0 from related earlier studies) over 2352 days. |
| CSCV OOS loss probability | 0.789 | < 0.1 | **FAIL** | The in-sample-best config lost money out of sample in 79% of 12870 CSCV splits (median OOS Sharpe -0.30); PBO 0.53 (diagnostic). |
| OOS Sharpe (CPCV median) | -0.683 | >= 1.0 | **FAIL** | Median OOS Sharpe -0.68 across 9 CPCV paths (IQR -0.78–-0.51); 0% of paths positive; in-sample -0.12. |
| Walk-forward procedure OOS | 0.07 / recent -0.07 | >= 0.5 and recent ⅓ > 0 | **FAIL** | WFO OOS Sharpe 0.07 over 1826 days (2018-05-02 → 2025-05-14); -0.07 over the most recent third (2023-01-10 → 2025-05-14); the procedure does not hold up out of sample. |
| Cost-stress Sharpe | -0.917 | > 0.5 | **FAIL** | Worst Sharpe -0.92 at 1.5× spread + 1 pip (10 points) slippage on all market and stop fills, bar-extreme stops, swap ×0.5 / 1 / 1.5 (-0.92, -0.92, -0.92); base -0.12. |
| Parameter plateau | 0 | >= 0.6 | **FAIL** | Weakest axis lookback 0/4 judge-run perturbations at ≥ 50% of peak Sharpe -0.12 (peak ≤ 0 → nothing can pass); 6 points outside the search bounds evaluated; selected at search-space edge (stop_mult = 1.0); 21 evaluations. |
| Positive years | 0.6 | >= 60% | **PASS** | 60% of 10 years positive; losing years [2016, 2018, 2019, 2025]. |
| Max single-year PnL share | inf | <= 40% | **FAIL** | Total PnL ≤ 0: no share defined. |
| Trade count vs MinTRL | 670 trades / need inf | ≥ MinTRL (95%) | **FAIL** | 670 trades vs per-trade MinTRL inf; 2352 days vs daily MinTRL inf at claimed Sharpe -0.68 (95%). |
| Mechanism check | — | ablation matches hypothesis | **MANUAL** | The user decides on the §3 evidence (not recorded). |

The full GateReport, including the plateau points and diagnostics, is in `results/probes/s5_gate_report.md`. The machine-readable version is `s5_gate_report.json`.

## Verdict: **FAIL**

8 FAIL, 1 PASS, 1 MANUAL. A FAIL is final for this study, whatever the mechanism decision. Under DESIGN §4.5 the kill rule would allow up to 3 improvement attempts, but S1 already recommended KILL-EARLY.

**Which gates bind.** None of them bind individually, because the verdict is over-determined.
- The selected configuration (lookback 80, stop_mult 1.0, hold 18) has a **negative in-sample** Sharpe (−0.12). The optimiser's plateau selection is the cause (FINDINGS #38).
- As a result, every performance gate fails by a wide margin: DSR 0.0002 against 0.95, CPCV median −0.68 against 1.0 with 0/9 paths positive, and cost stress −0.92 against 0.5.
- The re-optimisation procedure is also flat: WFO OOS is 0.07, and −0.07 over the recent third.
- The only PASS, positive years at exactly 60%, is marginal and comes alongside a negative total PnL.

## Trial count (DSR)

| | |
|---|---|
| n_trials_study | 840 (in memory = trial store = ledger `trials` event) |
| prior (ledger, `related_prior_trials`) | 0 (no other study shares the system name or issue 23) |
| **N used (raw)** | **840** |
| Hurdle SR0 | 1.07 annualised (V0 = 1/(T−1), T = 2352 days) |
| Selected SR | −0.12 (skew 1.16, kurtosis 12.3); PSR(0) = 0.36 |
| Effective N (diagnostic only) | eigen 7.1 · cluster (ρ ≥ 0.5) 12.0 · Li–Ji 111 |

Red-team M2 asked for the legacy SMA-cross EURUSD-4h grid (625 configurations) to be declared. The card was not amended, so the official run used the ledger count. As a diagnostic, not a gate: N = 1465 gives a hurdle of 1.12 and DSR 1.1e-4, and N = 2715 gives 1.17 and 5.5e-5. The verdict is unchanged.

## Registered holdout pass band (DESIGN §4.4)

This band was registered by this first logged run.
- **Construction.** It is built from the WFO OOS series (1826 d) with a stationary bootstrap: mean block 4.6 d, 2000 draws, seed 583009915.
- **Tail level.** The joint tail level is α = 0.046, and 90.1% of draws pass all four criteria.
- **Horizon.** 262 trading days, from 2025-05-15 to 2026-05-15 10:36 (source: manifest). No data newer than the locked year is available.

| Criterion | Pass if |
|---|---|
| Holdout Sharpe | ≥ −1.75 (median 0.06) |
| Mean monthly return at k = 0.172 | ≥ −0.27% |
| Max DD (unlevered) | ≤ 20.7% |
| Trade count | within [56, 117] (source: wfo_oos.n_trades) |

**P(pass | zero edge) = 0.89. The band is NOT decisive.** An all-criteria pass would be recorded NOT_DECISIVE. This happens by construction: the source series has about zero edge, so the "real-edge" band and the zero-edge band coincide. The system failed S5, so it will not reach checkpoint B anyway.

## §4.5 classification

Monthly return at a 10% DD budget, computed on the selected trial on the full dev window (in-sample):
- The leverage to budget is 0.110. Put the other way, the bootstrapped 95th-percentile max DD at 1% risk per trade is about 91%.
- The mean monthly return at that leverage is **−0.015%**, with a p5/p50/p95 band of −0.079% / −0.015% / +0.048%.
- **Label: unprofitable.**

## 3. Mechanism ablation (evidence only; no review recorded)

**What the card specifies.** Keep the entry timestamps (every ROC zero-cross), the ATR stop, the `hold` exit, the reversal timing and the sizing. Replace each cross's side with a fair coin (seeded) and run 1000 draws. It is a **contradiction** if the real Sharpe is not above the null's 95th percentile (p ≥ 0.05), or if real − median ≤ 0.

**Tooling.** `stats.ablation_compare` is an on/off rule comparison, and `random_entry_null` randomises timing, so neither fits (red-team P6). I implemented the card's null with:
- a wrapper strategy, `probes/s5_randside.py`, which replaces the side of every ±1 cross row with a seeded coin;
- the unchanged `RuleEvaluator` (M1 intrabar, same costs);
- `stats.empirical_pvalue` for the p-value.

`side_seed=-1` reproduces the trial store exactly: max |error| = 0 for trial 492 and for the WFO splice. Script: `probes/s5_ablation.py`; output: `probes/s5_ablation.json`; runtime 156 s on 8 workers.

| Test | Real Sharpe | Null median | Null p95 | Null s.d. | One-sided p | Real − median | Card rule |
|---|---|---|---|---|---|---|---|
| **A. Card as written** (in-sample best-of-840 selected config, full dev, fixed config in the null) | −0.12 | −0.28 | 0.15 | 0.29 | 0.27 | +0.16 | **contradiction** (p ≥ 0.05) |
| **B. OOS version** (red-team M1 option a: WFO OOS splice, 1826 d, 11 refit configs, each run with random sides and spliced as `opt.walk_forward` does) | 0.07 | −0.28 | 0.28 | 0.35 | 0.17 | +0.34 | **contradiction** (p ≥ 0.05) |

**Reading.**
- **Why the null median is negative.** The null median of about −0.28 is the cost drag of the entry schedule: a zero-edge side pays spread about 90 times a year. The real side beats that median by 0.16 to 0.34 Sharpe, but it does not clear the 95th percentile in either test.
- **Test A is biased towards confirmation.** The real Sharpe carries the best-of-840 selection premium (M1). Even so, it does not confirm the mechanism, so the contradiction is robust to that bias.
- **Test B has no selection premium on the real side.** The WFO configurations were chosen on earlier data, and the null draws use those same configurations. It is the fairer comparison, and it also gives a contradiction.
- **Power is low (m8).** The null s.d. is 0.29–0.35, so a true side edge of 0.2 would clear the 95th percentile only about 15% of the time. A contradiction here is weak evidence against the mechanism. It is not evidence for "no momentum".

**Deviation from the card.**
- **Same-side crosses.** The engine cannot close and reopen on a same-side cross. When two consecutive crosses draw the same side, the second is a no-op: the stop is not re-anchored and one round trip of spread is saved. The time-exit clock does restart. This makes the null slightly cheaper, which is conservative against the system.
- **Coins per draw.** In test B, coins are independent per configuration and per draw.

**What an unbiased version is.**
- **Option a (test B).** It is valid for the WFO procedure. It does not test the full-dev selected configuration.
- **Option b (exact, not run).** For each draw, randomise the sides of all 840 grid configurations, re-run the whole selection (plateau selection, or the WFO re-fits on the random-side matrix), and compare the real selected (or WFO OOS) Sharpe with the null's selected (or WFO OOS) Sharpe. At the measured throughput (about 0.012 s of wall time per evaluation on 8 workers), that is about 10 s per draw, or about 2.8 h for 1000 draws. The selection and WFO steps on the null matrix are cheap (`opt.walk_forward`). This version requires the side-randomisation hook to be in the library rather than in a probe.

## 4. Notebook §5

**It cannot run standalone.** I executed §0 and §5 alone on an in-memory copy with nbclient, and it raised `NameError: name 'study' is not defined`. §5 needs the `study` and `evaluator` objects built in §3 and §4.

A full top-to-bottom run would do two things:
- §4 would resume the study, adding 3 ledger rows and rewriting the artifacts.
- §5 would log a **second** gate run (`gate_run` 2).

For that reason I did not execute the full notebook. The ledger is unchanged by the notebook attempt.

## 5. Process findings (one line each)

1. **No read-only loader** (FINDINGS #34 confirmed). A store-based rebuild is about 40 lines and passes every gate guard; `opt.load_study(study_id)` should be added so that S5 and S7 never need `resume=True`.
2. **Probe scripts need `PYTHONPATH=.`** to import `quantlab` when run from outside the repo root. This has the same root cause as #33.
3. **Notebook §5 is not standalone** and has no guard against re-logging. `evaluate_gates(log=True)` only adds a warning banner on run 2; my script refuses to run if a `gates` event already exists, and the library or template should do the same.
4. **The API contradicts the process on the mechanism gate.** The agent file says to call `evaluate_gates(mechanism_check=…)`, which turns the mechanism gate PASS or FAIL automatically, but DESIGN §4.2 says the user decides via `record_mechanism_review`. It is unclear whether the statistician should pass it.
5. **No library helper for a random-side null at fixed timestamps** (P6 confirmed). I needed a wrapper strategy plus a custom WFO splice.
6. **The engine cannot express close-and-reopen on a same-side signal**, so the card's ablation cannot be implemented exactly.
7. **The plateau gate cannot pass when the peak Sharpe is ≤ 0.** This is stated in the `judge_plateau` docstring, not in DESIGN. Points with Sharpe 0.27 or 0.43 fail. The row text still says "sharp, fragile optimum", and "weakest axis lookback" is an arbitrary tie-break when every axis scores 0.
8. **The cost-stress swap band is vacuous at swap = 0** (uncalibrated spec: all three multipliers give −0.92). The gate text does not flag it.
9. **The cost-stress row's "median dev spread 32 points" uses D1 bars**, whose spread is the 00:00 rollover spike, so it overstates the typical EURUSD spread about 2.6× (S3 charged 12.4 points per trade). This is display only.
10. **`max_year_share` shows "inf"** in the table (null in the ledger) when total PnL ≤ 0. `positive_years` can PASS on a losing system. `trade_count` shows "need inf" when the claimed Sharpe is < 0. All three are correct but uninformative, and S1's MinTRL at the card's expected SR (67.6 y) would be more useful.
11. **A holdout band is registered even when the verdict is FAIL**, and a band built on a zero-edge WFO series is non-decisive by construction (P(pass | 0) = 0.89).
12. **`horizon_end` of 2026-05-15 10:36 is later than `locked_end` of 00:00**, yet `newer_data_included` = no. It is a minor inconsistency.
13. **The DSR has no rule for red-team-requested legacy trials** when the card is not amended (M2). The statistician can only raise N via `prior_trials=` on their own judgement. I used the ledger count and reported the sensitivity.
14. **Naming mismatch:** the agent file names the output `validation_<study_id>.md`, but this task asked for `validation_S5.md`.
15. **`GateReport` has no `to_json` / `as_dict`**, so a custom serialiser was needed for the machine-readable dump.
16. **The uncalibrated-broker UserWarning repeats** once per spawn worker in the gate run and the ablation (#37 again).
17. **The card's ablation rule (b)** (real − median ≤ 0) is redundant given (a) (m8). Its power at the card's SR is about 15%, which should be stated on the card template.

## Files

- `results/probes/s5_load_study.py`: the read-only StudyResult loader.
- `results/probes/s5_precheck.py`: the guard pre-check.
- `results/probes/s5_gates.py`: the single logged gate run.
- `results/probes/s5_gate_report.md` and `results/probes/s5_gate_report.json`: the full GateReport and its dump.
- `results/probes/s5_randside.py` and `results/probes/s5_ablation.py`: the mechanism ablation.
- `results/probes/s5_ablation.json`: the ablation output.
