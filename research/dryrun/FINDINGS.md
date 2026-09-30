# Phase 2 dry run — process findings (issue #23, toy_tsmom)

Pipeline gaps found while running the toy system through /research-cycle. Fixed at the end in one batch.

## S0 (quant-scout, card template)
1. No field for the grid step (put in the range column as "grid step N").
2. Integer params under a relative scale: the judge perturbs off-grid and snaps to the next distinct int (`gates.plateau_perturbations`) — undocumented on the card template.
3. The 5 %-of-declared-range floor overrides r/2 at small values (intended, R3-3) — undocumented on the card template.
4. "Expected trades/year" does not say entries vs round trips, nor exposure.
5. "Null / benchmark" vs "Mechanism ablation" overlap for a full system; neither says whether stops stay on.
6. Expected Sharpe not tied to a parameter point (prior mid-grid vs optimised).
7. (content) exact page/section citation not verified — scout rule; fine for a toy.

## S1 (validation-statistician, power check)
8. `stats.power_check`'s `feasible` = ~50 % power for a single trial; it ignores the DSR trial count (a 840-trial grid needs SR ≥ 1.07 just to clear the hurdle). S1 should report the DSR-aware requirement too.
9. `trades_per_year` never enters the verdict — contradicts the "redesign raising trade count" wording (DESIGN §3 / agent file).
10. `years_available` is typed by hand; should come from the data manifest for the card's symbols/timeframe.
11. The card's expected-Sharpe range is not tied to parameter points (same as #6).

## S2 (system folder)
12. BUG (fixed): `systems.create_system` could not parse "type **A**" (markdown emphasis) — the scout's own card failed. Parser now strips `*`/`` ` ``; regression test added.

## S2 (red-team card review; 0 BLOCKER, 5 MAJOR, 9 MINOR on the toy card — see results/redteam_S2_card.md)
Card-content majors that a real cycle would send back to checkpoint A: ablation compares a best-of-840 in-sample Sharpe with fixed-config nulls (selection ≈ SR 1.0 → biased to "confirmed"); legacy SMA/EURUSD-H4 walk-forward (625 configs) not counted as prior trials; swap = 0 (no broker export); `hold` mostly non-binding; null mis-specified for a full system.
13. P1: no public API to preview the judge's plateau moves (red team called private `gates._axis_ladder`).
14. P2: DESIGN says "a move is never smaller than 5 % of the declared range"; code floors the OUTER move at 5 %, the inner (r/2) at 2.5 %. Fix the DESIGN wording.
15. P3: S2 "code carries the card's scales" check can't run before the code exists — split S2 review into card review (before code) and code/space review (after).
16. P4: card has no field for cost-model calibration / swap status; the uncalibrated fallback silently charges swap 0.
17. P5: no field / counting rule for legacy work already done on the same data (legacy notebooks' grids) → DSR N understated.
18. P6: no library helper for side-randomisation at fixed timestamps (entry-signal null); `stats.ablation_compare` is on/off only.
19. P7: template doesn't say whether the ablation runs in- or out-of-sample (should be OOS / re-selection inside each null draw).
20. P8: "Null / benchmark for component tests" wording invites a component null for a full system (same as #5).

## S2 (strategy-engineer, implementation)
21. `quantlab.strategies.base` only had a plain-rolling-mean `atr()` (= MT5 `iATR`); the card's "Wilder ATR" (RMA, seed-then-recurse — NOT MT5 `iATR`, corrected after the code audit) is a materially different number past the first window. Added `base.wilder_atr` with tests (dry run #23 needed it; likely reusable).
22. Card gives the mid-grid default for only one of three parameters ("lookback 60"); `hold`'s own mid-grid point is never stated. Resolved by applying the same rule implied by that one example (the lower of the two middle grid levels when the level count is even) to `hold` too — worth making that rule explicit on the card template so it isn't re-derived per system.
23. "ATR(14)" is a fixed constant the card names in prose, not in the free-parameters table; there's no card field distinguishing a free parameter from a named-but-fixed one, so nothing cross-checks the strategy module's hardcoded period against the card automatically.

## S3 (baseline)
24. The template's `baseline_code` cell ends with `baseline.metrics` as the last statement of an `if` branch, not the cell's own last top-level statement — IPython's auto-display never fires, so a headless nbconvert run shows the evaluator's `describe()` but never the actual baseline metrics in the notebook's saved output. Worked around by reading `outcome.metrics`/`outcome.trades` in a separate script; the template should `print(baseline.metrics)` explicitly.
25. No library helper computes "gross vs net edge per trade (points)" or "cost as % of gross" (DESIGN §3 S3's own framing) from a `RuleEvaluator` `Outcome` — hand-derived from `trades['pnl_points'] + trades['spread_cost_points']` (gross) and `+ trades['swap_points']` (net) per trade. A reusable `evaluators`/`metrics` helper would save every future S3 baseline the same derivation.
26. "Cost as % of gross" is undefined/misleading when the gross edge itself is negative (toy_tsmom's case here): the ratio's sign flips and stops meaning "share of profit eaten by costs." Neither the card nor DESIGN §3 says how to report that case.

## S2 (red-team code audit; 0 BLOCKER, 1 MAJOR (swap = 0, uncalibrated), 7 MINOR — results/redteam_S2_code.md)
27. Docstring claimed Wilder ATR = MT5 `iATR` (it is the SMA `atr()`); fixed — would have broken D1 parity. Need an indicator-definition table (quantlab helper → MQL5 call / hand-rolled) for ports.
28. S3 baseline evaluations leave no ledger/audit trail (ad-hoc `RuleEvaluator` calls); S3 should log a `baseline` event (params, metrics, cost version).
29. S3 kill compares point estimates without uncertainty (gross −18.4 ± 22.9 pts/trade, t ≈ −0.8); a true 2-pip edge would be killed ~37 % of the time. S3 needs a one-sided test or a rule stated with its error rate.
30. Strategy contract can't see position state; card rules that depend on "was a stop hit" can't be implemented exactly (time-exit clock restart on same-direction cross). Card template should warn authors.
31. Look-ahead audits in tests/notebook run only at the prior params; should cover grid corners.
32. S2 "SearchSpace carries the card's scales" check moves to S4 (space not built until then) — make it an explicit S4 hand-off check.

## S4 (optimization-architect; study fbs-0023-a1, 840 trials, 19.6 s)
33. Notebook kernel can't import `quantlab` under nbconvert/nbclient (kernel cwd = system folder, package not installed) — needs PYTHONPATH or an editable install (`pip install -e .`).
34. No read-only `opt.load_study(study_id)` → StudyResult; only `run_study(resume=True)` rebuilds one, and it appends ledger rows. S5/S7 need a read-only loader.
35. Template §4→§5 assume one kernel session; a stage-by-stage run executes cells selectively, and a plain full run would fire S5 gates. Stage cells should be independently re-runnable (load the study by id).
36. Cosmetic: `cv_scheme` logs "purge=autod,embargo=autod".
37. Each spawn worker repeats the uncalibrated-broker UserWarning (8× in notebook output).
38. Plateau centre guard (own objective ≥ median) is weak on a mostly negative surface (picked centre objective −0.33 vs raw argmax +0.13). Worth a design look (e.g. require own objective ≥ smoothed, or ≥ the eligible-set upper quartile).
39. Optimizer plateau score is 0 whenever the centre's Sharpe ≤ 0 — no signal on a losing surface; say so in the report.
40. No card field / rule for `min_trades` (derived 200 from ~50 trades/param/fold) — compute it automatically from the card.
41. WFO's last refit can have a ~2-week test window; no minimum test length per refit.
42. No explicit "S3 kill → skip S4" gate in the pipeline/skill.
43. BUG (fixed): a test draft had leaked trial files into the real research/studies/ (fbs-9999-a1); removed, and tests/conftest.py now fails the session if any test touches research/ledger or research/studies.

## S5 (validation-statistician; FAIL 8/9 automatic gates, mechanism MANUAL; one logged gate run)
44. No read-only study loader (same as #34): the statistician rebuilt the StudyResult from the trial store in ~40 lines; it passed verify_trial_store / evaluator identity. Promote that to `opt.load_study`.
45. Nothing stops a second logged gate run (only a warning banner on re-run) — a notebook re-run would log gate_run 2.
46. API/process mismatch on the mechanism gate: `evaluate_gates(mechanism_check=)` sets PASS/FAIL automatically, while DESIGN has the user decide via `ledger.record_mechanism_review`. Pick one path.
47. No library null that randomises the side at fixed timestamps (confirmed red-team P6); the unbiased ablation (re-select inside each null draw) needs it plus ~3 h compute — needs a design.
48. Engine can't close+reopen on a same-side signal (limits nulls and some rules).
49. Plateau gate always FAILs when the peak Sharpe ≤ 0 (only in a docstring, not DESIGN); the row text "sharp, fragile optimum" is misleading on a losing peak.
50. Cost-stress swap band is a no-op when swap = 0 and the gate text doesn't say so.
51. Cost-stress text's "median spread" uses D1 bars (rollover spike, ~2.6× typical) — display only.
52. `inf`/"need inf" values and a positive-years PASS on a losing system are correct but unreadable; the report should translate them.
53. A holdout band is registered even when the verdict is FAIL (harmless, noisy).
54. Band `horizon_end` 2026-05-15 10:36 > `locked_end` 00:00 yet "no newer data" — minor inconsistency.
55. No rule for counting legacy trials the red team asks for when the card is not amended.
56. `GateReport` has no JSON export.
57. Output naming: agent file says `validation_<study_id>.md`.
58. Improvement loop skipped (dry run): the skill has no explicit "S1/S3 kill → skip to card" short-circuit (see #42).

## S7 (report-builder; tear sheet + killed card OK, 9 figures readable)
59. BUG: plateau heatmap colours sharpe/peak with vmin=0 — on a losing peak (−0.12) worse cells get the largest ratio (darkest "best") and the best point clips to white. Plot raw Sharpe with a diverging scale centred at 0.
60. `cost_pct_of_gross` explodes when gross is near zero (1808 %); report "n/a (gross ≈ 0)" below a threshold.
61. Signed-zero "-0%" for swap share when swap = 0.
62. nan/None render as literals in the Metrics table (Gates table uses "—").
63. metrics.json contains NaN tokens (invalid JSON) — write null.
64. `trade_count` and `mechanism` gates missing from the applicability table and interpretations (DESIGN §5 "one line per metric").
65. Several metrics have blank interpretation cells (skew, kurtosis, n_trades, verdict, expectancy, hold p95, wfo recent).
66. BUG: write_card writes 7 of 9 figures into the shared vault attachments folder under generic names — a second system overwrites them. Prefix every attachment with the slug (or only copy the two §6 figures).
67. Headline rounds −0.0148 %/month to "−0.0 %/month" — use 2 significant figures.
68. Notebook §7 cannot run standalone (needs `study` from §4/§5) — same root cause as #34/#35.

## S6 (red team: is the FAIL an artefact? No — kill stands; 0 BLOCKER, 0 MAJOR, 5 MINOR)
69. No ledger event for diagnostic evaluations (S6 ran 5,883 new-information evaluations: cost-free grid, 3 other symbols) — they must count for any follow-up; add `ledger.log_diagnostic(study_id, n_evaluations, scope, note)` and count them in related trials.
70. `evaluate_gates(log=False)` previews leave no trace — log preview runs as diagnostics.
71. No costs-off preset (`CostModel.frictionless()`), needed for every S6 "is it the costs?" check.
72. No power-aware "inconclusive" outcome for ablations (≈15 % power here; "contradicts" overstated).
73. 41.5 % of spread cost comes from 00:00 fills (FBS rollover-hour spread; H4 bar takes the first minute's spread) — note for the cost model / card cost section.
74. Kill scope should be recorded per symbol (USDJPY looked different) — card / ledger has no "scope of kill" field.

## Exit test (DESIGN §10 Phase 2): MET — the toy ran S0→S7 end to end (card → issue → power check → folder → strategy + look-ahead audits → baseline → study → gates → attack → tear sheet → killed card), every hand-off verified, sandbox isolation held (real research/ledger and research/studies untouched; a leak from an earlier test draft was found and guarded). 74 process findings; 4 real bugs found and fixed during the run (card parser, test leak, ATR docstring/parity, and 2 report bugs queued: #59, #66).
