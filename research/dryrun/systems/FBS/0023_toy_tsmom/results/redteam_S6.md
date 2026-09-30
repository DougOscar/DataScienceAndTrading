**DRY RUN — toy system** (DESIGN §10 Phase 2 exit test; sandbox `research/dryrun/`). None of the numbers below is evidence about H4 TSMOM.

# Red team — S6, `toy_tsmom`, study `fbs-0023-a1` (issue #23)

Red team, 2026-09-30. S5 verdict under attack: **FAIL** (8 FAIL, 1 PASS, 1 MANUAL; `validation_S5.md`). At S6 on a failed system, the question was reversed: **could the FAIL be an artefact** of a bug, a cost error, a data issue or a pipeline mistake?

**Result: the kill stands.** It holds with costs switched off, under a raw-argmax selection procedure, with and without the M1 intrabar path, and at the best configuration on the net surface. No bug, look-ahead or data defect was found that could have hidden an edge on EURUSD H4.

**Severity counts: BLOCKER 0 · MAJOR 0 · MINOR 5 · process 7.**

All probes are in `results/probes/s6_*.py`. They were run with `research/dryrun/env.sh` sourced. The ledger is unchanged: md5 `fdd0d2f4…` before and after, still 4 rows and one `gates` event. The only run that wrote ledger rows (the resume in the faithfulness probe) ran on a scratch copy of the ledger and trial store.

## Diagnostic evaluation count (for the record; not trials of this study)

| Probe | Evaluator calls | Of which new information |
|---|---|---|
| `s6_faithfulness.py`: `run_study(resume=True)` on the scratch copy (0 trials re-run) plus 2 × `evaluate_gates(log=False)` (plateau 21 + cost stress 1 each) | 44 | 0. These repeat the S5 judge points on identical inputs. |
| `s6_grid_diag.py`: EURUSD 840-grid, net | 840 | 0. Bit-identical to the trial store (max\|diff\| = 0). |
| `s6_grid_diag.py`: EURUSD 840-grid, gross (spread ×0) | 840 | 840 |
| `s6_grid_diag.py`: GBPUSD, USDJPY, AUDUSD 840-grid, net and gross | 5,040 | 5,040 |
| `s6_exec_diag.py` and its inline stop probe: trial 492 repeats, 492 with `use_m1=False`, USDJPY at 492's params | 4 | 1 (no-M1) |
| `s6_selection.py`: (90, 1.0, 18) and (70, 1.0, 12), base and stressed, plus a 492 repeat | 5 | 2 (the stressed runs) |
| `s6_lookahead.py`: truncation audits at 5 configurations | not performance evaluations | none |
| **Total** | **6,773** | **5,883** (843 on EURUSD, 5,040 on other symbols) |

If anyone proposes a multi-symbol TSMOM follow-up, the 5,040 other-symbol evaluations have to enter its DSR N as prior trials (see m4).

---

## Could the FAIL be an artefact? Checks run

### A. Costs off: the kill holds without any cost at all
`s6_grid_diag.py` + `s6_analyse.py`, EURUSD, full 840-grid with the spread multiplied by 0 (swap and slippage were already 0).

| Surface / procedure | Net (study) | Gross (costs off) | Gate threshold |
|---|---|---|---|
| Median configuration Sharpe | −0.29 | −0.07 | — |
| Share of configurations with Sharpe > 0 | 11 % | 39 % | — |
| Best of 840 | +0.45 (90, 1.0, 18) | +0.68 (90, 1.0, 18) | DSR hurdle 1.07 |
| Selected configuration (80, 1.0, 18) | −0.12 | +0.11 | — |
| CPCV median path Sharpe (study's own procedure replayed) | −0.68 (0/9 paths > 0) | −0.50 (0/9 paths > 0) | ≥ 1.0 |
| WFO OOS Sharpe / recent third | +0.07 / −0.07 | +0.26 / +0.19 | ≥ 0.5 and > 0 |

With zero costs, the best in-sample configuration is still below the DSR hurdle. The CPCV and WFO results for the procedure still fail by a wide margin. The costs are not what killed an edge: there is no gross edge large enough to pass.

### B. Cost model: the uncalibrated `fbs-v2-uncalibrated` model does not overstate costs
- **Spread.** It comes from the bar data and is not invented. EURUSD M1 spreads have a median of 8 points in every hour except 00:xx (27) and 23:xx (10). See m1 for the rollover hour.
- **Swap.** Charged as 0, which is optimistic. Trial 492 holds 312 short nights and 286 long nights, so its exposure is balanced. Any realistic FBS schedule, with its markup on both sides, would be net negative. See m3.
- **Slippage.** Base stop slippage is 0, which is optimistic.
- **Conclusion.** Every uncalibrated item errs in the system's favour.

### C. M1 intrabar path and short stops on the Ask
- **M1 on or off.** Trial 492 gives Sharpe −0.118 with M1 and −0.124 with `use_m1=False`, over the same 670 trades. The intrabar path does not move the result.
- **Stops that only the spread triggered.** 55 of the 136 short stops hit the stop on the Ask but not on the Bid in the triggering minute. The spread in those minutes was normal: median 8 points, p90 10, max 16. The Bid fell short of the stop by a median of 3.5 points (max 12), against a median stop distance of 290 points.
- **Would those stops have fired anyway?** In 53 of the 55 cases, the Bid itself reached the stop within the next 18 H4 bars.
- **Conclusion.** This is correct Ask-side execution, not a spike artefact. None of those triggers fell in the 00:00 or 01:00 hour.

### D. H4 resampling and data
- **Cross-check.** H4 bars resampled from M1 (`closed="left", label="left"`) were compared with an independent FBS H1 export resampled to H4, over 13,447 common bars. The median OHLC difference is 0 points and the p99 is 0 points. The two close-to-close return series have correlation 1.000.
- **Bar times.** Bars fall at 00/04/08/12/16/20 EET, and there are 6 stray Sunday 20:00 bars (5 in May–June 2016 and 1 on 2024-10-27, with tick volume of 68–830). Their impact is negligible.
- **Gaps.** Gaps longer than 3 days occur only over Christmas and New Year. The ledger records `data_gaps` = [] and `m_trades_across_data_gap` = 0.

### E. Look-ahead
`s6_lookahead.py` ran the following on the last 3,000 real dev H4 bars:
- `lint_strategy_source`, which came back clean;
- `assert_no_lookahead` (factory, 40 cuts);
- `assert_engine_causal` with the M1 path and `timeframe="H4"`.

It covered five configurations: the S5 selection (80, 1.0, 18), the raw argmax (70, 1.0, 12), the best net and gross configuration (90, 1.0, 18), and the grid corners (120, 4.0, 60) and (10, 1.0, 6). All were clean. S2 had already covered the other corners.

### F. Does the selected configuration represent the surface fairly?
- **Where trial 492 sits.** Its net Sharpe of −0.12 is at the **75th percentile** of the 840 configurations. The plateau pick is not a pessimistic outlier.
- **Replaying a raw-argmax procedure** (`PlateauConfig(min_centre_quantile=1.0)`) on the stored matrix gives a CPCV median of −0.19 and a WFO OOS of −0.27 (recent third −0.45). That is worse on WFO and still far below the CPCV threshold of 1.0. The plateau procedure did not throw away an edge the argmax would have kept.
- **Best net configurations under the cost-stress gate.**

| Configuration | Base Sharpe | Stressed Sharpe | Gate |
|---|---|---|---|
| Best net, (90, 1.0, 18) | +0.45 | **−0.34** | > 0.5 |
| Raw argmax, (70, 1.0, 12) | +0.31 | **−0.51** | > 0.5 |

  No pick from this surface passes DSR, cost stress or MinTRL.
- **CPCV check.** I recomputed three of the CPCV split OOS Sharpes independently: splits (0,2), (1,4) and (2,9) gave −1.0233, +0.5235 and −1.8068, identical to the library. The CPCV median of −0.68, lower than the full-sample −0.12, is the expected negative train/test coupling on a zero-edge surface (PBO slope −0.74). It is not a bug.

### G. Other symbols (sanity only; not new trials of this study)
Same 840-grid and code, dev window, FBS spread data, swap 0:

| Symbol | Net median | Net share > 0 | Net best | Gross median | Gross best |
|---|---|---|---|---|---|
| EURUSD | −0.29 | 11 % | 0.45 | −0.07 | 0.68 |
| GBPUSD | −0.04 | 44 % | 0.59 | +0.17 | 0.91 |
| AUDUSD | −0.34 | 9 % | 0.26 | +0.05 | 0.83 |
| USDJPY | **+0.19** | **70 %** | 0.82 | +0.49 | 1.20 |

- **EURUSD is not an odd failure.** On three of the four symbols the net surface is flat to negative.
- **USDJPY is the exception.** See m4.
- **No symbol's net best reaches the single-symbol DSR hurdle of 1.07.**

---

## MINOR

### m1. The H4 fill spread is the first M1 minute's, and FBS widens the whole 00:xx hour, so 41.5 % of the spread the system pays comes from the rollover hour
- **Mechanism.** `data._resample_full` sets `spread = spread.first()`, so an entry at the 00:00 H4 open pays the 00:00 M1 spread.
- **Size of the effect.** For trial 492:
  - 128 of the 670 entries fall on the 00:00 bar, at a mean of 22.6 points against 8–11 for the other hours.
  - The 102 legs that pay spread at 00:00 (long entries and short exits) carry **41.5 % of all spread cost**: 31.5 points each, against 8.0 for the rest.
- **The data behind it.** The M1 spread median is 26–30 points in every 10-minute block of 00:xx, falls to 9–12 in 01:xx, and has risen from 10 (2016) to 49–54 (2021–22) and 31 (2025). This is how FBS actually quotes, so the charge is realistic for a fill at 00:00.
- **What a live implementation would do.** It would defer those fills to about 01:00.
- **Why the verdict is unchanged.** The upper bound on any spread fix is the costs-off result: selected configuration +0.11, WFO +0.26, CPCV −0.50. A linear estimate of removing the rollover excess puts trial 492 at about −0.03.
- **Library point.** A `defer_rollover_fills` execution option (or a card-level rule) is worth adding for every H4/D1 system.

### m2. The "contradiction" label on the mechanism ablation reads stronger than the evidence
- **The numbers.** The real-minus-null-median side contribution is +0.16 (test A) and +0.34 (test B). Those values are close to the card's own expected Sharpe of 0.2, and they match the gross surface: the selected configuration's gross Sharpe is +0.11.
- **Power.** It is about 15 % at SR 0.2.
- **Reading.** The test cannot distinguish "no side edge" from "the card's claimed edge".
- **What S5 got right and what should change.** S5 said so ("weak evidence against the mechanism"). The table still prints **contradiction**, because that is what the pre-registered rule says. For the MANUAL decision, it should be read as **inconclusive**. This does not affect the verdict, which is over-determined by 8 automatic FAILs.
- **Fairness of the ablation.** It was fair otherwise:
  - with `side_seed=-1` it reproduces the trial store exactly;
  - the deviation on same-side crosses (the null pays about one less spread) is disclosed and conservative against the system;
  - test B has no selection premium.

### m3. Swap = 0 and base slippage = 0 flatter every net figure (carried from S2 code audit M1; downgraded here)
- **Impact at S6.** On a failed system this only strengthens the kill. Trial 492's exposure is balanced (312 short nights, 286 long), so a markup on both sides makes real swap net negative.
- **Why it is kept as a finding.** The cost-stress row's swap band (×0.5 / 1 / 1.5 all give −0.92) is vacuous at swap 0, as S5 #8 noted. Any future variant that clears S3 needs calibrated swap first.

### m4. Scope of the kill: it applies to EURUSD, not to "H4 TSMOM"
- **What the other symbols show.** On USDJPY the same grid is broadly positive net: median +0.19, 70 % of configurations positive, best 0.82.
- **Why it is not a clean edge.**
  - The profit is concentrated in 2022 and 2024, the BoJ-divergence trend years: +15.7k and +8.0k of the +24.7k total at 492's parameters.
  - It is earned on both sides.
  - It is ex-carry, because swap is 0, and USDJPY carry was large in exactly those years.
- **What it does not justify.** It does not reopen this study. The card pre-registered EURUSD only.
- **What it does require.** Any follow-up, whether a USDJPY card or a multi-symbol TSMOM card, is a **new hypothesis**. Its DSR N must include these 5,040 diagnostic evaluations plus the 840 EURUSD trials, and it must treat the USDJPY result as partly data-mined.
- **Wording.** The S7 report should phrase the kill as "no edge on EURUSD H4 under this rule set".

### m5. Stray Sunday 20:00 H4 bars
- **What they are.** 6 bars: 5 in May–June 2016 and 1 on 2024-10-27, with tick volume of 68–830. They are server pre-open quotes.
- **Impact.** They add a signal opportunity outside the trading week. The impact on 14,073 bars is negligible.
- **Suggested fix.** The resampler or catalog should flag or drop them for FX (`data.py` already handles FX Sunday closure for DST, so this is an edge case).

---

## What I checked and found nothing

- **Faithfulness of the S5 read-only rebuild** (`s6_faithfulness.py`). I compared the `s5_load_study` object with the library's own `run_study(resume=True)` rebuild on a scratch copy of the ledger and store:
  - selection, trials frame (all columns), returns matrix (max\|diff\| 0, no null cells), `cpcv_paths`, `wfo_oos` and `wfo_params` were all identical;
  - all ten gate values and statuses from `evaluate_gates(log=False)` were identical to the logged run.

  The loader leaves out 28 `meta` keys, but none of them is read by `gates` or by `report.tear_sheet`, which read the ledger. **The rebuild is faithful.**
- **One gate run.** The ledger has exactly one `gates` event (seq 3, `gate_run` 1), with no `n_trials_changed` or `resumed` rows. The trial store has 840 trials, equal to the ledger count, and `method=grid`, which is data-independent. `embargo_capped` is false, and no gate was SKIPPED.
- **Reproduction.** The net EURUSD grid re-run is bit-identical to the trial store for all 840 configurations. The M1 consistency guard, which requires H4 high/low to equal the M1 max/min, is active in every run.
- **Look-ahead and engine causality** at 5 configurations, with the M1 path (§E).
- **H4 resampling** against an independent H1 export (§D).
- **Spread-triggered short stops** at normal spreads, with no rollover spike stops (§C).
- **Annualisation.** There are 2,352 daily rows over 9.04 years, which is 260 per year: weekdays only, consistent with `periods_per_year` = 260.
- **Selection fairness** and the raw-argmax procedure (§F).

---

## Process findings

1. **"The gate ran once" can only be verified for logged runs.**
   - **What was verified.** S5's single `gates` event is genuine, and the script refuses to run a second time.
   - **The gap.** `evaluate_gates(log=False)` leaves no trace, so previews with gate outcomes are invisible. I ran two of them myself (on a scratch ledger) and they appear nowhere.
   - **Fix.** Log a `gates_preview` event, or refuse `log=False` on a study id that is in the real or sandbox ledger.
2. **There is nowhere to record diagnostic evaluations.**
   - **Scale.** S5's ablation made about 12,000 random-side evaluations, and S6 made 6,773, including 5,040 on symbols the card never registered.
   - **The gap.** None of these are in the ledger. The other-symbol grids are exactly the kind of look that inflates a later study's true N.
   - **Fix.** Add a `diagnostic` ledger event with its symbols, config count and purpose, so that `related_prior_trials` can see it.
3. **Scripts that call `opt.run_study` or `evaluate_gates(n_jobs>1)` need an `if __name__ == "__main__":` guard.** The pool uses spawn. My first faithfulness run died with `BrokenProcessPool`. The probe template or docs should say so; S5's scripts had the guard.
4. **There is no `CostModel.gross()` preset.** `CostModel(spread_multiplier=0.0)` works (version `…+spread_mult0`). A named preset and a gross-vs-net helper would make "does the kill hold with costs off" a one-liner at every S6 (see also FINDINGS #25).
5. **The S5 loader is faithful but partial.** It rebuilds 17 of 45 `meta` keys. Before S7 relies on it, it should be promoted into the library as `opt.load_study` (FINDINGS #34), with a test asserting equality with `run_study(resume=True)`, as `s6_faithfulness.py` does.
6. **The card template has no rule for scoping the kill.** Nothing says whether a FAIL on the registered symbol is reported as a symbol-level or a mechanism-level kill (m4). DESIGN §4.5 should require stating the scope.
7. **Ablation verdicts should have a power-aware third outcome.** When the pre-registered null has power below about 50 % at the card's expected SR, the verdict should be "inconclusive" rather than "contradiction" or "confirmed" (m2; S5 #17).

## Files
- `results/probes/s6_faithfulness.py`: loader vs `run_study(resume=True)` on a scratch copy.
- `results/probes/s6_grid_diag.py`: the 840-grid under a cost variant or on another symbol. Output matrices were written to the session scratchpad, not to the repo.
- `results/probes/s6_analyse.py`: surface statistics, and the procedure replay on the gross matrix.
- `results/probes/s6_exec_diag.py`: rollover spread, spread-triggered short stops, M1 vs no-M1, side and year splits.
- `results/probes/s6_selection.py`: selection percentile, raw-argmax replay, cost stress of the best configurations, nights by side.
- `results/probes/s6_lookahead.py`: lint, look-ahead and engine-causality audits.
