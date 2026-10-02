**DRY RUN — toy system** (DESIGN §10 Phase 2 exit test; sandbox `research/dryrun/`)

# Red team — S2 code audit, `toy_tsmom` (issue #23)

Red team, 2026-09-30. Audited `quantlab/strategies/toy_tsmom.py`, `quantlab/strategies/base.py` (`wilder_atr`),
`tests/test_strategy_toy_tsmom.py`, `tests/test_strategies.py` (Wilder tests), the executed `toy_tsmom.ipynb`,
the engine's signal/stop path (`quantlab/engine.py::_run_core`) and `RuleEvaluator.__call__`. Checked against
`hypothesis.md` and `results/redteam_S2_card.md`.

Probes (all in `results/probes/`, run with `research/dryrun/env.sh` sourced):
- `s2code_audits.py`: `lint_strategy_source` on both modules. `assert_no_lookahead` (factory, exhaustive forced cuts, 40 random cuts) and `assert_engine_causal` at 5 grid points: all 4 corners of (lookback, hold) × stop_mult {1.0, 4.0}, plus the prior. Run on 1,500 synthetic H4 bars, and `assert_no_lookahead` on the last 2,500 real dev EURUSD H4 bars at (10, 1.0, 6) and (60, 2.5, 30).
- `s2code_rule_fidelity.py`: an independent loop implementation of the card rules. It is compared row by row with the module's signals on the full dev EURUSD H4 series (14,073 bars) at 15 (lookback, hold) points, with stops ignored. It also includes a Wilder-ATR reference loop and a comparison with an SMA-ATR.
- `s2code_baseline.py`: re-runs the S3 baseline (the same prior config, so not a new trial) and breaks it down by side, exit reason, year, entry hour and holding time.

**Severity counts: BLOCKER 0 · MAJOR 1 · MINOR 7 · process 7.**

---

## MAJOR

### M1 (carried forward from card review M3, now confirmed in the output). Swap is charged as 0, so every net figure at S3 and later understates cost
`s2code_baseline.py`: `trades["swap_points"].mean() == 0.0` over all 780 trades, and the cost version is `fbs-v2-uncalibrated` (evaluator warning: "no calibrated broker export for FBS/EURUSD"). Base `CostModel` stop slippage is also 0.
- **What the engine charges.** The spread only, 12.4 pts per trade on average (1.24 pips; 25.6 pts on the 112 entries at the 00:00 rollover).
- **What the card budgets.** Swap of about 1–5 pips per trade on the paying side, so net cost of 1.5–4 pips.
- **Failure scenario.** At S4/S5, the net Sharpe, the break-even spread multiple and the cost-stress gate are all computed on a cost base that may be 2–4× too low. This does not change the S3 verdict: gross is already below spread-only cost, so the kill holds even more strongly. It would matter for any variant that clears S3.
- **Fix.** Calibrate the spec, or pre-register a swap sensitivity band before S4, as card review M3 already asked.

---

## MINOR

### m1. The time-exit clock restarts on a same-direction cross while already in a position. The card says only a reversal or a new entry starts a new clock
`toy_tsmom.py`: `episode_start = cross_row.forward_fill()` moves on **every** cross row, including a `+1` cross while the engine is already long. That cross arises from a `+, 0, +` ROC sequence, which card review m6 said the card had not specified. The engine no-ops the entry (`signal == position`), but the scheduled `0` moves to `new_cross + hold`, so the position is held longer than `hold` bars.
- **Evidence.** The loop reference differs from the module on 0–4 rows out of 14,073 per configuration, and there are at most 6 same-direction consecutive crosses per configuration. Real ROC == 0 bars number 2–9 per lookback over 9 years.
- **Impact.** Negligible on the numbers.
- **Root cause.** The rule cannot be implemented exactly under the contract. The signal frame cannot see whether a stop fired between the two crosses, and the card treats "same-direction cross after a stop" (new entry, new clock) differently from "same-direction cross while in position" (no new clock). See process P3.
- **Tests.** No test covers either case. Add one that pins the chosen behaviour (restart) and document it on the card.

### m2. `wilder_atr` is correct, but its docstring (and FINDINGS #21) wrongly calls it "MT5's `iATR` definition"
- **Correctness.** The implementation matches a reference Wilder loop to 5e-18 at every non-null row. The seed is the simple mean of TR[0..13], with TR[0] = H−L, at row 13. There are no NaNs, and exactly 13 null warm-up rows. It is causal (probes below). It matches the card's "Wilder ATR".
- **The error.** MT5's standard `ATR.mq5` / `iATR` is a **simple moving average** of TR, not Wilder's recursion. On dev EURUSD H4, Wilder and SMA-ATR differ by 5.2 % (median), 16 % (p95) and 55 % (max) in stop distance.
- **Failure scenario.** At D1, a port that trusts the docstring and calls `iATR` gets different stop distances and lot sizes, and the parity test fails or gets waved through.
- **Fix.** The docstring should say "Wilder's recursion; note that MT5's built-in `iATR` is an SMA of TR", and the MQL5 port must hand-roll the recursion rather than call `iATR`.
- **Seed dependence.** The seed depends on the frame's first row. This is harmless here: the evaluator slices with 500 warm-up bars, and (13/14)^500 ≈ 1e-16.

### m3. The mid-grid defaults are consistent with the card, but `hold = 30` rests on a convention the engineer chose
- **lookback 60.** The card states it: the lower of the middle pair 60/70.
- **stop_mult 2.5.** The unique middle of 7 levels.
- **hold 30.** The lower of the middle pair 30/36, by analogy with lookback. The card never states it (FINDINGS #22).
- **ATR(14).** Hard-coded as `_ATR_PERIOD = 14`, which matches the card's prose, and it is correctly not a free parameter.
- **Why it matters.** The S3 verdict is taken at these defaults, and nothing binds the defaults to the card mechanically (FINDINGS #23). Given the S3 noise level (next item), the choice between 30 and 36 is unlikely to change the verdict. The rule should still be on the card before S3 runs.

### m4. The S3 baseline reproduces exactly and is plausible. It also contradicts several card assumptions (informational)
Prior (60, 2.5, 30) on dev EURUSD H4, M1 path, costs on:

| Metric | Value |
|---|---|
| Trades | **780 in 8.95 y = 87.2/yr** (card: ≈ 90; random-walk prediction 91). There are 781 crosses: every cross is an entry, plus one `eod` close. |
| Gross (pnl + spread) | **−18.4 pts/trade**, s.d. 638, **SE 22.9 → t ≈ −0.8**, i.e. statistically zero |
| Net | −30.8 pts/trade; SR −0.41, max drawdown −37 % |
| Longs | 390, gross −39.6 pts/trade |
| Shorts | 390, gross +2.7 pts/trade |
| Long share of exposure | 49.8 % |
| Dev-window drift | −0.20 pts/bar, so drift explains < 2 pts/trade and the side gap is noise |
| Exits | 693 signal exits (+65.8 pts gross), 86 stops (−712 pts ≈ 2.5 × ATR ≈ 71 pips, consistent), 1 eod |
| Holding time | median **4 bars**, IQR 1–15, mean 9.4. The time exit binds on **15.8 %** of trades. |
| By year | alternating sign: 2017 and 2023 positive, 2016/18/20/21 negative. No concentration worth attributing. |

Card assumptions that do not hold:
- ATR(14) median is **27.9 pips** (card: 15–25).
- Typical holding time is 4 bars (card: 15–30).
- The per-trade s.d. is 64 pips (card: 90).

These confirm card review m4. The hold-binding fraction matches card review M4's simulation (P(gap > 30) ≈ 0.15–0.19).

### m5. Stop slippage in the base cost model is 0, while the card's cost list says stops slip "as per the engine's cost model"
86 stops out of 780 trades at 0 slippage. At even 1 pip per stop this is ≈ 0.1 pip per trade, which does not matter here. The card should say that the base model has no stop slippage and that only the §4.2 stress adds it.

### m6. The notebook audit cell and the unit-test audits cover only the prior parameters
- **Notebook.** Uses 600 synthetic **H1** bars and the default params.
- **Tests.** Use H4 synthetic bars, default params only.
- **Gap.** Neither exercises lookback 10 or 120, or hold 6 or 60, where warm-up gating (the ATR null until row 13 overlaps the ROC warm-up at lookback 10) and the time-exit schedule behave differently.
- **My corner audits.** All clean. The gap is coverage, not a bug.

### m7. The module docstring points to the wrong card path
`toy_tsmom.py` line 4 cites `research/systems/FBS/0023_toy_tsmom/hypothesis.md`. The card lives in `research/dryrun/systems/FBS/0023_toy_tsmom/`. This is cosmetic, but provenance links should resolve.

---

## Checks run that found nothing
- **Look-ahead.**
  - `assert_no_lookahead`: exhaustive forced cuts plus 40 random cuts, truncation and future-perturbation. Clean at (10, 1.0, 6), (120, 4.0, 60), (10, 4.0, 60), (120, 1.0, 6) and (60, 2.5, 30) on synthetic H4, and at (10, 1.0, 6) and (60, 2.5, 30) on real dev EURUSD H4.
  - `assert_engine_causal`: clean at all 5 synthetic points.
  - `lint_strategy_source`: `[]` for both `toy_tsmom.py` and `base.py`.
  - By inspection: no `shift(-k)`, centred windows, full-sample statistics or `ts` reads. `pl.int_range(pl.len())` gives row positions only, and `forward_fill` is causal.
- **ROC-zero edge cases.** `ROC_t == 0` never fires, because both branches are strict. `ROC_{t-1} == 0` serves as either side. The cross conditions match the card literally. Existing tests pin both cases, and the loop reference agrees on every cross row.
- **Signal semantics under the engine contract.**
  - Crosses are emitted once, with null elsewhere, so there is no re-entry after a stop (engine docstring minor-11).
  - The time-exit `0` after a stop is a no-op.
  - A reversal closes and reopens at the same open (`_run_core` Step 1).
  - Time exit fires at the open of `entry_bar + hold` (signal row `cross + hold`, fill at +1), so there are exactly `hold` bars of exposure, counted in bars.
  - The stop outranks the time/reversal exit: Step 2 checks bar j's range before the next open's Step 1.
- **Stop anchoring.**
  - `stop_dist` is on the signal row, and the engine reads `stop_dist[j-1]` (the ATR at bar t).
  - The stop is measured from the fill (long: Ask + slippage). The effective Bid-side distance is d − spread for both sides (long Ask-anchored/Bid-triggered, short Bid-anchored/Ask-triggered), so it is symmetric.
  - Sizing uses |entry − stop| = d. No trades were skipped (`n_skipped = 0`).
- **Data.** 14,073 H4 bars from 2016-05-02 to 2025-05-14 20:00, ending before the holdout. No NaN/null closes and no duplicate timestamps. Bars are evenly spread over server hours 00/04/…/20 (EET alignment). The evaluator's `end` is 2025-05-15, and `start=None` means no warm-up is needed.
- **Tests.** `pytest tests/test_strategy_toy_tsmom.py tests/test_strategies.py`: 26 passed. The hand-derived known-answer values (ATR_20 = 0.0045714…, entry 21 / exit 24) check out.

Not checkable yet (carried forward): whether the code's `SearchSpace` carries the card's scales and radius. `ToyTsmomParams` holds them only in a docstring, and the notebook's §4 has `space = None`. Re-check at S4: `study_created.plateau_radius == 0.2` and every param has `plateau_scale == "relative"`.

---

## Process findings
- **P1. The S3 baseline evaluation is not in any ledger.** `research/dryrun/ledger/` does not exist, yet DESIGN §1.3 says "every configuration evaluated on development data is a trial".
  - The prior config sits inside the 840-point grid, so S4 will count it.
  - Any ad-hoc engineer evaluations have no audit trail, including the "separate script" of FINDINGS #24/#25 and any debugging runs at other params.
  - Suggest that `RuleEvaluator` (or the S3 notebook cell) log a `baseline` event with params and a code hash.
- **P2. The S3 kill rule compares point estimates with no uncertainty.** "Gross ≤ modelled cost" at the prior: gross is −18 ± 23 pts against a cost of 12.4.
  - If the true gross equals the card's low-end edge (2 pips = 20 pts), P(observed ≤ 12.4) ≈ 0.37. A real 2-pip edge is killed about a third of the time at S3.
  - Suggest the rule require the gross estimate's **upper** 1-SE (or 90 %) bound to be ≤ cost before killing, or state explicitly that S3 is a cheap screen with a known false-kill rate.
- **P3. The strategy contract is position-blind, but the card writes rules that depend on engine state.** An example is "a same-direction cross after a stop is a new entry, while in position it is not" (m1). A signal frame cannot express this exactly. Either the card template should forbid rules that depend on stop state, or the engine should expose a hook (e.g. an optional `clock_reset` column interpreted against the actual position).
- **P4. Look-ahead audits run only at the prior.** The template notebook audit and the per-system test template should audit at least the grid corners (m6). A factory loop over `space` corners is cheap.
- **P5. The "MT5 iATR = Wilder" claim is in both code and FINDINGS #21** (m2). Deployment parity (D1) needs an explicit indicator-definition table: which MQL5 call, or which hand-rolled recursion, reproduces each `base` helper.
- **P6. The SearchSpace-vs-card check still cannot run at S2 code review** (card review P3), because `space = None`. Either the strategy module should export its `SearchSpace` (or have the card parsed into one) at S2, or the check moves to S4 explicitly.
- **P7. Disclosure of red-team data use.** My probes computed signals (no PnL) on dev data at 15 (lookback, hold) points, and re-evaluated PnL only at the S3 prior config. No new PnL trial was run, so nothing needs to be added to the trial count.
