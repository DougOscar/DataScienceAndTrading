**DRY RUN — toy system** (DESIGN §10 Phase 2 exit test; sandbox `research/dryrun/`)

# Red team — S2 card review, `toy_tsmom` (issue #23)

Red team, 2026-09-30. Reviewed `hypothesis.md`, `_drafts/toy_tsmom_S1.md`, `FINDINGS.md`, DESIGN §4.2/§4.3/§5,
`quantlab.gates._axis_ladder` / `plateau_perturbations`, `quantlab.opt` (SearchSpace, radius floor),
`quantlab.stats` (`ablation_compare`, `random_entry_null`), `quantlab.costs.load_instrument`, `quantlab.data`
(H4 resampling), `quantlab.evaluators` (M1 path), and the legacy notes `DocumentationVault/strategies/01_*`, `04_*`.
No market data was read. Probes (synthetic only):
- `results/probes/s2_plateau_ladders.py`: builds the card's exact SearchSpace and prints the judge's ladder at every grid level, plus the joint axis at both corners.
- `results/probes/s2_crossing_rate_sim.py`: 312k-bar Gaussian and t(3) random walks; ROC zero-crossing rate and the inter-crossing-gap distribution.

Context: S1 returned **KILL-EARLY** (MinTRL 67.6 y against a 9.0 y window; the 840-trial DSR needs an observed SR of ≈ 1.6).
Anything from S2 to S5 is a pipeline exercise only. No number it produces may be quoted as evidence about the idea.

**Severity counts: BLOCKER 0 · MAJOR 5 · MINOR 9 · process 8.**

---

## MAJOR

### M1. The mechanism ablation is run on the in-sample-selected configuration against an unselected null, so it is biased towards "mechanism confirmed"
The card compares "the real system's net Sharpe" with the 95th percentile of random-side Sharpes "under the same schedule".
At S5 the real system means the best of 840 configurations on the full dev window. S1's figure puts the expected maximum of
840 null Sharpes at ≈ 1.07 annualised. Each random-side draw, by contrast, runs one fixed configuration with no search. The real
Sharpe therefore carries the full selection premium and the null carries none.
- **Failure scenario.** A system with zero true side-edge whose selected configuration shows SR ≈ 1.0 in sample clears the random-side 95th percentile, which sits at about 1.65 × 0.33 ≈ 0.55 for 9 years. The ablation then records "momentum confirmed" on pure selection noise.
- **Fix, to pre-register now.** Run the ablation on out-of-sample output. Either:
  - randomise sides on the walk-forward OOS trade schedule and compare against the WFO OOS Sharpe, or
  - put the search inside the null: for each draw, randomise sides and re-select the best configuration over the grid. This costs more but is exact.

  State which one on the card.

### M2. Legacy trials on the same symbol, timeframe and window are omitted from the trial count, and the card mis-describes them
`01_SMA_Crossover_ATR_Risk.md` is **Status: Backtested**, not just a related idea:
- It ran a WFO over a 5×5×5×5 = **625-configuration grid** (fast, slow, sl_atr_mult, tp_atr_mult) on **EURUSD 4h**, EURCAD and GBPCHF, on M1 data resampled from 2016. That is the same symbol, timeframe and essentially the same dev window as this card.
- Recorded outcomes for EURUSD/4h: baseline Sharpe −0.14, WFO-optimised +0.16, and "good 2016–2018; edge degrades sharply from 2019".
- ROC(L) > 0 ⇔ close > close_{t−L}. A price-vs-lagged-price sign is in the same family as the SMA(fast)/SMA(slow) cross with small `fast` (grid down to 5). The ATR stop is the same, the stop-multiple range 1.0–3.0 overlaps, and the reversal rule is the same.

The card says only "neither is a ROC-sign cross with a time exit" and declares **0 mined candidates**. Two consequences:
- **Selection is not independent of the data.** The lead has seen that EURUSD H4 trend-following earned about 0.16 on this window. The card's expected SR of 0.2 matches that result but cites only MOP 2012. The symbol/timeframe choice and the prior were both made after that look.
- **The DSR N is understated.** DESIGN §4.2 lets the card add data-mining trials. At least the EURUSD-4h slice of the legacy grid (625 configurations, and arguably ×3 symbols ×3 timeframes) should be declared, or its exclusion argued explicitly. `04_MACD_Histogram_Momentum` is **Status: Idea** with an empty results table, so it adds no trials; mentioning it as family is fine.
- **Time-stability risk.** The legacy per-year finding (edge concentrated in 2016–18) is a direct prior against the §4.2 time-stability gate. The card should state it.

### M3. Swap is not modelled: the spec is an uncalibrated fallback with swap = 0, and the card's cost section assumes 1–5 pips of swap per trade
`load_instrument("EURUSD", "FBS")` returns `calibrated=False, swap_long=0.0, swap_short=0.0, commission 0` and warns
"no calibrated broker export … uncalibrated fallback" (`data/broker/` does not exist). The cost version is `fbs-v2-uncalibrated`.
- **Costs are understated.** The card's cost budget (net cost 1.5–4 pips/trade, of which swap is 1–5 pips on the paying side) is not what the engine will charge. The engine charges spread plus stop slippage only.
- **The S3 early kill is weakened.** "Gross edge ≤ modelled cost" will use a cost about 1–3 pips/trade too low, so the S3 kill criterion is softer than the card intends.
- **The swap narrative is wrong for this engine.** Even once calibrated, DESIGN §4.3 says only *current* swap rates can be exported, so every year of the backtest carries the 2026 rate. The card's reasoning ("long EURUSD paid negative carry for most of 2016–2022; shorts earned partly back") describes a history the engine will not reproduce.
- **Fix.** Add a cost-model/calibration line to the card. Pre-register a swap sensitivity band (DESIGN §4.3), e.g. ±1 pip/night per side, and a swap-free vs swap-charged break-even. Treat any baseline on the fallback spec as spread-only.

### M4. `hold` is largely inert over much of the grid, and where it binds it truncates exactly the trades momentum is supposed to pay on
Random-walk simulation (Gaussian; t(3) similar) of the ROC(L) zero-crossing process:

| L | crossings/yr | mean gap (bars) | **median gap** | P(gap > 6) | P(gap > 24) | P(gap > 36) | P(gap > 60) |
|---|---|---|---|---|---|---|---|
| 10 | 224 | 7.0 | **3** | 0.37 | 0.04 | 0.01 | 0.00 |
| 60 | 91 | 17.1 | **3** | 0.38 | 0.19 | 0.15 | 0.10 |
| 120 | 64 | 24.3 | **4** | 0.37 | 0.19 | 0.16 | 0.12 |

- **Inert region.** Crossings are strongly clustered (whipsaw around ROC ≈ 0). At L = 10–20, hold ∈ {36, 42, 48, 54, 60} binds on ≤ 1–6 % of trades, so those configurations are near-duplicates. Across the 12 × 10 lookback/hold plane a large block of the 840 configurations barely differ. The raw-N DSR still charges for all of them. That is conservative for the DSR, but it wastes the budget, and the plateau `hold` axis is trivially flat there, so it tells the judge nothing.
- **Where hold binds, it works against the mechanism.** The trades that outlast `hold` are the persistent-sign, trending ones. A time exit closes them and leaves the system flat until the next crossing, which is usually a whipsaw. MOP's TSMOM stays in the position while the signal persists. The card's justification ("exit before the autocorrelation is expected to decay") conflicts with its own mechanism and is not argued.
- **Fix.** Parametrise hold relative to lookback (e.g. `hold_ratio = hold/lookback ∈ [0.25, 2]`, relative scale), or drop the time exit and let the reversal and stop do the work. Otherwise justify why truncating persistent-sign trades is part of TSMOM. Include the hold-binding fraction in the S3 baseline diagnostics.

### M5. The null/benchmark mis-cites DESIGN, duplicates the ablation, never tests timing, and leaves the drift confound without a decision rule
- **Mis-citation.** "This is a full system, so the §4.2 entry-signal null applies" inverts DESIGN §4.2, which says component tests apply when the scout proposes a filter or risk rule *rather than* a full system. For a full system, the mechanism gate is the ablation.
- **Duplication.** As specified (same timestamps, fair-coin side, matched holding bars), the null is the ablation with the stop removed. It is a second, nearly identical test of side, not an independent one. Neither test ever randomises **timing**. The library's `stats.random_entry_null` does randomise timing (random non-overlapping placement, permuted holds, permuted real side mix), so the card's null does not match the tool the statistician will reach for.
- **Drift confound.** EURUSD in the dev window has multi-year one-sided legs (2017 up, 2018–22 down). A sign(ROC) system tilts long or short with those legs and profits from in-sample drift, which a fair-coin side cannot. This is "momentum" only in the trivial sense that any sample with a non-zero mean produces it. The buy-and-hold/short-and-hold benchmark is named but has **no pass/fail rule**.
- **Fix.** Pre-register a per-year check (the ablation's real-minus-random Sharpe is > 0 in ≥ 60 % of years), or a drift-matched side null (random side with P(long) equal to the real system's long-exposure share in that calendar year). Replace the null paragraph with "full system: mechanism gate = ablation (M1 form); optional timing null = `random_entry_null`", or drop it.

---

## MINOR

- **m1. Plateau ends: harsh at the low end, flat at the high end.** Probe output, card space, r = 0.20, floor = 5 % of range:
  - `lookback`: floor 5.5 bars binds for x < 27.5. At x = 10 the ladder is **5 / 7 / 13 / 16 (−50 / −30 / +30 / +60 %)**; at x = 20 it is 15 / 17 / 23 / 26 (±25 / ±15 %). From x = 30 upwards it is exactly ±10 / ±20 %. At x = 110 and 120 the up moves (121, 132; 132, 144) leave the search bounds; they are kept, evaluated and flagged.
  - `hold`: floor 2.7 binds for x < 13.5. At x = 6 the ladder is **3 / 5 / 7 / 9 (±17 % inner, ±50 % outer)**; at x = 12 it is 9 / 11 / 13 / 15 (±8 / ±25 %). Elsewhere integer snapping gives ±8–11 % and ±19–22 %, which is immaterial.
  - `stop_mult`: floor 0.15 never binds; the ladder is always exactly ±10 / ±20 %.
  - Joint axis at the (10, 1.0, 6) corner: "all −r" = (5, 0.8, 3), i.e. a 1-day ROC, a stop below the card's own "inside H4 noise" bound, and a 3-bar hold.

  **Verdict.** The scales are economically meaningful in the interior and never too kind. At the low ends of lookback and hold the floor plus snapping makes the test much harsher (a different system, ~300 trades/yr), so a genuine plateau at L ≤ 20 or hold = 6 would probably fail. Since costs dominate there, this is conservative and acceptable. At the high ends (stop_mult ≥ 3.5, hold ≥ 48) the axis is near-flat because the parameter rarely binds. Weakest-axis scoring means that cannot pass a fragile optimum, but those axes add no information. Relative scale is correct for all three (strictly positive, economically meaningful zero). A radius of 0.20 is at the default and above the 0.10 floor.
- **m2. The card's floor description is imprecise.** "Overrides r/2 at the low end" is not what the code does. The floor replaces the **outer (r) move** when r·|x| < 5 % of range, and the inner move becomes half the floor. See P2.
- **m3. The trade-frequency formula is right, but the fat-tail claim has the wrong sign.** The simulation matches √(2/L)/π to three digits for Gaussian steps. With t(3) steps, crossings **fall** 6–12 % (L = 60: 81/yr against 91), which contradicts "fat tails … usually increase whipsaw crossings". Mean reversion would raise the count; fat tails lower it. The "±30 %" band still covers this.
- **m4. The holding-time and per-trade s.d. figures describe the mean, not a typical trade.** With a median gap of 3–4 bars, most trades are short whipsaws. The "20 × √20 ≈ 90 pips" per-trade s.d. overstates the typical trade (≈ 35–40 pips at 3–4 bars) and understates how much of the book is cost-dominated churn.
- **m5. The expected Sharpe is not tied to a parameter point, and the card's own arithmetic implies about 0.05–0.1.** Gross 2–4.5 pips against net cost 1.5–4 pips gives a net of about +0.5 to +1 pip/trade at the midpoint. At s.d. 90 pips and 90 trades/yr that is SR ≈ 0.05–0.1. The card calls 0.2 "deliberately generous", which is honest, but it should say this is the prior-parameter OOS expectation. It should also say that the in-sample best of 840 will read ≈ 1.0 on selection alone (S1). The repo's own legacy EURUSD-4h number (+0.16 WFO, −0.14 baseline) is the more relevant prior (see M2).
- **m6. A same-direction signal while in a position is possible but not specified.** The card says it "cannot occur". It can when ROC is exactly 0 for a bar: the sequence +, 0, + gives `ROC_{t−1}=0 ≤ 0` and `ROC_t > 0`, a long signal while already long. It is rare at 5 digits, but the rule must be specified (ignore / restart clock / re-enter) so the code and the ablation agree.
- **m7. The risk-semantics claims are too strong.** "Flooring keeps realised risk ≤ target" ignores stop slippage and gap-through fills; weekend holds happen at every hold ≥ 6. The ≥ 1 % over-target cases should be reported, not assumed away. "Typical 30–80-pip stop" does not match the grid: 1.0–4.0 × ATR 15–25 is about 15–100 pips. At 1 × ATR, the 00:00 rollover spread spike (M1 path, Ask-triggered) is a material share of R for **shorts**, a side asymmetry worth reporting in the stop-hit statistics. Risk type **A** is consistent with the rules (fixed fraction, price stop, variable lots, no target).
- **m8. The ablation is badly underpowered at the card's SR.** The random-side Sharpe s.d. over 9 y is ≈ 0.33, so a true 0.2 side-edge exceeds the 95th percentile only about 15 % of the time. A "contradiction" is therefore almost uninformative for this card. That is fine because S1 already kills it, but the card should say so. Criterion (b) (real − median ≤ 0) is implied by (a) and is redundant.
- **m9. The range sits entirely below the literature horizon.** lookback ≤ 120 H4 bars ≈ 20 trading days only reaches MOP's shortest (1-month) lookback. The card acknowledges this as extrapolation. Anchoring the grid's top at the documented horizon (or extending to ~260 bars) would let the study show whether any edge rises towards where the mechanism is documented.

---

## Checks run that found nothing
- **Parameter types.** All three are numeric; no numeric knob is declared as categorical; no unordered categoricals. The card's 5 %-floor figures (5.5, 0.15, 2.7) are arithmetically correct; grid size 840 is confirmed by `SearchSpace.grid_size()`.
- **Relative scale** is applied only to strictly positive parameters with a meaningful zero; the radius of 0.20 is ≥ 0.10.
- **Rules / look-ahead at card level.**
  - Signal at close t, fill at open t+1.
  - ATR(14) is taken at signal bar t.
  - The stop is fixed at the fill.
  - The time exit is counted in bars.
  - No centred windows, `shift(−k)` or full-sample fitting are implied.
  - H4 bars are resampled from M1 with `group_by_dynamic(every="4h", closed="left", label="left")` on naive EET timestamps, so they align to server midnight (00/04/…/20).
  - The H4 `spread` column is the first M1 spread of the bar, so the 00:00 rollover spike *is* charged on entries at that bar, consistent with the card's "1/6 of entries at ≈ 2.8 pip".
- **Intrabar ordering.** `RuleEvaluator.use_m1=True` by default for H4, so stop-versus-open ordering is resolved at M1 and short stops trigger on Ask per minute.
- **Risk type A** is consistent with the rules; the lot arithmetic (1.25–3.3 lots at 30–80 pips, min lot never binding) checks out.
- **Honesty.** The card flags its own weaknesses (horizon extrapolation, underpowered, thin cost headroom), and the S1 verdict agrees with its t ≈ 0.6.

Not checkable at S2 (no code yet):
- **SearchSpace vs card.** `quantlab/strategies` has no `toy_tsmom` and the notebook's §4 has `space = None`, so I could not check that the code's `SearchSpace` carries exactly these scales and radius. Re-check at S3/S4, and confirm that `study_created.plateau_radius == 0.2` and that each param's `plateau_scale == "relative"`.
- Look-ahead probes (`assert_no_lookahead`, `assert_engine_causal`, `lint_strategy_source`) wait on the strategy code.

---

## Process findings (template / procedure)
- **P1. No public way to see the plateau ladder.** I had to call the private `gates._axis_ladder` to see what the judge will actually do at each grid level. Suggest a public `opt.describe_plateau(space, radius)` that prints the ladder at every grid level plus both joint corners, and have the scout paste it on the card.
- **P2. DESIGN and code disagree on the 5 % floor.** DESIGN §4.2 says "a move is never smaller than 5 % of the declared range". The code floors the outer move at 5 % and the inner (±r/2) move at **2.5 %** (`gates._axis_ladder`: `unit = max(r·|x|, floor)`, inner = unit/2). One of the two should change.
- **P3. S2 checks run before the code exists.** The agent file's S2 check "the code's `SearchSpace` must carry exactly the card's scales and radius" cannot run at card review. Split it into a card-level check (S2) and a code-level check (S3/S4 red team, or an automated assertion in `run_study` against the card table).
- **P4. No calibration field on the card.** The template has nothing for cost-model version or calibration status. A card can budget swap that the engine will charge as 0 (M3). Add a field such as "cost model: `<version_tag>`, calibrated yes/no, swap modelled yes/no".
- **P5. No field for legacy work.** There is no field for pre-ledger work on the same symbol/timeframe or the results the author has already seen, and DESIGN gives no rule for converting legacy grids into "declared mined trials" (M2). Add "Prior looks at this data (legacy or ledger): what, which grid, what was seen", and a rule for counting it.
- **P6. The mechanism-gate tool does not fit the ablation.** `stats.ablation_compare` is an on/off comparison for stop/sizing rules and returns no p-value. The card's random-side ablation (≥ 1000 draws, 95th percentile) needs a helper that does not exist (`random_entry_null` randomises timing, not side at fixed timestamps). DESIGN §4.2 names `ablation_compare` for the mechanism gate, so either extend `stats` with a side-randomisation null or reword DESIGN.
- **P7. The template does not say where the ablation runs.** It asks for an ablation but not whether it runs in-sample at the selected point or on OOS output. That is exactly the M1 selection-bias trap, and it should be a template field.
- **P8. The null/benchmark field invites misreading.** "Null / benchmark for component tests" led the scout to apply a component null to a full system (M5; also FINDINGS #5). Rename it to "Mechanism ablation (full system) / component null (filter or risk rule only)" and state whether stops stay on.
