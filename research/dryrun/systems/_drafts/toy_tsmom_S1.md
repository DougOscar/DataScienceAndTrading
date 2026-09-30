**DRY RUN — toy system** (DESIGN §10 Phase 2 exit test; sandbox `research/dryrun/`)

# S1 power check — `toy_tsmom` (issue #23)

Validation statistician, 2026-09-30. Card: `research/dryrun/systems/_drafts/toy_tsmom.md`.
Tool: `quantlab.stats.power_check(expected_sr_annual, trades_per_year, years_available, periods_per_year=260)`
(MinTRL on daily returns, 95 % one-sided, SR* = 0, normal moments). No data was read; no ledger write.

## Verdict: **KILL-EARLY**

At the card's point estimate (SR 0.2 net, annualised), MinTRL is **67.6 years (17,588 trading days)**. The dev
window has **9.03 years (≈ 2,358 weekdays)**, so MinTRL is 7.5× longer than the window. The card's optimistic end (SR 0.3)
still needs **30.1 years**. No point in the card's stated ranges is feasible.

## Inputs

| Input | Value | Source |
|---|---|---|
| Expected SR (annual, net) | 0.2 (range 0.0–0.3) | card |
| Trades/year | ≈ 90 at lookback 60 (range 64 at lookback 120 to 220 at lookback 10) | card |
| Dev window | 2016-05-02 → 2025-05-14 = 9.032 years, 2,358 weekdays | `data/manifest.json` (EURUSD M1 start), `config` FBS holdout_start 2025-05-15 |
| periods_per_year | 260 (daily) | DESIGN: Sharpe from daily %-equity returns |
| skew / kurt | 0 / 3 (default; the card gives none) | see the sensitivity below |

## Results: `power_check` across the card's ranges

| SR (annual) | trades/yr | MinTRL (years) | MinTRL (days) | feasible | trades in dev | trades needed |
|---|---|---|---|---|---|---|
| 0.0 | 64 / 90 / 220 | ∞ | ∞ | no | 578 / 813 / 1,987 | ∞ |
| 0.1 | 64 / 90 / 220 | 270.6 | 70,346 | no | 578 / 813 / 1,987 | 17,316 / 24,351 / 59,524 |
| **0.2** | 64 / **90** / 220 | **67.6** | **17,588** | **no** | 578 / **813** / 1,987 | 4,329 / **6,088** / 14,882 |
| 0.3 | 64 / 90 / 220 | 30.1 | 7,818 | no | 578 / 813 / 1,987 | 1,925 / 2,706 / 6,616 |

The per-trade MinTRL is computed the way the S5 `trade_count` gate does it: per-trade SR = SR_ann/√(trades/yr),
`min_trl(..., periods_per_year=1)`. At SR 0.2 it gives 4,331 / 6,090 / 14,883 trades against 578 / 813 / 1,987 available. This
matches `power_check`'s `trades_needed` to within rounding. On both the days test and the trades test the window falls
short by a factor of about 7.5.

**Why trade count cannot fix this.** Under the card's own model, trades/year changes the per-trade Sharpe in the
opposite direction by exactly the same factor (SR_trade = SR_ann/√n). The daily MinTRL in years does not depend on
trade frequency. The binding quantity is the annual Sharpe.

**Reference thresholds for this window (N = 1, no search penalty):**
- The smallest annual SR at which MinTRL ≤ 9.03 years is **0.55**. That is the SR at which the point estimate just reaches 95 %, which is only ≈ 50 % power.
- The SR for 80 % power at 95 % is **0.83**.
- At the card's SR, the t-stat over the window is 0.30 / **0.60** / 0.90 for SR 0.1 / 0.2 / 0.3. The card's own note (t ≈ 0.6) is confirmed. The probability of a 95 % detection is 9 % / 15 % / 23 %.
- To reach 80 % power the window would have to be 618 / 155 / 69 years long.

**Moment sensitivity.** Raising kurtosis from 3 to 10 or 30 moves MinTRL at SR 0.2 from 67.65 to 67.67 or 67.72 years. At a daily SR of about 0.012 the
(γ4−1)/4·SR² term is negligible. Skew behaves the same way. The missing moments on the card do not matter here.

## Raw N = 840 and the DSR hurdle

The DSR hurdle follows DESIGN §4.2 v1.2: SR0 = √(1/(T−1)) · E[max of N], with T ≈ 2,358 daily observations and **raw N = 840** (the full 12×7×10
grid). Prior trials are 0: there is no ledger study with this name or issue, and the card declares 0 mined candidates. The legacy
SMA/MACD notes are not in the ledger, so the DSR does not count them. The red team may argue family overlap at S2.

| raw N | SR0 hurdle (annual) | observed SR needed for DSR ≥ 0.95 |
|---|---|---|
| 1 | 0.00 | 0.55 |
| **840** | **1.07** | **≈ 1.61** |
| 1,000 | 1.08 | ≈ 1.63 |
| 1,680 (e.g. one more attempt of equal size) | 1.13 | ≈ 1.68 |

The search penalty is decisive at this Sharpe:
- **The hurdle.** The 840-trial hurdle alone (1.07) is 5× the card's expected SR.
- **Observed at expectation.** If the selected configuration were observed at exactly 0.2 or 0.3, its DSR would be **0.005 / 0.011**.
- **Chance of passing.** The probability that a true-SR-0.2 system clears DSR ≥ 0.95 is ≈ 1×10⁻⁵ for the selected configuration alone. Even treating all 840 grid points as independent draws, each with true SR 0.2, it is ≈ 0.9 %; the null value under the same assumption is 0.05 %.
- **Better statistics would not help.** Even a true SR of 1.0 would pass only ≈ 3 % of the time with N = 840. MinTRL is therefore not the only blocker. Even if a longer window existed, the pre-registered grid size puts the DSR out of reach for any Sharpe near the card's range.
- **Effective N** (eigen / cluster / Li–Ji) can only be estimated from the trial matrix at S5 and is a diagnostic only. It does not lower the gate's N.

## Process findings: card vs `power_check`

1. **`feasible` is not a power statement.** `feasible` means MinTRL ≤ window at the *point* SR: about 50 % power, N = 1, SR* = 0. It ignores
   the search penalty (the DSR's raw N) that S5 will apply. A card can be "feasible" at S1 and still have
   essentially no chance of passing the DSR gate. Here it makes no difference because S1 already fails. For cards near the margin, S1
   should also report the N-adjusted hurdle, as done above.
   → Suggest `power_check` take `n_trials` and return power at SR0(N) plus the SR for 80 % power.
2. **`trades_per_year` does not enter the decision.** It is only used for the reported `expected_trades` /
   `trades_needed`, and `trades_needed` equals the per-trade MinTRL only under the iid mapping
   SR_trade = SR_ann/√n, which held here. DESIGN §3 S1 says "MinTRL given the expected Sharpe *and trade frequency*", and the
   card's ending ("may kill early ... or a redesign raising trade count") suggests trade count can rescue a card. Under this model it cannot.
   The card's trade range therefore has no effect on the verdict.
3. **The card's SR range is not tied to its parameter range.** The Sharpe (0.0–0.3) is given once, while trades/yr is given per
   lookback. `power_check` takes one (SR, trades) pair, so the sensitivity above crosses them as a full grid. The card template does not say
   whether the expected SR is for the prior-parameter configuration or for the best of the grid.
4. **`years_available` is supplied by hand.** Nothing derives it from `config` (holdout_start) and the manifest's first bar.
   I used the EURUSD M1 start (2016-05-02). **Flag for S2:** the manifest has no EURUSD H4 file. The H1 file starts
   2016-09-22, so if H4 is resampled from H1 instead of M1 the dev window shrinks to ≈ 8.6 years. The verdict does not change.
5. **No skew or kurtosis on the card.** This is harmless at low SR (shown above), but the template could ask for them for high-SR,
   fat-tailed systems.
6. **Positive:** the card's own power-check note (t ≈ 0.6, "S1 should flag this as underpowered") matches the tool.

## Trial-count calculation (for the record)
- N used by the DSR = 840 (this study's pre-registered grid) + 0 (related ledger studies: none; the dry-run ledger has no
  study for this system or issue) + 0 (mined candidates declared on the card) = **840**.
- Effective N is not computable at S1 (no trial matrix). It is a diagnostic only at S5.

## Recommendation
**KILL-EARLY** at S1: the dev window cannot reach MinTRL at any Sharpe in the card's stated range, and the 840-trial DSR
hurdle (≈ 1.6 observed SR needed) is far above it. This is the outcome the card anticipated. Whether the dry run continues
past S1 anyway, to exercise S2–S5 as a pipeline test, is the lead's or the user's decision. It is not a statistical CONTINUE.
