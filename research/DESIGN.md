# Research Team Design — v1.2

Status: **v1.1: approved 2026-09-23** · **v1.2 (2026-09-24): Phase 1 gate recalibration (§4.2, §4.4, §11 #7–#8)** · The decisions from §11 are merged into the sections below.

---

## 0. Principles

1. **Evidence, not opinions.** Agents only act through a tested library (`quantlab/`) and an append-only trial ledger. An agent's claim counts only if it comes with code output.
2. **Separation of powers.** Whoever proposes an idea doesn't tune it, and whoever tunes it doesn't judge it. The Statistician and Red Team never propose or optimise. The Scout never evaluates.
3. **Every trial counts.** Every configuration evaluated on development data is a trial in the ledger. The number of trials feeds the deflation of every statistic.
4. **The holdout is locked in code**, not by honour system. Only the user can unlock it, once per system.
5. **Two separate books that never mix:**

   | | FBS book | B3 book |
   |---|---|---|
   | Markets | Forex, metals, crypto | WIN / WDO futures |
   | Timeframes | H1 / H4 / D1, swing | M1 / M5 / M15, intraday only |
   | Costs | Spread + swap | ~0 fees, 20% monthly day-trade tax |
   | Priority | **First** | Later |

   Each book has its own cost model, holdout, portfolio and tax model.
6. **The system's own rules decide which metrics apply** (see §5). The report states which metrics are excluded and why.
7. **Efficiency is a design constraint**, not an afterthought (see §7).

---

## 1. Architecture

```
You ──► main session running /research-cycle   (the orchestrator; subagents cannot call subagents)
            │
            ├── specialist subagents (.claude/agents/*.md)
            │
            ▼
        quantlab/  (library: the only way to touch data, run backtests, compute stats)
            │
            ├── research/ledger/     append-only record of every study and every holdout access
            └── research/systems/    one folder per system: hypothesis card, notebook, results
```

### `quantlab/` modules

| Module | Responsibility |
|---|---|
| `data` | Lazy Parquet loader. Timezones: FBS server time (EET) → UTC, Clear is BRT. D1 bars follow the server day (EET midnight), exactly like MT5: normally 17:00 New York, 18:00 during the 2–3 US/EU DST-mismatch weeks. Resampling. **Holdout guard**: dev-period reads are allowed; holdout reads raise `HoldoutLocked` unless an unlock token exists in the ledger. Economic calendar join. |
| `costs` | One cost model per book (§4.3). Takes its parameters from the MT5 export (`ExportBrokerSpecs.mq5`). |
| `engine` | Signals are vectorised (Polars/NumPy); the fill/stop/size loop is compiled with Numba. Entries execute at the next bar's open. Stop/target order inside a bar is resolved from the **M1 path**. Lot-step and minimum-lot rounding. |
| `stats` | PSR/DSR, MinTRL, CSCV-PBO, SPA / Reality Check, stationary bootstrap, regime splits, component ablation tests, effective-number-of-trials estimation. |
| `opt` | Search spaces, Optuna studies that log every trial, CPCV and anchored walk-forward with purging/embargo, plateau selection. |
| `portfolio` | Per-book combination (HRP / risk parity / equal-risk), stress-period correlation, leverage to a drawdown budget. |
| `report` | Notebook tear sheet with a metric-interpretation table, plus the Obsidian card generator. |
| `ledger` | Append-only writer and reader, trial counters, holdout access log. |

`source/` stays frozen for the legacy notebooks. Old strategy *logic* may be ported, but none of the old evaluation or optimisation code.

### Directory layout

```
quantlab/                               library (tested, typed)
tests/                                  unit tests + look-ahead tests + known-answer tests
research/
  DESIGN.md                             this file
  ledger/studies.jsonl                  git-tracked, append-only, one row per study
  ledger/holdout_access.jsonl           git-tracked, append-only
  studies/<study_id>/                   gitignored: per-trial parameters, metrics, return matrices (parquet)
  systems/<book>/<issue#>_<slug>/
      hypothesis.md                     the card approved at checkpoint A
      <slug>.ipynb                      notebook: implement → optimise → validate → report
      results/                          figures + metrics.json
DocumentationVault/systems/<slug>.md    Obsidian summary card (§6)
```

---

## 2. The team

The model split follows your answer: Opus for judgment roles, Sonnet for mechanical ones.

| Agent | Model | Mission | Must not |
|---|---|---|---|
| **quant-scout** | Opus | Find ideas for signals, filters, risk management and full systems. Sources: arXiv/SSRN/journals, MQL5 and BabyPips forums, data mining. Writes **hypothesis cards** as GitHub issues. Runs only when you ask. | Run backtests, judge results, cite social media or news as a source |
| **strategy-engineer** | Sonnet | Turn an approved card into `quantlab` strategy code and a notebook. Writes the look-ahead tests. | Tune parameters, pick the "best" result |
| **optimization-architect** | Opus | Design the search space, the parameter budget, the cross-validation scheme and the objective. Run studies and select parameters from a plateau. | Look at the holdout, declare a system valid |
| **validation-statistician** | Opus | Compute every statistical gate (§4). Holds veto power. Can also kill an idea early when the power analysis shows it can't reach significance. | Propose fixes or tune parameters (it only measures) |
| **red-team** | Opus | Attack each candidate: look-ahead, leakage, snooping, cost realism, regime dependence, code bugs, whether the claimed mechanism actually drives the profit. Separate context, so an independent pair of eyes. | Rewrite the strategy (it files findings only) |
| **portfolio-risk-manager** | Opus | Measure each system's marginal value to its book: correlation (including in crises), risk allocation, the book's return at the drawdown budget. | Mix books, change a system's rules |
| **data-auditor** | Sonnet | Gaps, timezone and DST issues, bad ticks, spread anomalies, back-adjusted series (WIN/WDO), calendar alignment, manifest integrity. | Change raw data without logging it |
| **execution-cost-modeler** | Sonnet | Build and calibrate the per-book cost models (spread by hour of day, swaps, commission, slippage, gaps, B3 tax). | Tune strategies |
| **performance-engineer** | Sonnet | Profile, then optimise: Numba, Polars, caching, shared-memory parallelism, GPU for ML, checkpointing. Guards the performance budgets in §7. | Change numerical results (outputs must match bit-for-bit or within a stated tolerance) |
| **report-builder** | Sonnet | Notebook tear sheet, metric interpretation, the "metrics applicability" table, the Obsidian card. | Editorialise beyond what the stats support |
| **mql5-porter** | Sonnet | **Only for systems you've promoted.** Port the system to MQL5, or export it to ONNX for ML. Run the parity test on the holdout: Python vs MT5 Strategy Tester, and Python vs ONNX predictions. | Run at any other time |

**Recurring loop vs build-once.** After the library exists, a normal research cycle uses:
- **Every cycle:** scout → engineer → optimizer → statistician → red team → reporter.
- **At promotion time only:** the portfolio manager.
- **Mostly during the build and when data changes:** the data auditor, cost modeler and performance engineer.

---

## 3. Pipeline and checkpoints

```
S0 Idea ─► hypothesis card (GitHub issue, label `hypothesis`)
   ★ A  you approve the batch of ideas to test
S1 Power check      statistician: can the dev period (2016-05→2025-05) even show this edge?
                    (MinTRL given the expected Sharpe and trade frequency) ─► may kill early
S2 Implement        engineer + look-ahead tests; red team audits the code          label `testing`
S3 Baseline         prior parameters, costs on. Is the gross edge larger than the costs?
S4 Optimise         optimizer: CPCV/walk-forward, every trial logged, plateau selection
S5 Validate         statistician: gates (§4)
S6 Attack           red team
     ↺ Improvement loop: at most 3 attempts (each a new logged variant) ─► then `killed`
S7 Report           tear sheet + classification
   ★ B  you unlock the holdout (one shot)
S8 Holdout          pass = results fall inside the predicted band (§4.4)
S9 Portfolio fit    portfolio manager: marginal contribution to the book
   ★ C  you decide ─► `promoted` (closed)  or  `killed` (closed)
──────────── research ends here; deployment begins ────────────
D1 MQL5/ONNX port + parity test ─► MT5
```

- **Issue labels:** `hypothesis` → `testing` → one of the two **done** states, `killed` or `promoted`. Both close the issue. `promoted` hands the system to deployment. After deployment, `retired` marks a system removed by a decay review (§4.6).
- **Killed systems still get an Obsidian card**, with the reason. Recorded negative results stop us from retesting the same idea.

---

## 4. Statistical protocol

### 4.1 Where the data goes

| Book | Development | Holdout (locked) | Newer data |
|---|---|---|---|
| FBS | 2016-05 → 2025-05-14 | 2025-05-15 → 2026-05-15 | Anything exported after 2026-05-15. Used as additional unseen data (a renewing holdout). |
| B3 | Start → end − 12 months | Last 12 months of each series | Same |

### 4.2 Gates a single system must pass (development data, after costs)

| Gate | Threshold | Why |
|---|---|---|
| Deflated Sharpe probability | ≥ 0.95 | Is the Sharpe real after accounting for *all* trials? Hurdle from the null sampling variance 1/(T−1) and the **raw** trial count of the study plus earlier attempts (v1.2; effective-N estimates are diagnostics only) |
| CSCV out-of-sample loss | P(OOS Sharpe of the in-sample best < 0) < 0.10 | Does the chosen configuration lose money out of sample? (v1.2: replaces PBO < 0.30, which stays as a diagnostic) |
| OOS Sharpe (CPCV path median) | ≥ 1.0 annualised | The minimum edge needed to enter a book |
| Walk-forward procedure OOS | Sharpe ≥ 0.5 over the whole WFO OOS span, and > 0 over its most recent third | The re-optimisation procedure (§4.6) still works, and recently (v1.2, new) |
| Cost stress | Sharpe > 0.5 at 1.5× spread + 1 pip adverse slippage on every market and stop fill | The edge must survive worse fills |
| Parameter plateau | ≥ 60% of neighbours within 50% of the peak Sharpe; neighbours = ±20% of each parameter (pre-registrable on the card), computed by the judge, not the optimizer | Rejects sharp, fragile optima |
| Time stability | Positive in ≥ 60% of years, and no single year > 40% of total PnL | Your own finding: past edges were concentrated in 2016–18 and 2021–22 |
| Trade count | ≥ MinTRL | Enough evidence for the claimed Sharpe |
| Mechanism check | Component ablation matches the hypothesis | The system must make money *for the stated reason* |

**Component tests** apply when the Scout proposes a filter or a risk-management rule rather than a full system. The null model depends on the component:
- **Filter:** compare against a random filter with the same selectivity, applied to the same base system.
- **Entry signal:** compare against random entries with the same exits and holding-time distribution.
- **Stop/sizing rule:** compare with and without it, judged on DD, CVaR and return at the DD budget.

**Across a batch of ideas:** Benjamini–Hochberg FDR control, and Hansen's SPA test for "is the best of the batch better than the benchmark?".

### 4.3 Execution realism (built into the engine)

- **MT5 bars are Bid prices:**
  - Longs enter at Ask (Bid + spread) and exit at Bid.
  - Shorts enter at Bid and exit at Ask.
  - A short's stop triggers when the Ask touches it.
- **Spread comes from each bar's spread column** (hour-of-day aware). The bar spread is treated as optimistic, and the cost model calibrates a multiplier for it.
- **Stops:** fill at stop + slippage. If the price gapped past the stop, the fill is at the gap open.
- **Swap:** charged per night held at 00:00 server time, triple on the rollover day, using the broker's swap mode. Only *current* swap rates can be exported, so the cost-stress test includes a swap sensitivity band.
- **P&L currency:** converted to the account currency using cross rates from our own data.
- **B3:** forced flat before the session close, taken from the exported sessions. Tax is applied monthly with loss carry-forward. All tax rates are settings you confirm.

### 4.4 Holdout pass criterion (defined *before* unlocking)

Build the predictive distribution from the **walk-forward procedure's OOS series** (the procedure the holdout runs, §4.6) plus a stationary bootstrap. The system passes if:
- its holdout Sharpe and return-at-budget are **above** their lower band;
- its max drawdown is **within** its upper band;
- its trade count is **within the predicted range** (from the procedure's own trade counts).

v1.2: the four bands are set **jointly**, so that about 90% of bootstrap draws of a real edge pass all four at once. The band also reports its power against a zero-edge holdout; a one-year holdout has little power, so the report says when it isn't decisive on its own.

Fail → `killed`, with no re-tries on the same holdout.

### 4.5 Classification (descriptive; goes on the card)

"Monthly return at a 10% drawdown budget": scale the position size so that the bootstrapped 95th-percentile max DD equals 10%, then take the mean monthly return.

| Monthly return at budget | Label |
|---|---|
| > 10% | profitable |
| 6–10% | reasonable |
| < 6% | unprofitable |

**Kill criterion (approved):** failing the §4.2 gates after 3 improvement attempts. The label above only *describes* a system; it isn't a kill criterion. A real but modest edge (e.g., 3%/month at the budget) can still be valuable in a book of uncorrelated systems. The 6%/month target applies at the book level.

### 4.6 Alpha decay and re-optimisation (approved)

Edges decay, so a system is validated as a **procedure**, not as one fixed parameter set.

1. **The re-optimisation schedule is part of the system.** The optimizer fixes a schedule (e.g., re-fit quarterly on an anchored or rolling window of length L) and validates *that schedule* with an anchored walk-forward. Every re-fit uses only data available at that moment. CPCV is still used for parameter-plateau and PBO analysis inside each window.
2. **The holdout exam runs the procedure.** The scheduled re-fits happen *inside* the holdout, exactly as they would live. The procedure is frozen; the parameters are allowed to move.
3. **Parameter drift is a diagnostic.** If the chosen parameters jump between very different regions from one re-fit to the next, the edge is probably unstable. This counts as a red-team finding.
4. **Decay review** (`/decay-review <system>`, run when you ask). Each promoted system is re-run on all data after its holdout (newer MT5 exports). Checks:
   - rolling performance vs. the predicted band;
   - a CUSUM / sequential test on the Sharpe;
   - live results vs. backtest drift, once it's deployed.

   **Retirement rule:** 6 consecutive months below the predicted 10th percentile, **or** a CUSUM alarm → the system is flagged `retired` (a deployment-phase state) and removed from its book. A retired system may come back only as a *new* hypothesis card, with its trials counted.

---

## 5. Rule-aware metrics

Every hypothesis card declares the system's **risk semantics**. The report switches metrics on or off based on it.

**Units: points first (approved).** Trade-level analysis for a single symbol is reported in **points/pips**, which work for any account size. It's also reported in **R** (types A–C) and in **ATR units**, because points aren't comparable *across* symbols (1 EURUSD pip ≠ 1 XAUUSD point). Anything that aggregates across symbols or needs a DD budget (Sharpe from daily equity, return at budget, the book level, lot-rounding error) uses **% of equity on a nominal 100,000 account**, in USD for FBS and BRL for B3.

| Type | Rule | Relevant | Not relevant / reinterpreted |
|---|---|---|---|
| **A** | Fixed-fraction risk (e.g., 2%) with a price-based stop (e.g., ATR) → **lot size varies** | R-multiples, expectancy in R, **risk-realisation error** (actual risk vs target after lot rounding and gaps), max losing streak in R | Raw points per trade (not comparable) |
| **B** | Fixed lot size, stop distance varies | P&L in currency and % of equity, the distribution of risk per trade | R-multiples (the "R" changes every trade) |
| **C** | No hard stop (exit on signal or time) | MAE/MFE, worst trade, CVaR, holding time | R-multiples, stop-hit statistics |
| **D** | Continuous / volatility-targeted position (ML, RL) | Return-based metrics, turnover, exposure, costs per unit of turnover | Win rate, profit factor, per-trade statistics |

**Always reported:**
- **Returns:** CAGR, average monthly return, return at the 10% DD budget.
- **Risk-adjusted:** Sharpe (from daily returns), Sortino, Calmar, PSR/DSR.
- **Risk:** max DD, longest drawdown duration, Ulcer index, daily CVaR95, worst month, % of time under water.
- **Costs:** cost as % of gross P&L, break-even spread multiple, share of costs from swap.
- **Robustness:** PBO, plateau score, cost-stress Sharpe, year consistency, holdout z-score.

**Every tear sheet includes:**
- a **metrics applicability table**: metric → included/excluded → why;
- one line of **interpretation** per metric (e.g., "PBO 0.42: the selected parameters beat the median out of sample only 58% of the time; high overfitting risk").

---

## 6. Obsidian card template

```markdown
# <System name>
**Book:** FBS | B3   **Status:** promoted | killed (<stage>, <reason>)   **Issue:** #N
**Idea:** <≤200 chars>
**Result:** profitable | reasonable | unprofitable — <X.X>%/month at 10% DD budget
**Risk profile:** <label> — MaxDD <..>, longest DD <n> months, max losing streak <n>,
                  skew <..>, CVaR95 <..>
**Robustness:** DSR <0.xx> · PBO <0.xx> · holdout z <..> (or "not unlocked")
![[<slug>_sharpe_yearly.png]]
![[<slug>_sharpe_monthly.png]]
```

**Risk-profile labels** are based on shape, not leverage, since every system is scaled to the same DD budget:

| Label | Typical shape |
|---|---|
| **Steady** | Short drawdowns, mild tails |
| **Grinder** | High win rate, negative skew. Hidden tail risk. |
| **Trend-like** | Low win rate, positive skew, long losing streaks |
| **Lumpy** | P&L concentrated in a few periods |

The monthly Sharpe chart shows bootstrap confidence bands, because a single month (about 21 daily returns) is very noisy.

---

## 7. Performance standards

| Budget | Target on this laptop (8 cores / 16 threads, 32 GB RAM, RTX 3060 6 GB) |
|---|---|
| One 9-year H1 backtest with M1 intrabar resolution | < 1 s |
| Optimisation study (≈500 trials × CPCV) | < 1 h |
| Full validation suite for one system | < 15 min |

Rules:
- **Profile before optimising.**
- **No DataFrame pickled per task** (the old `runner` does this). Use shared-memory arrays or memory-mapped Parquet.
- **Cache indicators** keyed by (symbol, timeframe, parameters, data hash).
- **Checkpoint and resume** any job longer than 10 minutes.
- **RAM guard:** a job refuses to start if its estimated footprint is more than 24 GB.
- **ML:** GPU LightGBM/XGBoost, with batch sizes that fit 6 GB of VRAM.
- The performance engineer keeps **benchmark tests** so regressions are caught.

---

## 8. Ledger schema (`research/ledger/studies.jsonl`, append-only)

```json
{"study_id": "fbs-0007-a2", "issue": 7, "book": "FBS", "system": "donchian_breakout",
 "attempt": 2, "parent_study": "fbs-0007-a1", "created": "2026-10-02T14:03Z",
 "git_commit": "abc1234", "data_manifest_sha": "…", "cost_model_version": "fbs-v1",
 "dev_window": ["2016-05-02", "2025-05-14"], "search_space": {"…": "…"},
 "cv_scheme": "CPCV(n=10,k=2,purge=…,embargo=…)",
 "n_trials": 480, "effective_trials": 37.2, "selected_params": {"…": "…"},
 "gates": {"dsr": 0.97, "pbo": 0.18, "oos_sharpe": 1.21, "…": "…"},
 "decision": "pass|fail|killed", "notes": "…"}
```

- Detail for each trial (parameters, per-fold metrics, return matrix) goes in `research/studies/<id>/*.parquet`. These files are gitignored and can be backed up to the T7 drive.
- `holdout_access.jsonl` has one row per unlock. The data guard refuses a second unlock for the same system.

---

## 9. Claude Code configuration (applied only after approval)

- **Agents:** `.claude/agents/` holds the 11 agents in §2. Each has model, tools, charter, prohibitions and a required output format.
- **New skills:**

  | Skill | What it does |
  |---|---|
  | `/research-cycle` | Orchestrator; drives S0–S9 with checkpoints A, B and C |
  | `/scout` | Idea sweep, on demand only |
  | `/validate <system>` | Statistician + red team on an existing study |
  | `/unlock-holdout <system>` | Checkpoint B |
  | `/promote <system>` | Checkpoint C, then the MQL5/ONNX port |
  - `/decay-review <system>`: alpha-decay check on newer data (§4.6)
- **Existing commands:** all retired. `implement-strategy` is replaced by `/research-cycle`. The other six were stale copies of AlphaForge commands; the up-to-date versions live in `~/Finances/AlphaForge/.claude/commands/`.

---

## 10. Build order

| Phase | Deliverable | Exit test |
|---|---|---|
| 0 | `quantlab.data` (timezones, holdout guard, calendar), `ledger`, FBS `costs` (needs the MT5 export), `engine` | Look-ahead **truncation test** (signals computed on data up to t = signals on full data, at t); known-answer backtests; hand-checked trades |
| 1 | `stats` + `opt` | **Validate the validator:** a zero-edge synthetic system fails the gates ≥ 95% of the time; a planted edge passes |
| 2 | Agents, skills, `report`, card generator | Dry run on a toy system |
| 3 | **Pilot:** run 2–3 old strategies (SMA crossover, Donchian) through the full pipeline | Pipeline runs end to end. Old strategies probably get killed, which is fine. |
| 4 | First Scout batch → real research | — |
| 5 | Portfolio manager, once ≥ 2 systems pass in a book | — |
| 6 | B3 book: cost and tax model, intraday session rules | — |

---

## 11. Decisions log

| # | Item | Decision (2026-09-23) |
|---|---|---|
| 1 | Kill criterion | Fails the §4.2 gates after 3 attempts |
| 2 | Gate thresholds | Accepted as starting values; recalibrate in Phase 1 |
| 3 | Re-optimisation | Periodic re-fit schedule validated as a procedure, plus alpha-decay review and retirement (§4.6) |
| 4 | Units / account | Points first; 100,000 nominal account when % equity is needed (§5) |
| 5 | Commands | Old commands retired; new skills and agents as in §9 |
| 6 | Naming | `quantlab/`, `research/systems/<book>/…` |
| 7 | Gate recalibration (2026-09-24) | Adopted R1 (DSR hurdle from null variance + raw trial count) and R2 (CSCV P(OOS loss) < 0.10 replaces PBO < 0.30), plus the red-team fixes: WFO procedure gate, judge-computed plateau, 1-pip stress slippage on all fills, joint holdout band. See `research/audits/2026-09-24_phase1_fix_plan.md` |
| 8 | Open (2026-09-24) | WFO-gate thresholds, holdout length / renewing-holdout policy (band power), time-stability replacement (R3): pending the recalibration results |
