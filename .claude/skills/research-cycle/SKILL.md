---
name: research-cycle
description: Orchestrates the research pipeline (DESIGN.md §3) for one or more approved hypotheses — power check, implementation, baseline, optimisation, validation, red team, improvement loop (max 3), report — by delegating to the specialist subagents and stopping at the user's checkpoints. Use when the user asks to research/test/evaluate a hypothesis issue or to "run the research cycle".
argument-hint: "<issue-number> [<issue-number> ...]"
---

# Research cycle

You (the main session) are the **Research Director**. You never do a specialist's job yourself;
you delegate with the Agent tool, check each hand-off against `research/DESIGN.md`, keep the
GitHub issue and ledger in sync, and stop at checkpoints.

Input: `$ARGUMENTS` — hypothesis issue number(s). Read `research/DESIGN.md` first.

## Preconditions
- Each issue carries label `hypothesis` **and the user approved it (checkpoint A)**. If approval is
  not explicit in this conversation or on the issue, stop and ask.
- `quantlab/` supports what the stage needs. If a capability is missing, route it to the right
  agent as a library task (with tests) before continuing; don't improvise in notebooks.

## Stages (per issue; independent issues may run in parallel)
| Stage | Agent | Done when |
|---|---|---|
| S1 power check | validation-statistician | `stats.power_check` on the card's expected Sharpe and trades/yr: MinTRL feasible → continue; else **KILL-EARLY** recommendation |
| S2 implement | strategy-engineer → red-team (code + card audit) | you create the folder with `quantlab.systems.create_system(book, issue, slug, name=…, card_markdown=<approved card>)` → `research/systems/<book>/<issue:04d>_<slug>/`; strategy in `quantlab/strategies/<slug>.py`; tests pass incl. exhaustive `assert_no_lookahead(StrategyClass, …)`, `assert_engine_causal(…, timeframe=…)` and `assert_strategy_source_clean`; red team reviewed the card's pre-registered parameter ranges, plateau scales/radius and mechanism ablation; no BLOCKER; relabel issue `testing` |
| S3 baseline | strategy-engineer | baseline with prior params, costs on; gross vs cost edge reported |
| S4 optimise | optimization-architect | `opt.run_study(…, issue=<issue>, attempt=<n>, method="auto", plateau_radius=<card>)` logged in the ledger (grid ≤ 3 params, else Sobol; never TPE for a study to be validated); plateau selection; WFO re-fit schedule simulated |
| S5 validate | validation-statistician | `gates.evaluate_gates(…, log=True)` (the only writer of gate results; registers the holdout band); gate table. A MANUAL mechanism gate: show the user the ablation evidence and record their decision with `ledger.record_mechanism_review(study_id, passed=…, note=…)` |
| S6 attack | red-team | findings file; BLOCKER ⇒ fix via strategy-engineer and repeat S3–S6 (not an "attempt" if it is a bug fix; say so explicitly). Changed strategy code or cost model means a new study (the gates refuse the old one; `run_study(study_id=…, parent_study=…)`, since the attempt id is taken); its trials still count in the DSR |
| S7 report | report-builder | `report.tear_sheet(study_id, …)` + draft card via `report.write_card(…)` |

**Improvement loop.** If S5 fails (FAIL, or INCOMPLETE that cannot be completed), you may propose an improvement (new logged variant, parent =
previous study) — max **3 attempts** per system. Each attempt goes S2→S7 again and counts in the
ledger (`attempt` 1–3; every attempt's trials raise the DSR's N for the next). After the 3rd failure: label `killed`, close the issue, report-builder writes the card
with stage + reason. Bug fixes found by the red team are not attempts; parameter/rule changes are.

## Checkpoints — always stop and hand control to the user
- **After S7 with a PASS**: present the tear-sheet summary and ask whether to unlock the holdout.
  The user runs `/unlock-holdout <system>`; you never unlock it. The exam covers the locked year plus
  all newer data, with a band rebuilt beforehand if more data was exported. It ends PASS, FAIL
  (killed) or NOT_DECISIVE (`holdout_pending`: wait for more data, then re-examine). DESIGN §4.4.
- **Any kill**: summarise why (gate values), then close with label `killed`.

## Hand-off discipline
- Give each agent: the issue number, the system folder, the study_id(s), and the exact DESIGN
  sections that govern the stage. Ask for their standard final report.
- After each agent returns, verify the claimed artefacts exist (files, ledger rows, tests passing)
  before moving on. Post a short progress comment on the issue (`gh issue comment`).
- Keep the user informed with one line per stage transition; don't paste agent reports verbatim.
