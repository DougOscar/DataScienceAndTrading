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
| S1 power check | validation-statistician | MinTRL feasible → continue; else **KILL-EARLY** recommendation |
| S2 implement | strategy-engineer → red-team (code audit) | tests pass, no BLOCKER; relabel issue `testing`; create `research/systems/<book>/<issue#>_<slug>/` with `hypothesis.md` |
| S3 baseline | strategy-engineer | baseline with prior params, costs on; gross vs cost edge reported |
| S4 optimise | optimization-architect | study logged in ledger, plateau selection, WFO re-fit schedule simulated |
| S5 validate | validation-statistician | gate table + pre-registered holdout band |
| S6 attack | red-team | findings file; BLOCKER ⇒ fix via strategy-engineer and repeat S3–S6 (not an "attempt" if it is a bug fix; say so explicitly) |
| S7 report | report-builder | tear sheet + draft Obsidian card |

**Improvement loop.** If S5 fails, you may propose an improvement (new logged variant, parent =
previous study) — max **3 attempts** per system. Each attempt goes S2→S7 again and counts in the
ledger. After the 3rd failure: label `killed`, close the issue, report-builder writes the card
with stage + reason. Bug fixes found by the red team are not attempts; parameter/rule changes are.

## Checkpoints — always stop and hand control to the user
- **After S7 with a PASS**: present the tear-sheet summary and ask whether to unlock the holdout.
  The user runs `/unlock-holdout <system>`; you never unlock it.
- **Any kill**: summarise why (gate values), then close with label `killed`.

## Hand-off discipline
- Give each agent: the issue number, the system folder, the study_id(s), and the exact DESIGN
  sections that govern the stage. Ask for their standard final report.
- After each agent returns, verify the claimed artefacts exist (files, ledger rows, tests passing)
  before moving on. Post a short progress comment on the issue (`gh issue comment`).
- Keep the user informed with one line per stage transition; don't paste agent reports verbatim.
