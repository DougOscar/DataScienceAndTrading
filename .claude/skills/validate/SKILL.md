---
name: validate
description: Re-run independent validation (validation-statistician gates + red-team attack) on an existing study or system without re-optimising — e.g., after a cost-model version bump, a data correction, or when a result looks too good.
argument-hint: "<system-slug | study_id>"
---

# Validate a study

1. Resolve `$ARGUMENTS` to the system folder and latest study row in `research/ledger/studies.jsonl`.
2. In parallel, delegate to **validation-statistician** (full gate table, DESIGN §4.2) and
   **red-team** (attack checklist). Both write to the system's `results/` folder.
3. Summarise for the user: verdict, any gate that changed vs the previous validation and why,
   BLOCKER/MAJOR findings. No re-optimisation, no rule changes — if they are needed, that is a new
   attempt under `/research-cycle` and counts toward the 3-attempt limit.
