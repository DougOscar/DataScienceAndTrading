---
name: validate
description: Re-run independent validation (validation-statistician gates + red-team attack) on an existing study or system without re-optimising — e.g., after a data correction or when a result looks too good. (After a cost-model version bump or a strategy-code change the gates refuse the old study; it must be re-run.)
argument-hint: "<system-slug | study_id>"
---

# Validate a study

1. Resolve `$ARGUMENTS` to the system folder and latest study row in `research/ledger/studies.jsonl`.
2. In parallel, delegate to **validation-statistician** (full gate table, DESIGN §4.2 v1.2: load the
   study read-only with `opt.load_study(study_id)`, then
   `quantlab.gates.evaluate_gates(…, log=True, rerun_reason="<why this re-validation>")` — a study
   that S5 already gated refuses a second logged run without `rerun_reason`) and **red-team**
   (attack checklist; every extra evaluation logged with `ledger.log_diagnostic`). Both write to
   the system's `results/` folder. A re-run is logged as a new, flagged gate run on the study; it
   never replaces the registered holdout band (only `gates.rebuild_holdout_band` does, for newer data).
   The gates refuse (`GateError`) a study whose cost-model version or strategy source changed since
   it ran: after a cost-model bump or a code fix the study must be re-run (`/research-cycle` S4, new
   study id), and its trials count in the DSR.
3. Summarise for the user: verdict, any gate that changed vs the previous validation and why,
   BLOCKER/MAJOR findings. No re-optimisation, no rule changes — if they are needed, that is a new
   attempt under `/research-cycle` and counts toward the 3-attempt limit.
