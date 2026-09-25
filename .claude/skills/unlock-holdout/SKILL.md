---
name: unlock-holdout
description: Checkpoint B — the user's unlock of the holdout (locked year + all newer data) for a validated system, run the frozen procedure on it, and judge it against the pre-registered band → PASS / FAIL / NOT_DECISIVE. Also used for the re-exam of a NOT_DECISIVE system once more data exists. User-invoked only.
argument-hint: "<system-slug>"
disable-model-invocation: true
---

# Unlock the holdout (DESIGN §4.4)

The user invoked this; that is the authorisation. Do **not** run it for any other system than `$ARGUMENTS`.

**Horizon rule.** The exam covers the holdout start (FBS 2025-05-15) **through the end of the data
available now**: the locked year plus every newer export (the renewing holdout, DESIGN §4.1). The
band must be built for exactly that horizon **before** the unlock. The horizon comes from the data
manifest (`quantlab.data.holdout_horizon`, metadata only; it never reads holdout bars).

**Verdicts.** `PASS` (all criteria pass and the band is decisive) · `FAIL` (any criterion fails) ·
`NOT_DECISIVE` (all criteria pass, but the band's P(pass | zero edge) > 0.30, so this horizon cannot
tell a real edge from a dead one). The 0.30 limit is `quantlab.stats.DECISIVE_MAX_ZERO_EDGE_PASS`
and is read-only.

1. **Pre-flight (refuse if any fails):**
   - System passed S5 with no open BLOCKER from the red team.
   - `quantlab.ledger.holdout_state(book, system)`:
     - `killed` (an earlier FAIL) → **stop**. A FAIL is final; there is no re-exam.
     - `passed` → stop; the holdout is done (go to `/promote`).
     - `pending` (unlocked, no exam recorded) → stop and finish that exam (steps 4–6) instead.
     - `last_status == "NOT_DECISIVE"` → this is a **re-exam**. It is allowed only if data newer than
       the last exam's `last_horizon_end` has been exported. Otherwise stop: the system keeps waiting
       (`holdout_pending`).
   - The study has a **registered band** (`quantlab.ledger.registered_holdout_band(study_id)`) with
     `horizon_source == "manifest"`.
   - **Band current?** Compare the band's `horizon_end` with `quantlab.data.holdout_horizon(book, band["symbols"])["horizon_end"]`.
     If the manifest now ends later (newer data was exported), or this is a re-exam, delegate to
     **validation-statistician** to run `quantlab.gates.rebuild_holdout_band(study, periods_per_year=…, reason="<what was exported>")`
     **before anything else**. It rebuilds the band for the new horizon with the same construction
     and seed, and logs a `holdout_band_registered` event (new `band_version`, the horizon it
     replaces, and the reason). Show the user the new band and its `p_pass_zero_edge`. No one may
     read the new data before this step: until the unlock, `quantlab.data` refuses holdout reads,
     and after a NOT_DECISIVE exam it serves holdout bars only up to the last exam's horizon.
   - Working tree clean for the system's code; record `git_commit`.

   Show the user the frozen spec (params / re-optimisation procedure, commit), the band, its horizon
   (`horizon_days`, `holdout_start` → `horizon_end`, `newer_data_included`), its `p_pass_zero_edge`,
   and whether an all-criteria pass would be decisive. Then ask them to **type the exact phrase**
   `UNLOCK HOLDOUT <BOOK>/<system>` (`quantlab.ledger.unlock_phrase`). The phrase is the same for a
   re-exam. Never type, suggest-and-accept, or compose the phrase yourself; if the user doesn't type
   it, stop.
2. Write the unlock row with `quantlab.ledger.record_holdout_unlock(..., pass_band=<the registered band>, user_confirmation=<the user's text verbatim>)`.
   The ledger refuses the unlock if the band is not the registered one, if the band is stale versus
   the manifest, if the horizon doesn't come from the manifest, after a FAIL or a PASS, while an exam
   is pending, or (for a re-exam) if the band doesn't reach past the last exam's horizon. The row
   records `exam` (1, 2, …) and the horizon. This is what makes `quantlab.data` serve holdout data
   for this system, up to the band's `horizon_end`.
3. Delegate to **optimization-architect** only to *execute* the frozen procedure over the **full**
   holdout span, from the holdout start to the unlock's `horizon_end` (scheduled re-fits included,
   DESIGN §4.6). A re-exam also re-runs the full span, not just the new data. No design changes, no
   reruns.
4. Delegate to **validation-statistician** to judge the exam with `quantlab.gates.run_holdout_exam(book=…, system=…, study_id=…, holdout_daily=…, n_trades=…, periods_per_year=…)`.
   It uses the band stored with the unlock row, returns `status` ∈ PASS / FAIL / NOT_DECISIVE, and
   appends a `holdout_exam` event with its horizon (in `holdout_access.jsonl`, mirrored on the study).
5. Delegate to **report-builder** to fill notebook §8 and update the card: verdict, holdout z-score,
   horizon, exam number and `p_pass_zero_edge`.
6. Act on the verdict:
   - **FAIL** ⇒ label `killed` and close the issue. There are no re-tries and no re-exams: the ledger
     refuses any later band, unlock or exam for this system.
   - **NOT_DECISIVE** ⇒ label `holdout_pending` and leave the issue open. This is neither a pass nor a
     fail, and the system **cannot be promoted**. Tell the user roughly how much more data would make
     the band decisive, using the calibration: 1 year is decisive for ~55% of true edges and 2 years
     for ~82% (`research/calibration/2026-09-24_phase1_calibration_v12.md` §8.2). Re-run this skill
     after the next data export. That run is a new exam on the full, longer span, with a band rebuilt
     *before* the new data is visible. It is not a retry: the same data is never re-used to overturn
     a FAIL.
   - **PASS** ⇒ ask **portfolio-risk-manager** for the S9 book review, then present everything to the
     user for **checkpoint C** (`/promote`).
