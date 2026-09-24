---
name: unlock-holdout
description: Checkpoint B — the user's one-shot unlock of the locked 12-month holdout for a validated system, run the frozen procedure on it, and judge it against the pre-registered band. User-invoked only.
argument-hint: "<system-slug>"
disable-model-invocation: true
---

# Unlock the holdout (one shot)

The user invoked this; that is the authorisation. Do **not** run it for any other system than `$ARGUMENTS`.

1. **Pre-flight (refuse if any fails):**
   - System passed S5 with no open BLOCKER from the red team.
   - The validation report contains a **pre-registered holdout pass band** (DESIGN §4.4).
   - `research/ledger/holdout_access.jsonl` has **no** prior unlock for this system.
   - Working tree clean for the system's code; record `git_commit`.
   Show the user the frozen spec (params / re-optimisation procedure, commit, band) and ask them to
   **type the exact phrase** `UNLOCK HOLDOUT <BOOK>/<system>` (`quantlab.ledger.unlock_phrase`).
   Never type, suggest-and-accept, or compose the phrase yourself; if the user doesn't type it, stop.
2. Write the unlock row via `quantlab.ledger.record_holdout_unlock(..., user_confirmation=<the user's text verbatim>)`; this is
   what makes `quantlab.data` serve holdout data for this system.
3. Delegate to **optimization-architect** only to *execute* the frozen procedure over the holdout
   (scheduled re-fits included, DESIGN §4.6) — no design changes, no reruns.
4. Delegate to **validation-statistician** to judge against the band → PASS / FAIL.
5. Delegate to **report-builder** to fill notebook §8 and update the card (holdout z-score).
6. FAIL ⇒ label `killed`, close the issue. PASS ⇒ ask **portfolio-risk-manager** for the S9 book
   review, then present everything to the user for **checkpoint C** (`/promote`).
