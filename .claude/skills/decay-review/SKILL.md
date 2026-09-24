---
name: decay-review
description: Alpha-decay review (DESIGN §4.6) of promoted systems on data newer than their holdout — rolling performance vs predicted band, CUSUM/sequential Sharpe test, scheduled re-fits, parameter drift — and retirement decisions. User-invoked only.
argument-hint: "<system-slug | book: fbs|b3 | all>"
disable-model-invocation: true
---

# Alpha-decay review

1. Ask **data-auditor** to audit any data newer than the last review (new MT5 exports converted
   with `tools/mt5_csv_to_parquet.py`).
2. For each system in scope: **optimization-architect** executes the frozen re-optimisation
   procedure forward over the new data (no redesign); **validation-statistician** compares against
   the predicted band and runs the CUSUM / sequential test; flags parameter drift.
3. Retirement rule: 6 consecutive months below the predicted 10th percentile **or** a CUSUM alarm
   ⇒ recommend `retired`. The user decides; on retirement, **portfolio-risk-manager** re-reviews the
   book and **report-builder** updates the card.
4. Summarise per system: status (healthy / watch / retire), evidence, and book impact.
