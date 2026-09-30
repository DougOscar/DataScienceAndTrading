"""Notebook tear sheet + Obsidian card generator (DESIGN §7, §6; report-builder agent).

Public API (fixed; the notebook template calls exactly this, DESIGN §10 Phase 2)::

    ts = report.tear_sheet(study_id, *, evaluator=None, ledger_dir=None, studies_dir=None,
                           risk_type=None, holdout=None)          # -> TearSheet
    ts.headline / ts.metrics / ts.applicability / ts.interpretations / ts.gate_table
    ts.figures(out_dir)            # -> {name: Path}, PNGs
    ts.to_markdown()               # -> str
    ts.write_results(results_dir)  # metrics.json + figures/ + tear_sheet.md

    report.write_card(ts, *, slug, name, idea, status, stage=None, reason=None, issue, book,
                      vault_dir=None) -> Path
    report.risk_profile_label(daily, trades) -> (label, evidence)

Gate *values* are always the ones the validation-statistician logged to the ledger
(``evaluate_gates(..., log=True)``'s ``gates`` event) — this package never recomputes a gate.
"""

from __future__ import annotations

from ._card import risk_profile_label, write_card
from .tear_sheet import TearSheet, tear_sheet

__all__ = ["tear_sheet", "TearSheet", "write_card", "risk_profile_label"]
