"""S5 (validation-statistician), dry run #23: read-only StudyResult loader for fbs-0023-a1.

There is no public read-only loader (FINDINGS #34). This rebuilds the StudyResult from the
trial store (ledger.load_trials / load_trial_returns / load_study_artifact) and the ledger's
study_created + selection rows, WITHOUT writing anything. The gates' own guards
(verify_trial_store, ledger_context, check_evaluator_identity) are the acceptance test.
Run from the repo root with research/dryrun/env.sh sourced.
"""
from __future__ import annotations

import json

import polars as pl

from quantlab import ledger, opt
from quantlab.contracts import StudyResult
from quantlab.evaluators import RuleEvaluator
from quantlab.strategies.toy_tsmom import ToyTsmom

SID = "fbs-0023-a1"


def make_evaluator() -> RuleEvaluator:
    # Same as S3/S4: M1 intrabar (default for H4), default FBS cost model.
    return RuleEvaluator(ToyTsmom, "EURUSD", "H4", book="FBS")


def load_study(sid: str = SID) -> StudyResult:
    row = ledger.created_row(sid)
    sel = ledger.study_events(sid, "selection")[-1]
    space = opt.space_from_json(row["search_space"])
    t = ledger.load_trials(sid).sort("trial_id")
    pj = [json.loads(s) for s in t["params"].to_list()]
    t = t.with_columns([pl.Series(f"param_{n}", [p.get(n) for p in pj]) for n in space.names])
    r = ledger.load_trial_returns(sid)
    ids = sorted(int(c[1:]) for c in r.columns if c != "date")
    r = r.select(pl.col("date").cast(pl.Date), *[pl.col(f"t{i}").fill_null(0.0) for i in ids])
    art = {n: ledger.load_study_artifact(sid, n) for n in ledger.STUDY_ARTIFACTS}
    meta = {
        "study_id": sid, "book": row["book"], "system": row["system"], "issue": row["issue"],
        "attempt": row["attempt"], "seed": row["seed"], "method": row["method"],
        "candidate_set": row["candidate_set"], "space": row["search_space"],
        "plateau_radius": row["plateau_radius"],
        "candidate_set_data_dependent": bool(sel["candidate_set_data_dependent"]),
        "embargo_capped": bool(sel["embargo_capped"]),
        "cpcv": {"embargo_capped": bool(sel["embargo_capped"]), "n_paths": sel["cpcv_paths"],
                 "embargo_days": sel["cpcv_embargo_days"], "purge_days": sel["cpcv_purge_days"]},
        "cv_scheme": sel["cv_scheme"], "n_trials": int(t.height), "search_space_obj": space,
        "trade_counts": art["trade_counts"], "entry_counts": art["entry_counts"],
        "symbols": row.get("symbols"), "cost_model_version": row["cost_model_version"],
        "loaded_read_only": True,
    }
    return StudyResult(study_id=sid, param_names=space.names, trials=t, returns=r,
                       selected_params=sel["selected_params"], selection=sel["selection"],
                       cpcv_paths=art["cpcv_paths"], wfo_oos=art["wfo_oos"], wfo_params=art["wfo_params"],
                       meta=meta)
