"""S5 official gate run (validation-statistician), dry run #23, study fbs-0023-a1.

Runs gates.evaluate_gates(..., log=True) ONCE on the read-only StudyResult (s5_load_study).
Mechanism gate left MANUAL (mechanism_check=None): the user decides on the ablation evidence.
Refuses to run if a gates event already exists for the study (single official run).
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from s5_load_study import SID, load_study, make_evaluator  # noqa: E402

from quantlab import gates, ledger  # noqa: E402

OUT = Path(__file__).parent


def _jsonable(o):
    try:
        json.dumps(o)
        return o
    except TypeError:
        if isinstance(o, dict):
            return {str(k): _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonable(v) for v in o]
        return repr(o)


if __name__ == "__main__":
    if ledger.study_events(SID, "gates"):
        raise SystemExit(f"{SID} already has a gates event; the official S5 run is done — not re-running")
    study = load_study()
    ev = make_evaluator()
    rep = gates.evaluate_gates(study, ev, periods_per_year=ev.periods_per_year, n_jobs=8, log=True)
    rep.write_markdown(OUT / "s5_gate_report.md")
    dump = {"verdict": rep.verdict, "rows": [r.__dict__ for r in rep.rows], "holdout_band": rep.holdout_band,
            "effective_trials": rep.effective_trials, "diagnostics": rep.diagnostics,
            "ledger_context": rep.ledger_context, "ledger_row_seq": (rep.ledger_row or {}).get("seq"),
            "periods_per_year": rep.periods_per_year}
    (OUT / "s5_gate_report.json").write_text(json.dumps(_jsonable(dump), indent=1, default=repr))
    print("VERDICT", rep.verdict)
    for r in rep.rows:
        print(f"{r.gate:20s} {r.display if r.display is not None else r.value!s:30s} {r.threshold:28s} {r.status}")
