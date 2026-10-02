"""S5 pre-check (read-only): the gates' own identity guards on the read-only StudyResult."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from s5_load_study import SID, load_study, make_evaluator  # noqa: E402

from quantlab import gates, ledger  # noqa: E402

if __name__ == "__main__":
    n_before = len(ledger.study_events(SID))
    study = load_study()
    ev = make_evaluator()
    ctx = gates.ledger_context(study)
    store = gates.verify_trial_store(study)
    ident = gates.check_evaluator_identity(SID, ev, None)
    print("ctx:", {k: ctx[k] for k in ("book", "system", "issue", "attempt", "method", "plateau_radius",
                                       "data_dependent")})
    print("store: n_store", store["n_store"], "n_ledger", store["n_ledger"], "tpe", store["tpe_in_store"],
          "dir", store["studies_dir"])
    print("identity cost version:", ident["cost_model_version"])
    print("selected:", study.selected_params, "trial", gates.selected_trial_id(study))
    print("ledger rows before/after:", n_before, len(ledger.study_events(SID)))
    print("prior gate runs:", len(ledger.study_events(SID, "gates")))
    print("related prior trials:", ledger.related_prior_trials(SID))
