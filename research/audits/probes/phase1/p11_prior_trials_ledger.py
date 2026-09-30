"""P11 — cross-attempt deflation plumbing.
(a) evaluate_gates does NOT read the ledger: prior_effective_trials defaults to 0, so a caller
    that forgets it gets a DSR with zero cross-attempt deflation (no warning).
(b) log_gates logs effective_trials = n_eff INCLUDING the prior it was given, and
    system_effective_trials sums those logged values -> attempt k double-counts attempts < k-1."""
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/douglaso/Finances/DataScienceAndTrading/tests")
from test_gates import make_study, FakeEvaluator, _mech_ok
from quantlab import gates as G, ledger

tmp = Path(tempfile.mkdtemp(prefix="rt_p11_"))
for att in (1, 2, 3):
    sid = f"fbs-0099-a{att}"
    ledger.create_study(ledger_dir=tmp, study_id=sid, book="FBS", system="sys", issue=99, attempt=att,
                        parent_study=None, dev_window=["2016-05-02", "2025-05-14"], cost_model_version="t",
                        cv_scheme="x", search_space={}, n_trials_planned=25) if True else None
    study, trades = make_study(seed=att)
    study.study_id = sid
    prior = ledger.system_effective_trials("FBS", "sys", exclude_study=sid, ledger_dir=tmp)
    rep_forgot = G.evaluate_gates(study, FakeEvaluator(study, trades), periods_per_year=260, selected_trades=trades,
                                  mechanism_check=_mech_ok)
    rep = G.evaluate_gates(study, FakeEvaluator(study, trades), periods_per_year=260, selected_trades=trades,
                           mechanism_check=_mech_ok, prior_effective_trials=prior)
    own = rep.effective_trials["n_eff"] - prior
    G.log_gates(rep, sid, ledger_dir=tmp)
    print(f"attempt {att}: this study's own n_eff {own:.2f}; prior read from ledger {prior:.2f}; "
          f"n_eff used {rep.effective_trials['n_eff']:.2f}; DSR with prior {rep.values()['dsr']:.4f} vs forgot-prior {rep_forgot.values()['dsr']:.4f}")
