"""R1-misc — m3 (embargo cap -> OOS gates SKIPPED), m4 (dsr rejects annualised SR), m9 (repeat gate
runs flagged), and the deprecated prior_effective_trials alias (an old-style effective-N prior is
now taken as the RAW prior and bypasses the ledger)."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from r1_common import ExploitEval, tmpdirs, mech_true
from quantlab import opt, gates as G, stats as st
from quantlab.evaluators import SyntheticEvaluator

# m4
try:
    st.dsr(sr=1.2, n=2000, var_sr=1e-4, n_eff=10); print("m4: annualised sr accepted (BAD)")
except ValueError as e:
    print("m4: dsr raises on |sr|>1:", str(e)[:60])
# m3: long holds -> embargo capped
class LongHold(ExploitEval):
    def __call__(self, params, *, cost=None):
        o = super().__call__(params, cost=cost); o.metrics["hold_days_max"] = 400.0; return o
space = opt.SearchSpace([opt.IntParam("a", 0, 4), opt.IntParam("b", 0, 4)])
inner = SyntheticEvaluator(bounds={"a": (0, 4), "b": (0, 4)}, bumps=({"center": {"a": 2, "b": 2}, "height": 3.0, "width": .5},), rho=.5, seed=2)
ld, sd = tmpdirs()
res = opt.run_study(LongHold(inner), space, book="FBS", system="m3", issue=9999, attempt=1, study_id="m3-a1", n_jobs=1,
                    ledger_dir=ld, studies_dir=sd)
rep = G.evaluate_gates(res, LongHold(inner), periods_per_year=260, mechanism_check=mech_true, n_boot=200, ledger_dir=ld)
print("m3: embargo_capped", res.meta["embargo_capped"], "->", {g: rep.statuses()[g] for g in ("oos_sharpe", "cscv_oos_loss", "wfo_oos")}, "verdict", rep.verdict)
# m9
G.log_gates(rep, ledger_dir=ld); G.log_gates(rep, ledger_dir=ld)
rep2 = G.evaluate_gates(res, LongHold(inner), periods_per_year=260, mechanism_check=mech_true, n_boot=200, ledger_dir=ld)
print("m9: prior gate runs visible:", len(rep2.prior_gate_runs), "| warning in markdown:", "gated 2 time(s) before" in rep2.to_markdown())
# deprecated alias
ld2, sd2 = tmpdirs()
for a in (1, 2):
    r = opt.run_study(ExploitEval(inner), space, book="FBS", system="alias", issue=9999, attempt=a, study_id=f"al-a{a}",
                      n_jobs=1, wfo=None, ledger_dir=ld2, studies_dir=sd2)
    if a == 1:
        G.log_gates(G.evaluate_gates(r, ExploitEval(inner), periods_per_year=260, n_boot=100, ledger_dir=ld2), ledger_dir=ld2)
from quantlab import ledger
eff = ledger.system_effective_trials("FBS", "alias", exclude_study="al-a2", ledger_dir=ld2)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    ra = G.evaluate_gates(r, ExploitEval(inner), periods_per_year=260, n_boot=100, ledger_dir=ld2, prior_effective_trials=eff)
rl = G.evaluate_gates(r, ExploitEval(inner), periods_per_year=260, n_boot=100, ledger_dir=ld2)
print(f"alias: v1.1-style caller passes prior_effective_trials={eff:.2f} -> prior used {ra.effective_trials['n_trials_prior']:.2f} "
      f"(ledger raw would be {rl.effective_trials['n_trials_prior']:.0f}); DSR {ra.values()['dsr']:.3f} vs {rl.values()['dsr']:.3f}; "
      f"warnings: {[type(x.message).__name__ for x in w][:3]}")
