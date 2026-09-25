"""R2-P03 (48e696f) -- the gates do not check that the evaluator passed to evaluate_gates is the study's
(ledger row 'evaluator' / cost_model_version).  Study run on a SPIKY surface; gates given a BROAD-surface,
cheaper evaluator.  Judge plateau + cost stress are then measured on a different system."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_common import ExploitEval, tmpdirs, mech_true
from quantlab import opt, gates as G, ledger
from quantlab.evaluators import SyntheticEvaluator

sp = opt.SearchSpace([opt.IntParam("a", 1, 101, 5)])
spiky = ExploitEval(SyntheticEvaluator(bounds={"a": (1, 101)}, bumps=({"center": {"a": 51}, "height": 2.5, "width": 0.02},),
                                       base_sharpe=0.3, rho=1.0, seed=3))
broad = ExploitEval(SyntheticEvaluator(bounds={"a": (1, 101)}, bumps=({"center": {"a": 51}, "height": 2.5, "width": 0.5},),
                                       base_sharpe=0.3, rho=1.0, seed=3), drag_per_unit=0.0)
ld, sd = tmpdirs()
res = opt.run_study(spiky, sp, book="FBS", system="swap", issue=88, attempt=1, study_id="sw-a1", n_jobs=1, wfo=None,
                    ledger_dir=ld, studies_dir=sd)
for name, e in (("study's own evaluator", spiky), ("swapped broad + zero-drag evaluator", broad)):
    rep = G.evaluate_gates(res, e, periods_per_year=260, mechanism_check=mech_true, n_boot=100, ledger_dir=ld)
    d = rep.diagnostics["plateau"]
    print(f"{name:38s}: plateau {rep.row('plateau').value:.2f} {rep.row('plateau').status}; peak re-evaluated "
          f"{d['peak_sharpe']:.2f} vs study {d['peak_sharpe_study']:.2f}; cost_stress {rep.row('cost_stress_sharpe').value:.2f} "
          f"{rep.row('cost_stress_sharpe').status}")
print("ledger evaluator description:", ledger.created_row("sw-a1", ledger_dir=ld).get("evaluator", {}).get("bumps"))
