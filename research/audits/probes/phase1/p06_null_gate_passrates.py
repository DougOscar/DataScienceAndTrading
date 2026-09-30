"""P06 — Per-gate pass rates for a ZERO-edge family run through the full pipeline
(run_study grid + evaluate_gates), with and without adversarial grid design.

Family A ("plain null"): 10x10 grid, flat 0 surface, trial correlation rho=0.8.
Family B ("padded null"): same, but configs with a >= 6 (40% of the grid) bleed
  -1.0 annual Sharpe (think: fast SMA pairs that churn through the spread).
The common factor is re-drawn per seed; its realised drift is what all trials share.
We report the share of seeds in which each gate PASSES.  A calibrated gate should pass
a zero-edge system rarely (DSR: <= 5% by construction).
"""
import sys
import numpy as np
from quantlab import opt
from quantlab.evaluators import SyntheticEvaluator
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from e2e_common import ExploitEval, run_and_gate

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
space = opt.SearchSpace([opt.IntParam("a", 0, 9), opt.IntParam("b", 0, 9)])
fams = {
    "A_plain_null": (),
    "B_padded_null": ({"center": {"a": 9, "b": 4.5}, "height": -1.0, "width": 0.34, "kind": "box"},),
}
for fam, bumps in fams.items():
    rows = []
    for seed in range(NS):
        inner = SyntheticEvaluator(bounds={"a": (0, 9), "b": (0, 9)}, bumps=bumps, rho=0.8, seed=500 + seed,
                                   trades_per_year=100)
        res, rep = run_and_gate(ExploitEval(inner), space, f"fbs-9999-{fam}-{seed}", wfo=None)
        rows.append((rep.statuses(), rep.values(), rep.verdict, rep.effective_trials))
    gates = [g for g in rows[0][0]]
    print(f"== {fam}: {NS} zero-edge worlds")
    for g in gates:
        print(f"   {g:20s} pass rate {np.mean([r[0][g] == 'PASS' for r in rows]):.2f}")
    print(f"   verdict PASS rate {np.mean([r[2] == 'PASS' for r in rows]):.2f}")
    print(f"   median n_eff used {np.median([r[3]['n_eff'] for r in rows]):.2f} of 100 raw; "
          f"median sr0 hurdle (annual) {np.median([r[3]['sr0_annual'] for r in rows]):.2f}; "
          f"median PBO {np.median([r[1]['pbo'] for r in rows]):.2f}; median plateau {np.median([r[1]['plateau'] for r in rows]):.2f}")
    # conditional: among seeds whose realised common drift made the selected Sharpe > 0
    pos = [r for r in rows if r[3]["sr_annual"] > 0]
    if pos:
        print(f"   among {len(pos)} worlds with selected in-sample Sharpe > 0: DSR pass {np.mean([r[0]['dsr']=='PASS' for r in pos]):.2f}, "
              f"PBO pass {np.mean([r[0]['pbo']=='PASS' for r in pos]):.2f}, plateau pass {np.mean([r[0]['plateau']=='PASS' for r in pos]):.2f}")
