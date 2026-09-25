"""R1-P06 (B1, M1, M5 end-to-end) — per-gate pass rates of zero-edge families through run_study +
evaluate_gates (v1.2 gates), 10x10 grid, rho=0.8, default WFO.
A plain null; B padded null (40 % of grid bleeds -1.0 Sharpe: the M1 PBO exploit);
C 'drifting null': every config shares a +0.6 base Sharpe common to the whole grid is NOT null -
   skipped; instead C = plain null with rho=0.95 (very correlated grid)."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

FAMS = {
    "A_plain_null": ((), 0.8),
    "B_padded_null": (({"center": {"a": 9, "b": 4.5}, "height": -1.0, "width": 0.34, "kind": "box"},), 0.8),
    "C_rho095_null": ((), 0.95),
}

def one(job):
    import warnings; warnings.filterwarnings("ignore")
    from r1_common import ExploitEval, run_and_gate
    from quantlab import opt
    from quantlab.evaluators import SyntheticEvaluator
    fam, seed = job
    bumps, rho = FAMS[fam]
    space = opt.SearchSpace([opt.IntParam("a", 0, 9), opt.IntParam("b", 0, 9)])
    inner = SyntheticEvaluator(bounds={"a": (0, 9), "b": (0, 9)}, bumps=bumps, rho=rho, seed=500 + seed,
                               trades_per_year=100)
    res, rep = run_and_gate(ExploitEval(inner), space, f"fbs-9999-r1-{fam}-{seed}")
    return fam, rep.statuses(), rep.verdict, rep.effective_trials["sr_annual"], rep.diagnostics.get("pbo_detail", {}).get("pbo")

if __name__ == "__main__":
    from r1_common import pmap
    NS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    rows = pmap(one, [(f, s) for f in FAMS for s in range(NS)])
    for fam in FAMS:
        rs = [r for r in rows if r[0] == fam]
        gates = list(rs[0][1])
        print(f"== {fam}: {len(rs)} zero-edge worlds; verdict PASS {np.mean([r[2] == 'PASS' for r in rs]):.2f}; "
              f"median PBO (diag) {np.nanmedian([r[4] for r in rs]):.2f}")
        pos = [r for r in rs if r[3] > 0]
        for g in gates:
            print(f"   {g:20s} pass {np.mean([r[1][g] == 'PASS' for r in rs]):.2f}   | among {len(pos)} with IS SR>0: "
                  f"{np.mean([r[1][g] == 'PASS' for r in pos]) if pos else float('nan'):.2f}")
