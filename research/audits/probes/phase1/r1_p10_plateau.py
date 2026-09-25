"""R1-P10 (B4, M5 + new plateau attacks).
(a) B4: optimizer PlateauConfig games (peak_fraction=0.05, knn=2) no longer move the gate.
(b) M5: same narrow bump (width 2% of range) on grids of step 10/5/2/1, default radius.
(c) NEW: evaluate_gates(plateau_radius=1e-6) -> no level inside the radius -> 'adjacent levels'
    fallback = the v1.1 grid-step neighbourhood; the free kwarg restores M5.
(d) NEW: irrelevant params (surface ignores them) add exact copies of the peak as neighbours.
(e) NEW: Sobol candidate sets in d = 2..5: neighbour counts under the ±20 % box; share of
    studies with 0 neighbours (plateau SKIPPED -> verdict can never PASS) or <= 2 neighbours."""
import sys, tempfile, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from r1_common import ExploitEval, mech_true, tmpdirs
from quantlab import opt, gates as G
from quantlab.evaluators import SyntheticEvaluator

def study(ev, space, sid, **kw):
    ld, sd = tmpdirs()
    return opt.run_study(ev, space, book="FBS", system="p10", issue=9999, attempt=1, study_id=sid, n_jobs=1,
                         wfo=None, ledger_dir=ld, studies_dir=sd, **kw), ld

print("(a) B4")
inner = SyntheticEvaluator(bounds={"a": (0, 10), "b": (0, 10)},
                           bumps=({"center": {"a": 5, "b": 5}, "height": 2.5, "width": 0.06},), rho=0.7, seed=11)
ev = ExploitEval(inner)
space = opt.SearchSpace([opt.IntParam("a", 0, 10), opt.IntParam("b", 0, 10)])
for name, cfg in (("default", opt.PlateauConfig()), ("peak_fraction=0.05", opt.PlateauConfig(peak_fraction=0.05)),
                  ("knn=2", opt.PlateauConfig(neighbourhood="knn", knn=2))):
    res, ld = study(ev, space, f"p10a-{len(name)}", selection=cfg)
    rep = G.evaluate_gates(res, ev, periods_per_year=260.0, mechanism_check=mech_true, n_boot=100, ledger_dir=ld)
    r = rep.row("plateau")
    print(f"   {name:20s} optimizer score {res.selection.get('plateau_score')} -> gate {r.value} {r.status} "
          f"({rep.diagnostics['plateau']['n_neighbours']} nbrs)")

print("(b)/(c) grid resolution, narrow bump width 2% of range; a in [1,101] (relative mode) and [0,100] (range mode)")
for lo in (0, 1):
    smooth = ExploitEval(SyntheticEvaluator(bounds={"a": (lo, lo + 100)},
                         bumps=({"center": {"a": lo + 50}, "height": 2.0, "width": 0.02},), base_sharpe=0.5, rho=1.0, seed=3))
    for step in (10, 5, 2, 1):
        sp = opt.SearchSpace([opt.IntParam("a", lo, lo + 100, step)])
        res, ld = study(smooth, sp, f"p10b-{lo}-{step}")
        out = []
        for rad in (None, 1e-6):
            rep = G.evaluate_gates(res, smooth, periods_per_year=260.0, mechanism_check=mech_true, n_boot=100,
                                   ledger_dir=ld, plateau_radius=rad)
            r = rep.row("plateau")
            out.append(f"radius={rad}: {r.value if r.value is None else round(r.value, 2)} {r.status} "
                       f"({rep.diagnostics['plateau']['n_neighbours']} nbrs)")
        print(f"   low={lo} step {step:3d}: selected a={res.selected_params['a']}; " + " | ".join(out))

print("(d) irrelevant parameters: relevant 'a' 0..10 with a bump giving a partial plateau; + k-level dummies")
ev_d = ExploitEval(SyntheticEvaluator(bounds={"a": (0, 10)}, bumps=({"center": {"a": 5}, "height": 2.0, "width": 0.09},),
                                      rho=1.0, seed=5))
for nd, lv in ((0, 0), (1, 21), (2, 21)):
    ps = [opt.IntParam("a", 0, 10)] + [opt.IntParam(f"z{i}", 0, lv - 1) for i in range(nd)]
    sp = opt.SearchSpace(ps)
    res, ld = study(ev_d, sp, f"p10d-{nd}", allow_large_grid=True)
    pi = G.plateau_score(res)
    print(f"   {nd} dummies x {lv} levels: plateau {pi['plateau_score']:.3f} over {pi['n_neighbours']} nbrs; rules {pi['rules']}")

print("(e) Sobol neighbour counts (planted zero surface, rho=0.8)")
specs = [opt.IntParam("look", 10, 200), opt.FloatParam("mult", 1.0, 4.0), opt.IntParam("hold", 1, 20),
         opt.FloatParam("thr", 0.0, 2.0), opt.IntParam("slow", 20, 300)]
for d in (2, 3, 4, 5):
    for n in (64, 256):
        sp = opt.SearchSpace(specs[:d])
        nb, pas = [], []
        for s in range(6):
            inner = SyntheticEvaluator(bounds={p.name: (p.low, p.high) for p in specs[:d]}, rho=0.8, seed=900 + s)
            ev_e = ExploitEval(inner)
            res, ld = study(ev_e, sp, f"p10e-{d}-{n}-{s}", method="sobol", n_trials=n, seed=s)
            pi = G.plateau_score(res)
            nb.append(pi["n_neighbours"]); pas.append(pi["plateau_score"])
        nb = np.array(nb)
        print(f"   d={d} n={n:3d}: neighbours per study {nb.tolist()} -> zero {np.mean(nb == 0):.0%}, <=2 {np.mean(nb <= 2):.0%}; "
              f"scores {np.round(pas, 2).tolist()}")
