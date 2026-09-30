"""P10 — The plateau gate trusts the optimizer's own plateau_score, computed with the
optimizer-chosen PlateauConfig (peak_fraction, radius, knn).  Same spiky study, three configs:
the gate's value moves from FAIL to PASS with no change in the data.  Also: grid resolution
(step) changes the plateau score of the SAME smooth surface."""
import sys, tempfile
from pathlib import Path
import polars as pl
from dataclasses import replace
from quantlab import opt, gates as G
from quantlab.evaluators import SyntheticEvaluator
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from e2e_common import ExploitEval, mech_true

# single sharp spike on a flat-zero surface + mild noise
inner = SyntheticEvaluator(bounds={"a": (0, 10), "b": (0, 10)},
                           bumps=({"center": {"a": 5, "b": 5}, "height": 2.5, "width": 0.06},), rho=0.7, seed=11)
ev = ExploitEval(inner)
space = opt.SearchSpace([opt.IntParam("a", 0, 10), opt.IntParam("b", 0, 10)])
for name, cfg in (("default", opt.PlateauConfig()),
                  ("peak_fraction=0.05", opt.PlateauConfig(peak_fraction=0.05)),
                  ("knn=2 (neighbourhood='knn')", opt.PlateauConfig(neighbourhood="knn", knn=2))):
    tmp = Path(tempfile.mkdtemp(prefix="rt_p10_"))
    res = opt.run_study(ev, space, book="FBS", system="p10", issue=9999, attempt=1, study_id=f"p10-{len(name)}",
                        n_jobs=1, wfo=None, selection=cfg, ledger_dir=tmp / "l", studies_dir=tmp / "s")
    rep = G.evaluate_gates(res, ev, periods_per_year=260.0, mechanism_check=mech_true, n_boot=200)
    own = G.plateau_score(res)       # gate-side recomputation with DESIGN parameters
    print(f"{name:30s} selected {res.selected_params}  gate plateau value {rep.values()['plateau']:.2f} "
          f"({rep.statuses()['plateau']})  | gate's own recompute {own['plateau_score']:.2f} ({own['method']})")

# grid resolution: same narrow bump (width 2 units of a), coarse vs fine grid
smooth = ExploitEval(SyntheticEvaluator(bounds={"a": (0, 100)}, bumps=({"center": {"a": 50}, "height": 2.0, "width": 0.02},),
                                        base_sharpe=0.5, rho=1.0, seed=3))
for step in (10, 5, 2, 1):
    sp = opt.SearchSpace([opt.IntParam("a", 0, 100, step)])
    tmp = Path(tempfile.mkdtemp(prefix="rt_p10b_"))
    res = opt.run_study(smooth, sp, book="FBS", system="p10b", issue=9999, attempt=1, study_id=f"p10b-{step}",
                        n_jobs=1, wfo=None, ledger_dir=tmp / "l", studies_dir=tmp / "s")
    print(f"same surface, grid step {step:3d} ({sp.grid_size()} trials): plateau {res.selection['plateau_score']:.2f}, "
          f"selected a={res.selected_params['a']}")
