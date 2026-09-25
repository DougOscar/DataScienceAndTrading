"""R3-P02 (30b23b6) -- judge plateau after the R2-1 fix: re-run of r2_p02 G1..G6 with the new API, plus
the candidate mechanical guards for the residual author-chosen scale.

Planted spike as r2_p02: 'a' on [1, 101], gauss bump at 50, width 0.02 of range (sd 2 units), height 2,
base 0.5, rho 1.  A point passes if Sharpe >= 50 % of the peak -> |a - 50| <~ 2.8 units.  An honest
+-20 % neighbourhood (+-10 units) must FAIL it.

Guards evaluated (the judge re-run with plateau_step := max(declared tolerance, floor)):
  F1  floor = f x (declared high - low), f = 0.02 / 0.05
  F2  floor = one search-grid step (the optimizer cannot resolve finer than its own grid)
and their false-positive cost on honest params (a lookback on [5, 200] at 10 / 20 / 50 / 120).
"""
import sys, warnings
from dataclasses import dataclass, replace
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from r1_common import tmpdirs
from quantlab import opt, gates as G
from quantlab.evaluators import SyntheticEvaluator

SPIKE = SyntheticEvaluator(bounds={"a": (1, 101)}, bumps=({"center": {"a": 50}, "height": 2.0, "width": 0.02},),
                           base_sharpe=0.5, rho=1.0, seed=3)


@dataclass
class MapEval:
    inner: SyntheticEvaluator
    fn: object
    tag: str = "map"
    periods_per_year: float = 260.0
    book: str = "FBS"
    requires_refit = False
    cost_version = "synthetic"
    is_synthetic = True

    @property
    def dev_window(self):
        return self.inner.dev_window

    def describe(self):
        return {"evaluator": "MapEval", "tag": self.tag, **self.inner.describe()}

    def prepare(self):
        pass

    def __call__(self, params, *, cost=None):
        return self.inner({"a": float(self.fn(params))}, cost=cost)


def study_and_plateau(ev, space, sid, **kw):
    ld, sd = tmpdirs("rt_r3_p02_")
    res = opt.run_study(ev, space, book="FBS", system=sid, issue=9999, attempt=1, study_id=sid, n_jobs=1, wfo=None,
                        ledger_dir=ld, studies_dir=sd, allow_large_grid=True, **kw)
    ctx = G.ledger_context(res, ld)
    pj = G.judge_plateau(res, ev, space=space, radius=ctx["plateau_radius"], periods_per_year=260.0)
    return res, pj, ctx["plateau_radius"]


def show(label, res, pj):
    pts = ", ".join(f"{q['param']}={q['value'] if not isinstance(q['value'], dict) else 'J'}:{'P' if q['pass'] else 'f'}"
                    for q in pj["points"] if q["param"] != "joint")
    sc = pj["plateau_score"]
    print(f"  {label:60s} sel={res.selected_params} score={sc if sc != sc else round(sc, 2)} "
          f"({'PASS' if sc >= 0.6 else ('SKIPPED' if sc != sc else 'FAIL')}), axes={pj['n_axes']} | {pts}")


def guarded(res, ev, space, radius, f=None, grid=False):
    """Judge again with plateau_step := max(declared tolerance, floor) on every numeric param."""
    ps = []
    for p in space.params:
        if p.kind == "categorical":
            ps.append(p); continue
        x = float(res.selected_params[p.name])
        tol = p.plateau_step if p.plateau_step is not None else G._radius_for(radius, p.name) * abs(x)
        floor = 0.0
        if f is not None:
            floor = max(floor, f * (p.high - p.low))
        if grid and p.step:
            floor = max(floor, float(p.step))
        ps.append(replace(p, plateau_scale=None, plateau_step=max(tol, floor)))
    sp2 = opt.SearchSpace(ps)
    return G.judge_plateau(res, ev, space=sp2, radius=radius, periods_per_year=260.0)["plateau_score"]


def guards(res, ev, space, radius):
    return (f"   guards: F1 f=0.02 -> {guarded(res, ev, space, radius, f=0.02):.2f}, F1 f=0.05 -> "
            f"{guarded(res, ev, space, radius, f=0.05):.2f}, F2 grid step -> {guarded(res, ev, space, radius, grid=True):.2f}")


print("== M5 control: honest a in [1, 101] step 1, relative 0.20")
sp = opt.SearchSpace([opt.IntParam("a", 1, 101, 1, plateau_scale="relative")])
ev = MapEval(SPIKE, lambda p: p["a"], "m5")
res, pj, rad = study_and_plateau(ev, sp, "m5")
show("a relative", res, pj); print(guards(res, ev, sp, rad))

print("== G1: offset reparametrisation d = a - 49.9 on [0.05, 5] step 0.05")
for lab, kw in (("relative (allowed: low > 0)", dict(plateau_scale="relative")),
                ("plateau_step=0.01", dict(plateau_step=0.01)), ("plateau_step=1.0 (honest 20% of range)", dict(plateau_step=1.0))):
    sp = opt.SearchSpace([opt.FloatParam("d", 0.05, 5.0, 0.05, **kw)])
    ev = MapEval(SPIKE, lambda p: 49.9 + p["d"], "g1")
    res, pj, rad = study_and_plateau(ev, sp, "g1")
    show(f"d {lab}", res, pj); print(guards(res, ev, sp, rad))

print("== G1c: tiny absolute step on the honest parametrisation (a on [1, 101], plateau_step=1)")
sp = opt.SearchSpace([opt.IntParam("a", 1, 101, 1, plateau_step=1)])
ev = MapEval(SPIKE, lambda p: p["a"], "g1c")
res, pj, rad = study_and_plateau(ev, sp, "g1c")
show("a plateau_step=1", res, pj); print(guards(res, ev, sp, rad))

print("== G1b: honest small positive threshold thr on [0.01, 2] step 0.01, spike at 0.05 (2 % of range)")
TH = SyntheticEvaluator(bounds={"a": (0.01, 2.0)}, bumps=({"center": {"a": 0.05}, "height": 2.0, "width": 0.02},),
                        base_sharpe=0.5, rho=1.0, seed=3)
for lab, kw in (("relative", dict(plateau_scale="relative")), ("plateau_step=0.2", dict(plateau_step=0.2))):
    sp = opt.SearchSpace([opt.FloatParam("thr", 0.01, 2.0, 0.01, **kw)])
    ev = MapEval(TH, lambda p: p["thr"], "g1b")
    res, pj, rad = study_and_plateau(ev, sp, "g1b")
    show(f"thr {lab}", res, pj); print(guards(res, ev, sp, rad))

print("== G2: k averaged duplicates a = mean(a1..ak), levels 46..54 step 2, relative (joint axis now)")
for k in (1, 2, 3, 4, 5):
    ps = [opt.IntParam(f"a{i}", 46, 54, 2, plateau_scale="relative") for i in range(k)]
    ev = MapEval(SPIKE, lambda p, k=k: np.mean([p[f"a{i}"] for i in range(k)]), f"g2{k}")
    res, pj, rad = study_and_plateau(ev, opt.SearchSpace(ps), f"g2-{k}", method="grid")
    print(f"  k={k}: score {pj['plateau_score']:.2f} ({'PASS' if pj['plateau_score'] >= 0.6 else 'FAIL'}); "
          f"per-axis {pj['pass_count_by_param']}")

print("== G3: spiky param as UNORDERED categorical")
LV = tuple(range(10, 101, 10))
try:
    opt.CategoricalParam("a", LV, ordered=False)
    print("  numeric unordered categorical: ACCEPTED")
except ValueError as e:
    print("  numeric unordered categorical: refused:", str(e)[:90])
LVs = tuple(f"L{v}" for v in LV)
sp = opt.SearchSpace([opt.CategoricalParam("a", LVs, ordered=False), opt.IntParam("b", 5, 15, plateau_scale="relative")])
ev = MapEval(SPIKE, lambda p: float(p["a"][1:]), "g3s")
res, pj, rad = study_and_plateau(ev, sp, "g3s")
show("string-labelled unordered categorical 'L10'..'L100' + b", res, pj)
LVm = tuple(f"{v}" for v in LV)
try:
    opt.CategoricalParam("a", LVm, ordered=False)
    print("  numeric STRINGS ('10'..'100') unordered: ACCEPTED")
except ValueError as e:
    print("  numeric strings unordered: refused:", str(e)[:90])
sp = opt.SearchSpace([opt.CategoricalParam("a", LVs, ordered=False)])
res, pj, rad = study_and_plateau(MapEval(SPIKE, lambda p: float(p["a"][1:]), "g3o"), sp, "g3o")
show("string-labelled unordered categorical ALONE", res, pj)

print("== G4: ordered categorical with fine levels (+-1/+-2 levels now)")
for lv in ((48, 49, 50, 51, 52), (46, 48, 50, 52, 54), tuple(range(40, 61, 1))):
    sp = opt.SearchSpace([opt.CategoricalParam("a", lv, ordered=True)])
    res, pj, rad = study_and_plateau(MapEval(SPIKE, lambda p: p["a"], "g4"), sp, f"g4-{len(lv)}")
    show(f"levels {lv[0]}..{lv[-1]} ({len(lv)})", res, pj)

print("== G6: per-param radius 0.10 on a wider spike (sd 4 units)")
W = SyntheticEvaluator(bounds={"a": (1, 101)}, bumps=({"center": {"a": 50}, "height": 2.0, "width": 0.04},),
                       base_sharpe=0.5, rho=1.0, seed=3)
for radius in (None, {"a": 0.10}):
    sp = opt.SearchSpace([opt.IntParam("a", 1, 101, 1, plateau_scale="relative")])
    ev = MapEval(W, lambda p: p["a"], "g6")
    res, pj, rad = study_and_plateau(ev, sp, f"g6-{radius is None}", plateau_radius=radius)
    show(f"radius {radius or 0.20}", res, pj); print(guards(res, ev, sp, rad))

print("== False-positive cost of the guards: HONEST broad edge in a lookback on [5, 200] step 1")
print("   (edge in log(lookback): Sharpe falls to 50 % at x2 / x0.5 of the optimum; relative 0.20)")
import math
for centre in (10, 20, 50, 120):
    # bump in log-space: SyntheticEvaluator on u = log(L); width chosen so half-height at a factor ~2
    LB = SyntheticEvaluator(bounds={"a": (math.log(5), math.log(200))},
                            bumps=({"center": {"a": math.log(centre)}, "height": 2.0, "width": 0.16},),
                            base_sharpe=0.0, rho=1.0, seed=3)
    sp = opt.SearchSpace([opt.IntParam("L", 5, 200, 1, plateau_scale="relative")])
    ev = MapEval(LB, lambda p: math.log(p["L"]), f"lb{centre}")
    res, pj, rad = study_and_plateau(ev, sp, f"lb-{centre}")
    show(f"lookback optimum {centre}", res, pj); print(guards(res, ev, sp, rad))
