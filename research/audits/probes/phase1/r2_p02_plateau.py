"""R2-P02 (48e696f) -- judge-run plateau: re-runs of M5 / N4 / N8 and gaming attempts.

Planted spike: SyntheticEvaluator, 'a' on [1, 101], gauss bump at a = 50 with width 0.02 of the range
(sd = 2 units), base Sharpe 0.5, height 2.0, rho = 1 (the measured Sharpe is an exact function of a).
A perturbation passes if Sharpe >= 50 % of the peak (2.5) -> |a - 50| <~ 2.8 units.  An honest
+-20 % plateau (+-10 units) must FAIL it.

 M5  grid resolution (step 10/5/2/1) with the judge plateau.
 N4  Sobol d = 3..5, n = 64: does the judge plateau ever SKIP?
 N8  irrelevant params.
 G1  offset reparametrisation: d = a - 49.9 declared on [0.05, 5] (strictly positive -> relative mode);
     and the same d declared on [0, 5] (range mode: +-20 % of the DECLARED range).
 G1b small-valued positive param, no cheating: thr on [0.01, 2], spike at thr = 0.05 (width 0.02 of range).
 G2  k averaged duplicate params (a = mean(a1..ak)).
 G3  the spiky param declared as an UNORDERED categorical (+ one harmless numeric param).
 G4  ordered categorical with fine levels (49, 50, 51).
 G5  is_valid constraint that rejects every perturbation.
 G6  per-param pre-registered radius {'a': 0.10} (floor).
"""
import sys, warnings
from dataclasses import dataclass
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from r1_common import ExploitEval, tmpdirs, mech_true
from quantlab import opt, gates as G
from quantlab.evaluators import SyntheticEvaluator

SPIKE = SyntheticEvaluator(bounds={"a": (1, 101)}, bumps=({"center": {"a": 50}, "height": 2.0, "width": 0.02},),
                           base_sharpe=0.5, rho=1.0, seed=3)


@dataclass
class MapEval:
    """Strategy author's parametrisation: params -> the underlying economic parameter 'a'."""
    inner: SyntheticEvaluator
    fn: object
    periods_per_year: float = 260.0
    book: str = "FBS"
    requires_refit = False
    cost_version = "synthetic"
    is_synthetic = True

    @property
    def dev_window(self):
        return self.inner.dev_window

    def describe(self):
        return {"evaluator": "MapEval", **self.inner.describe()}

    def prepare(self):
        pass

    def __call__(self, params, *, cost=None):
        return self.inner({"a": float(self.fn(params))}, cost=cost)


def study_and_plateau(ev, space, sid, **kw):
    ld, sd = tmpdirs()
    res = opt.run_study(ev, space, book="FBS", system=sid, issue=9999, attempt=1, study_id=sid, n_jobs=1, wfo=None,
                        ledger_dir=ld, studies_dir=sd, allow_large_grid=True, **kw)
    ctx = G.ledger_context(res, ld)
    pj = G.judge_plateau(res, ev, space=space, radius=ctx["plateau_radius"], periods_per_year=260.0)
    return res, pj


def show(label, res, pj):
    pts = ", ".join(f"{q['param']}={q['value']}:{'P' if q['pass'] else 'f'}" for q in pj["points"])
    sc = pj["plateau_score"]
    print(f"  {label:58s} sel={res.selected_params} score={sc if sc != sc else round(sc, 2)} "
          f"({'PASS' if sc >= 0.6 else ('SKIPPED' if sc != sc else 'FAIL')}), axes={pj['n_axes']} | {pts}")


print("== M5: grid resolution, honest parametrisation a in [1, 101]")
for step in (10, 5, 2, 1):
    sp = opt.SearchSpace([opt.IntParam("a", 1, 101, step)])
    res, pj = study_and_plateau(MapEval(SPIKE, lambda p: p["a"]), sp, f"m5-{step}")
    show(f"step {step}", res, pj)

print("== N4: Sobol d=3..5 n=64 (zero surface, rho .8) -> judge plateau points / SKIP?")
specs = [opt.IntParam("look", 10, 200), opt.FloatParam("mult", 1.0, 4.0), opt.IntParam("hold", 1, 20),
         opt.FloatParam("thr", 0.0, 2.0), opt.IntParam("slow", 20, 300)]
for d in (3, 4, 5):
    sk, sc = 0, []
    for s in range(4):
        inner = SyntheticEvaluator(bounds={p.name: (p.low, p.high) for p in specs[:d]}, rho=0.8, seed=900 + s)
        ld, sd = tmpdirs()
        sp = opt.SearchSpace(specs[:d])
        res = opt.run_study(ExploitEval(inner), sp, book="FBS", system=f"n4{d}{s}", issue=9999, attempt=1,
                            study_id=f"n4-{d}-{s}", method="sobol", n_trials=64, seed=s, n_jobs=1, wfo=None,
                            ledger_dir=ld, studies_dir=sd)
        pj = G.judge_plateau(res, ExploitEval(inner), space=sp, radius=0.2, periods_per_year=260.0)
        sk += pj["plateau_score"] != pj["plateau_score"]
        sc.append(round(pj["plateau_score"], 2))
    print(f"  d={d}: points per study {4 * d}, SKIPPED {sk}/4, scores {sc}")

print("== N8: irrelevant params (spike on 'a' + k dummies)")
for nd in (0, 1, 2):
    ps = [opt.IntParam("a", 1, 101, 1)] + [opt.IntParam(f"z{i}", 1, 5) for i in range(nd)]
    res, pj = study_and_plateau(MapEval(SPIKE, lambda p: p["a"]), opt.SearchSpace(ps), f"n8-{nd}")
    print(f"  {nd} dummies: score {pj['plateau_score']:.2f} (pooled {pj['pooled_share']:.2f}), weakest {pj['weakest_axis']}")

print("== G1: offset reparametrisation (same economic spike)")
sp = opt.SearchSpace([opt.FloatParam("d", 0.05, 5.0, 0.05)])
res, pj = study_and_plateau(MapEval(SPIKE, lambda p: 49.9 + p["d"]), sp, "g1")
show("d = a - 49.9 on [0.05, 5] (relative mode)", res, pj)
sp = opt.SearchSpace([opt.FloatParam("d", 0.0, 5.0, 0.05)])
res, pj = study_and_plateau(MapEval(SPIKE, lambda p: 49.9 + p["d"]), sp, "g1r")
show("same, declared low = 0 (range mode, +-20% of 5)", res, pj)

print("== G1b: honest small-valued positive param (thr on [0.01, 2], spike at 0.05, width 2 % of range)")
TH = SyntheticEvaluator(bounds={"thr": (0.01, 2.0)}, bumps=({"center": {"thr": 0.05}, "height": 2.0, "width": 0.02},),
                        base_sharpe=0.5, rho=1.0, seed=3)
for lo in (0.01, 0.0):
    sp = opt.SearchSpace([opt.FloatParam("thr", lo, 2.0, 0.01)])
    ev = MapEval(TH, lambda p: p["thr"])
    ev.fn = None
    class ThrEval(MapEval):
        def __call__(self, params, *, cost=None):
            return self.inner({"thr": params["thr"]}, cost=cost)
    res, pj = study_and_plateau(ThrEval(TH, None), sp, f"g1b-{lo}")
    show(f"thr declared low={lo}", res, pj)

print("== G2: k averaged duplicates a = mean(a1..ak), each on [1, 101]; levels 46..54 step 2")
for k in (1, 2, 3, 4, 5):
    ps = [opt.IntParam(f"a{i}", 46, 54, 2) for i in range(k)]
    res, pj = study_and_plateau(MapEval(SPIKE, lambda p, k=k: np.mean([p[f"a{i}"] for i in range(k)])),
                                opt.SearchSpace(ps), f"g2-{k}", method="grid")
    print(f"  k={k}: score {pj['plateau_score']:.2f} ({'PASS' if pj['plateau_score'] >= 0.6 else 'FAIL'}); "
          f"per-axis {pj['pass_count_by_param']}")

print("== G3: spiky param as UNORDERED categorical + harmless numeric 'b'")
LV = tuple(range(10, 101, 10)) + (50,)
LV = tuple(sorted(set(range(10, 101, 10))))
sp = opt.SearchSpace([opt.CategoricalParam("a", LV, ordered=False), opt.IntParam("b", 5, 15)])
res, pj = study_and_plateau(MapEval(SPIKE, lambda p: p["a"]), sp, "g3")
show("a unordered categorical (10..100), b int 5..15 (ignored)", res, pj)

print("== G4: ordered categorical with fine levels")
sp = opt.SearchSpace([opt.CategoricalParam("a", (48, 49, 50, 51, 52), ordered=True)])
res, pj = study_and_plateau(MapEval(SPIKE, lambda p: p["a"]), sp, "g4")
show("a ordered categorical (48..52)", res, pj)

print("== G5: constraint rejecting perturbations")
sp = opt.SearchSpace([opt.IntParam("a", 1, 101, 1)], constraint=lambda p: p["a"] % 10 == 0)
res, pj = study_and_plateau(MapEval(SPIKE, lambda p: p["a"]), sp, "g5")
show("constraint a % 10 == 0", res, pj)

print("== G6: pre-registered per-param radius 0.10 vs 0.20 on a wider spike (width 4 %, sd 4 units)")
W = SyntheticEvaluator(bounds={"a": (1, 101)}, bumps=({"center": {"a": 50}, "height": 2.0, "width": 0.04},),
                       base_sharpe=0.5, rho=1.0, seed=3)
for rad in (None, {"a": 0.10}):
    sp = opt.SearchSpace([opt.IntParam("a", 1, 101, 1)])
    res, pj = study_and_plateau(MapEval(W, lambda p: p["a"]), sp, f"g6-{rad is None}", plateau_radius=rad)
    show(f"radius {rad or 0.20}", res, pj)
