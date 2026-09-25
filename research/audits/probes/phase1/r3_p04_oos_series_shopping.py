"""R3-P04 (30b23b6) -- how much can the in-memory `cpcv_paths` / `wfo_oos` move the two OOS gates?

The gates read `study.cpcv_paths` and `study.wfo_oos` from memory; nothing ties them to the trial store or
to the CV / WFO scheme recorded in the ledger (`cv_scheme`).  Besides outright forgery (r3_p01 m1-m3), the
realistic path is *recomputing* them in a notebook with another scheme (opt.cpcv_paths / opt.walk_forward
on the verified returns matrix) and assigning the result back.  Worlds: zero-edge (r1_p06 family A, 10x10, rho 0.8); dead-edge
(r2_p05: 7x7, H=3, edge 0 from 55 % / 70 % of dev) and the live control; then
  * the pre-registered scheme (CPCV(10,2), WFO default) as run_study produced it,
  * best-of-K recomputed schemes: 15 CPCV (n_groups x k_test) and 36 WFO (refit x window x min_train).
Gate rules as in gates.py: oos_sharpe = median path SR >= 1.0; wfo_oos = SR >= 0.5 and recent third > 0.
Tmp ledgers, <= 4 workers, no data files.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import math
import numpy as np

CPCV = [(n, k) for n in (6, 8, 10, 12, 16) for k in (1, 2, 3)]
WFO = [(r, w, m) for r in ("1mo", "3mo", "6mo", "12mo") for w in ("anchored", "rolling:2y", "rolling:3y")
       for m in ("1y", "2y", "3y")]


def sr(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * math.sqrt(260)) if x.size > 2 and x.std() > 0 else float("nan")


FAMS = {"null": None, "dead55": 0.55, "dead70": 0.70, "live_H3": "live"}


def one(job):
    fam, seed = job
    import warnings; warnings.filterwarnings("ignore")
    import polars as pl
    from r1_common import ExploitEval, tmpdirs
    from quantlab import opt
    from quantlab.evaluators import SyntheticEvaluator
    if fam == "null":
        space = opt.SearchSpace([opt.IntParam("a", 0, 9, plateau_step=2), opt.IntParam("b", 0, 9, plateau_step=2)])
        inner = SyntheticEvaluator(bounds={"a": (0, 9), "b": (0, 9)}, rho=0.8, seed=500 + seed, trades_per_year=100)
    else:   # r1_p05 / r2_p05 worlds: 7x7, broad bump H=3.0, rho 0.5, edge exactly 0 from FROM of dev
        FROM = FAMS[fam]
        space = opt.SearchSpace([opt.IntParam("a", 0, 6, plateau_step=2), opt.IntParam("b", 0, 6, plateau_step=2)])
        reg = () if FROM == "live" else ({"from": FROM, "bumps": (), "base_sharpe": 0.0},)
        inner = SyntheticEvaluator(bounds={"a": (0, 6), "b": (0, 6)},
                                   bumps=({"center": {"a": 3, "b": 3}, "height": 3.0, "width": 0.35},),
                                   rho=0.5, seed=100 + seed, trades_per_year=100, regimes=reg)
    ld, sd = tmpdirs("rt_r3_p04_")
    res = opt.run_study(ExploitEval(inner), space, book="FBS", system=f"{fam}{seed}", issue=9999, attempt=1,
                        study_id=f"p04-{fam}-{seed}", n_jobs=1, wfo=opt.WFOConfig(), ledger_dir=ld, studies_dir=sd)
    tc = res.meta.get("trade_counts")

    def oos_med(paths):
        ps = paths.group_by("path_id").agg((pl.col("ret").mean() / pl.col("ret").std(ddof=1) * math.sqrt(260)).alias("s"))["s"].to_numpy()
        ps = ps[np.isfinite(ps)]
        return float(np.median(ps)) if ps.size else float("nan")

    def wfo_ok(w):
        if w is None or w.height == 0:
            return False, float("nan")
        r = w.sort("date")["ret"].fill_null(0.0).to_numpy()
        n_rec = max(2, int(math.ceil(r.size / 3)))
        a, b = sr(r), sr(r[-n_rec:])
        return (a >= 0.5 and b > 0), a

    base_oos = oos_med(res.cpcv_paths)
    base_wfo_ok, base_wfo = wfo_ok(res.wfo_oos)
    oos_alt = []
    for n, k in CPCV:
        try:
            p, _, _ = opt.cpcv_paths(res.returns, res.trials, space, opt.CPCVConfig(n, k), trade_counts=tc)
            oos_alt.append(oos_med(p))
        except Exception:
            oos_alt.append(float("nan"))
    wfo_alt = []
    for r, w, m in WFO:
        cfg = (opt.WFOConfig(refit_every=r, window="anchored", min_train=m) if w == "anchored" else
               opt.WFOConfig(refit_every=r, window="rolling", window_length=w.split(":")[1], min_train=m))
        try:
            wo, _, _ = opt.walk_forward(res.returns, res.trials, space, cfg, trade_counts=tc)
            wfo_alt.append(wfo_ok(wo))
        except Exception:
            wfo_alt.append((False, float("nan")))
    is_sr = sr(res.returns[f"t{res.selection['trial_id']}"].fill_null(0.0).to_numpy())
    return {"fam": fam, "seed": seed, "is_sr": is_sr, "base_oos": base_oos, "base_wfo_ok": base_wfo_ok, "base_wfo": base_wfo,
            "oos_alt": oos_alt, "wfo_alt": wfo_alt}


if __name__ == "__main__":
    from r1_common import pmap
    NS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    allrows = pmap(one, [(f, k) for f in FAMS for k in range(NS)])
    for fam in FAMS:
        rows = [r for r in allrows if r["fam"] == fam]
        b_oos = np.mean([r["base_oos"] >= 1.0 for r in rows])
        k_oos = np.mean([np.nanmax(r["oos_alt"]) >= 1.0 for r in rows])
        b_wfo = np.mean([r["base_wfo_ok"] for r in rows])
        k_wfo = np.mean([any(ok for ok, _ in r["wfo_alt"]) for r in rows])
        both_b = np.mean([r["base_oos"] >= 1.0 and r["base_wfo_ok"] for r in rows])
        both_k = np.mean([np.nanmax(r["oos_alt"]) >= 1.0 and any(ok for ok, _ in r["wfo_alt"]) for r in rows])
        print(f"== {fam}: {len(rows)} worlds")
        print(f"  oos_sharpe PASS: pre-registered CPCV(10,2) {b_oos:.2f} | best of {len(CPCV)} recomputed CPCV schemes {k_oos:.2f}")
        print(f"  wfo_oos    PASS: pre-registered WFO        {b_wfo:.2f} | best of {len(WFO)} recomputed WFO schedules {k_wfo:.2f}")
        print(f"  both PASS      : pre-registered            {both_b:.2f} | shopped {both_k:.2f}")
        print("  per world (seed, IS SR, base oos, max alt oos, base wfo SR, #WFO schedules passing):")
        for r in rows:
            print(f"    {r['seed']:2d} {r['is_sr']:+.2f}  {r['base_oos']:+.2f}  {np.nanmax(r['oos_alt']):+.2f}  "
                  f"{r['base_wfo']:+.2f}  {sum(ok for ok, _ in r['wfo_alt'])}/{len(WFO)}")
