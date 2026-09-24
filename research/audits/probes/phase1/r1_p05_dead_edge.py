"""R1-P05 (B2 + M3) — dead edge vs the v1.2 gate set (incl. the new wfo_oos gate) and the joint
holdout band.  SyntheticEvaluator 7x7, broad bump H=3.0, rho=0.5 (as p05b/p05c); the edge is
exactly 0 from fraction FROM of dev.  Schedules: default WFO (min_train 2y) and a user-chosen
min_train=1y (longer WFO span -> longer 'recent third').  'live' = never dies (control).
For every gate-PASS study: P(zero-edge 1y holdout passes the band) with the v1.2 joint band and
with the same bootstrap forced to the v1.1 marginal level alpha = 0.10."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

def one(job):
    import warnings; warnings.filterwarnings("ignore")
    from r1_common import ExploitEval, run_and_gate
    from e2e_common import dead_holdout_pass_rate
    from quantlab import opt, stats as st, gates as G
    from quantlab.evaluators import SyntheticEvaluator
    FROM, sched, seed = job
    space = opt.SearchSpace([opt.IntParam("a", 0, 6), opt.IntParam("b", 0, 6)])
    reg = () if FROM is None else ({"from": FROM, "bumps": (), "base_sharpe": 0.0},)
    inner = SyntheticEvaluator(bounds={"a": (0, 6), "b": (0, 6)},
                               bumps=({"center": {"a": 3, "b": 3}, "height": 3.0, "width": 0.35},),
                               rho=0.5, seed=100 + seed, trades_per_year=100, regimes=reg)
    ev = ExploitEval(inner)
    wfo = opt.WFOConfig() if sched == "default" else opt.WFOConfig(min_train="1y")
    res, rep = run_and_gate(ev, space, f"fbs-9999-r1dead-{FROM}-{sched}-{seed}", wfo=wfo)
    stt = rep.statuses()
    out = {"verdict": rep.verdict, "st": stt, "wfo": rep.diagnostics.get("wfo_oos"),
           "not_wfo_pass": all(s == "PASS" for g, s in stt.items() if g != "wfo_oos")}
    if rep.verdict == "PASS" or out["not_wfo_pass"]:
        out["p_dead_new"] = dead_holdout_pass_rate(res, rep)
        out["p0_band"] = rep.holdout_band["p_pass_zero_edge"]
        out["alpha"] = rep.holdout_band["tail_level"]
        orig = st.joint_tail_level
        st.joint_tail_level = lambda *a, **k: (0.10, float("nan"))
        try:
            rep2 = G.evaluate_gates(res, ev, periods_per_year=260.0, n_boot=500,
                                    ledger_dir=Path(__import__("tempfile").mkdtemp()))
        finally:
            st.joint_tail_level = orig
        out["p_dead_v11"] = dead_holdout_pass_rate(res, rep2)
    return job, out

if __name__ == "__main__":
    from r1_common import pmap
    NS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    jobs = [(f, s, k) for f in (0.55, 0.70, None) for s in ("default", "min_train=1y") for k in range(NS)]
    res = pmap(one, jobs)
    from collections import defaultdict
    grp = defaultdict(list)
    for (f, s, k), o in res:
        grp[(f, s)].append(o)
    for (f, s), os_ in grp.items():
        n = len(os_)
        pv = sum(o["verdict"] == "PASS" for o in os_)
        pre = sum(o["not_wfo_pass"] for o in os_)
        wfo_fail = sum(o["not_wfo_pass"] and o["st"]["wfo_oos"] != "PASS" for o in os_)
        rec = [o["wfo"]["sharpe_recent"] for o in os_ if o["wfo"]]
        rf = [o["wfo"]["recent_from"] for o in os_ if o["wfo"]][:1]
        print(f"dead_from={f} sched={s:13s}: verdict PASS {pv}/{n}; all-other-gates PASS {pre}/{n}, of which wfo_oos "
              f"stopped {wfo_fail}; recent-third from {rf}, median recent SR {np.median(rec):+.2f}")
        ps = [o for o in os_ if o["verdict"] == "PASS"]
        if ps:
            print(f"   passing studies: P(zero-edge holdout passes) v1.2 joint band "
                  f"{np.round([o['p_dead_new'] for o in ps], 2).tolist()} | v1.1-level (alpha .10) "
                  f"{np.round([o['p_dead_v11'] for o in ps], 2).tolist()} | band's own p_pass_zero_edge "
                  f"{np.round([o['p0_band'] for o in ps], 2).tolist()} | alpha {np.round([o['alpha'] for o in ps], 3).tolist()}")
