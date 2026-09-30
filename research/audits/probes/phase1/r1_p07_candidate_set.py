"""R1-P07 (B3) — candidate-set leak after the fix.
Zero-edge smooth random field (SmoothNull from p07).  Per seed:
  grid   : all 441 configs                       (data-independent reference)
  sobol  : 60 Sobol configs                      (the fix)
  tpe    : 60 TPE configs -> gates must SKIP oos_sharpe / wfo_oos / cscv_oos_loss
  tpe->sobol : TPE 60 configs, then run_study(resume=True, method="sobol", n_trials=60) on the
               SAME study id: candidate set = TPE picks + Sobol points.  Is it flagged?
Reports mean CPCV path-median / WFO OOS Sharpe (true value 0) and the gate statuses."""
import math, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

def one(s):
    import warnings; warnings.filterwarnings("ignore")
    import importlib.util
    from r1_common import tmpdirs, ann_sharpe
    from quantlab import opt, gates as G
    src = open(str(Path(__file__).resolve().parent / "p07_tpe_candidate_leak.py")).read()
    ns = {}
    exec(src[:src.index("space = opt.SearchSpace")], ns)
    ev = ns["SmoothNull"](seed=s)
    space = opt.SearchSpace([opt.FloatParam("x", 0.0, 1.0, 0.05), opt.FloatParam("y", 0.0, 1.0, 0.05)])
    out = {}
    for arm in ("grid", "sobol", "tpe", "tpe->sobol"):
        ld, sd = tmpdirs()
        sid = f"p07r1-{arm.replace('->','2')}-{s}"
        m0 = "tpe" if arm.startswith("tpe") else arm
        res = opt.run_study(ev, space, book="FBS", system="p07", issue=9999, attempt=1, study_id=sid, method=m0,
                            n_trials=None if m0 == "grid" else 60, seed=s, n_jobs=1, wfo=opt.WFOConfig(),
                            ledger_dir=ld, studies_dir=sd)
        if arm == "tpe->sobol":
            res = opt.run_study(ev, space, book="FBS", system="p07", issue=9999, attempt=1, study_id=sid,
                                method="sobol", n_trials=60, seed=s, n_jobs=1, wfo=opt.WFOConfig(),
                                ledger_dir=ld, studies_dir=sd, resume=True)
        cp = res.cpcv_paths
        med = float(np.median([ann_sharpe(g.sort("date")["ret"].to_numpy()) for _, g in cp.group_by("path_id")]))
        wfo = ann_sharpe(res.wfo_oos["ret"].to_numpy())
        rep = G.evaluate_gates(res, None, periods_per_year=260.0, n_boot=100, ledger_dir=ld)
        st = rep.statuses()
        out[arm] = (med, wfo, res.trials.height, res.meta["candidate_set_data_dependent"],
                    st["oos_sharpe"], st["wfo_oos"], st["cscv_oos_loss"])
    return out

if __name__ == "__main__":
    from r1_common import pmap
    NS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    rows = pmap(one, range(NS))
    for arm in ("grid", "sobol", "tpe", "tpe->sobol"):
        a = np.array([[r[arm][0], r[arm][1]] for r in rows])
        se = a.std(0, ddof=1) / math.sqrt(len(a))
        r0 = rows[0][arm]
        print(f"{arm:11s} n_trials={r0[2]:3d} data_dependent_flag={r0[3]!s:5s} | mean CPCV-med OOS {a[:,0].mean():+.3f} "
              f"(se {se[0]:.3f}), WFO OOS {a[:,1].mean():+.3f} (se {se[1]:.3f}) | gate statuses oos/wfo/cscv: "
              f"{sorted(set((r[arm][4], r[arm][5], r[arm][6]) for r in rows))}")
    for arm in ("sobol", "tpe->sobol"):
        d = np.array([[r[arm][0] - r["grid"][0], r[arm][1] - r["grid"][1]] for r in rows])
        print(f"paired {arm} - grid: CPCV {d[:,0].mean():+.3f} (se {d[:,0].std(ddof=1)/math.sqrt(len(d)):.3f}), "
              f"WFO {d[:,1].mean():+.3f} (se {d[:,1].std(ddof=1)/math.sqrt(len(d)):.3f})")
