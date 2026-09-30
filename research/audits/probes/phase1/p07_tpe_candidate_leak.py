"""P07 — TPE candidate-set leak into CPCV / walk-forward.

run_study(method="tpe") lets Optuna choose WHICH configurations exist using the FULL-dev
objective.  cpcv_paths()/walk_forward() then select, on each train fold / each refit date,
among those candidates only.  The candidate set therefore already encodes the test folds
(and, for WFO, the future).  With parameter-smooth noise (real strategies: neighbouring
configurations share most trades), TPE concentrates candidates in the region that was
lucky on the full sample, and the "OOS" paths inherit that luck.

Zero-edge world: r(p) = vol * ( sqrt(rho) z_common + sqrt(1-rho) * GP(p) ), GP a smooth
random field over p in [0,1]^2 (RBF lengthscale L, built from 64 lattice basis series).
Compare, over seeds: CPCV path-median OOS Sharpe and WFO OOS Sharpe for
  grid (all 21x21 = 441 configs fixed in advance)  vs  TPE (60 configs chosen on full data).
Both should be ~0 on average; a systematic positive gap for TPE is the leak.
"""
import math, sys, tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np, polars as pl
from quantlab import opt
from quantlab.contracts import Outcome

T = 2340
DATES = np.busday_offset(np.datetime64("2016-05-02"), np.arange(T), roll="forward")


@dataclass
class SmoothNull:
    seed: int
    L: float = 0.15
    rho: float = 0.3
    vol: float = 0.005
    book: str = "FBS"
    periods_per_year: float = 260.0
    requires_refit = False
    cost_version = "probe"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        g = np.linspace(0, 1, 8)
        self.C = np.array([(x, y) for x in g for y in g])
        self.Z = rng.standard_normal((T, len(self.C)))
        self.zc = rng.standard_normal(T)

    @property
    def dev_window(self):
        return (str(DATES[0]), str(DATES[-1]))

    def __call__(self, params, *, cost=None):
        p = np.array([params["x"], params["y"]])
        w = np.exp(-((self.C - p) ** 2).sum(1) / (2 * self.L ** 2))
        field = self.Z @ w / np.linalg.norm(w)
        r = self.vol * (math.sqrt(self.rho) * self.zc + math.sqrt(1 - self.rho) * field)
        daily = pl.DataFrame({"date": DATES.astype("datetime64[D]"), "ret": r}).with_columns(pl.col("date").cast(pl.Date))
        tr = pl.DataFrame({"entry_ts": [], "exit_ts": []}, schema={"entry_ts": pl.Datetime("ms"), "exit_ts": pl.Datetime("ms")})
        return Outcome(daily=daily, trades=tr, metrics={"n_trades": 900.0, "hold_days_max": 3.0})


def sh(x):
    return float(x.mean() / x.std(ddof=1) * math.sqrt(260))


space = opt.SearchSpace([opt.FloatParam("x", 0.0, 1.0, 0.05), opt.FloatParam("y", 0.0, 1.0, 0.05)])
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
out = {"grid": [], "tpe": []}
for s in range(NS):
    ev = SmoothNull(seed=s)
    for method in ("grid", "tpe"):
        tmp = Path(tempfile.mkdtemp(prefix="rt_p07_"))
        res = opt.run_study(ev, space, book="FBS", system="p07", issue=9999, attempt=1, study_id=f"p07-{method}-{s}",
                            method=method, n_trials=None if method == "grid" else 60, seed=s, n_jobs=1,
                            wfo=opt.WFOConfig(), ledger_dir=tmp / "l", studies_dir=tmp / "s")
        cp = res.cpcv_paths
        med = float(np.median([sh(g.sort("date")["ret"].to_numpy()) for _, g in cp.group_by("path_id")]))
        wfo = sh(res.wfo_oos["ret"].to_numpy())
        ins = res.selection["sharpe"]
        out[method].append((med, wfo, ins))
    print(f"seed {s:2d}: grid CPCV-med {out['grid'][-1][0]:+.2f} WFO {out['grid'][-1][1]:+.2f} IS {out['grid'][-1][2]:+.2f} | "
          f"tpe CPCV-med {out['tpe'][-1][0]:+.2f} WFO {out['tpe'][-1][1]:+.2f} IS {out['tpe'][-1][2]:+.2f}", flush=True)
for m in out:
    a = np.array(out[m])
    print(f"{m}: mean CPCV path-median OOS Sharpe {a[:,0].mean():+.3f} (se {a[:,0].std(ddof=1)/math.sqrt(len(a)):.3f}); "
          f"mean WFO OOS {a[:,1].mean():+.3f} (se {a[:,1].std(ddof=1)/math.sqrt(len(a)):.3f}); mean IS {a[:,2].mean():+.3f}")
d = np.array(out["tpe"])[:, :2] - np.array(out["grid"])[:, :2]
print(f"paired TPE - grid: CPCV {d[:,0].mean():+.3f} (se {d[:,0].std(ddof=1)/math.sqrt(len(d)):.3f}), "
      f"WFO {d[:,1].mean():+.3f} (se {d[:,1].std(ddof=1)/math.sqrt(len(d)):.3f})")
