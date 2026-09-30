"""S6 red team: does the plateau pick misrepresent the surface? Replays the study's CPCV/WFO on
the stored net matrix with (a) the study's plateau selection and (b) a raw-argmax procedure
(min_centre_quantile=1.0); cost-stress Sharpe for the full-sample net best and the raw-argmax
configs; nights by side for the swap argument.
Evaluations (diagnostic): 2 configs x (base + stressed) = 4, plus 1 repeat of trial 492.
"""
import sys, warnings
from dataclasses import replace
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from s5_load_study import load_study, make_evaluator  # noqa
from quantlab import opt, costs

PPY = 260.0


def sh(x):
    x = np.asarray(x, float); s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(PPY)) if s > 0 else 0.0


def main():
    st = load_study(); space = st.meta["search_space_obj"]; tc = st.meta["trade_counts"]
    obj = opt.Objective(kind="block_sharpe", n_blocks=4, lam=0.5, min_trades=200)
    cv = opt.CPCVConfig(n_groups=10, k_test=2)
    wfo = opt.WFOConfig(refit_every="3mo", window="anchored", min_train="2y")
    cols = [c for c in st.returns.columns if c != "date"]
    S = np.array([sh(st.returns[c].to_numpy()) for c in cols])
    s492 = S[cols.index("t492")]
    print(f"selected trial 492 net SR {s492:+.3f} = percentile {np.mean(S < s492):.2f} of the 840-config surface")
    for name, cfg in (("plateau (study)", opt.PlateauConfig()), ("raw argmax", opt.PlateauConfig(min_centre_quantile=1.0))):
        paths, _, _ = opt.cpcv_paths(st.returns, st.trials, space, cv, objective=obj, selection=cfg, trade_counts=tc, periods_per_year=PPY)
        psr = [sh(g["ret"].to_numpy()) for _, g in paths.group_by("path_id")]
        w, wp, _ = opt.walk_forward(st.returns, st.trials, space, wfo, objective=obj, selection=cfg, trade_counts=tc, periods_per_year=PPY)
        wr = w["ret"].to_numpy(); k = len(wr) // 3
        print(f"[{name}] CPCV median path SR {np.median(psr):+.3f} (max {np.max(psr):+.3f}); WFO OOS {sh(wr):+.3f}, recent third {sh(wr[-k:]):+.3f}")
    ev = make_evaluator()
    spec = costs.load_instrument("EURUSD", book="FBS")
    stressed = ev.cost.stressed(spec=spec) if "spec" in ev.cost.stressed.__code__.co_varnames else ev.cost.stressed()
    print("stress version:", stressed.version)
    for p in (dict(lookback=90, stop_mult=1.0, hold=18), dict(lookback=70, stop_mult=1.0, hold=12)):
        b = ev(p); s = ev(p, cost=stressed)
        print(p, f"base SR {b.metrics['sharpe']:+.3f}  stressed SR {s.metrics['sharpe']:+.3f}")
    t = ev(dict(lookback=80, stop_mult=1.0, hold=18)).trades
    print("nights by side (492):", t.group_by("direction").agg(pl.col("nights").sum().alias("n_sum"), pl.col("nights").mean().round(2).alias("n_mean")).sort("direction").rows())


if __name__ == "__main__":
    main()
