"""P05c — dead-since-~2021 edge (P05b, dead_from=0.55, H=3.0): for every study that PASSES the
gates, (1) Sharpe of the selected trial over the last 45% of dev (true value 0), (2) Sharpe of the
anchored-WFO OOS series over the same span (not gated), (3) probability that a ZERO-edge
holdout year passes the pre-registered §4.4 band (holdout_check)."""
import sys
import numpy as np, polars as pl, math
from quantlab import opt
from quantlab.evaluators import SyntheticEvaluator
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from e2e_common import ExploitEval, run_and_gate, recent_sharpe, dead_holdout_pass_rate

space = opt.SearchSpace([opt.IntParam("a", 0, 6), opt.IntParam("b", 0, 6)])
FROM, H = 0.55, 3.0
for seed in range(8):
    inner = SyntheticEvaluator(bounds={"a": (0, 6), "b": (0, 6)},
                               bumps=({"center": {"a": 3, "b": 3}, "height": H, "width": 0.35},),
                               rho=0.5, seed=100 + seed, trades_per_year=100,
                               regimes=({"from": FROM, "bumps": (), "base_sharpe": 0.0},))
    ev = ExploitEval(inner)
    res, rep = run_and_gate(ev, space, f"fbs-9999-deadho-{seed}")
    w = res.wfo_oos
    cut = res.returns["date"][int(FROM * res.returns.height)]
    wr = w.filter(pl.col("date") >= cut)["ret"].to_numpy()
    wsr = wr.mean() / wr.std(ddof=1) * math.sqrt(260)
    hb = rep.holdout_band
    print(f"seed={seed} verdict={rep.verdict:5s} selected-trial Sharpe last45%={recent_sharpe(res, FROM):+.2f} "
          f"WFO-OOS Sharpe last45%={wsr:+.2f} | band sharpe_p10={hb['sharpe_p10']:+.2f} "
          f"median={hb['sharpe_median']:.2f} | P(dead holdout passes §4.4)={dead_holdout_pass_rate(res, rep):.2f}")
