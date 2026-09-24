"""P05 — Can an edge that exists ONLY in 2016-05 .. ~2019-02 (first 30% of dev) and is
exactly zero afterwards PASS every DESIGN §4.2 gate?

Planted: broad Gaussian bump (height H annual Sharpe, width 0.35 in unit space) on a 7x7
grid, trial correlation rho=0.5, 100 trades/yr.  After frac 0.30 the surface is flat 0
for every configuration.  Mechanism gate is forced True (we test the statistical gates).
Reported: verdict, per-gate status, and the anchored WFO OOS Sharpe from 2019 on (which
exposes the dead edge but is not a gate).
"""
import sys
import numpy as np
from quantlab import opt
from quantlab.evaluators import SyntheticEvaluator
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from e2e_common import ExploitEval, run_and_gate, wfo_sharpe_after

space = opt.SearchSpace([opt.IntParam("a", 0, 6), opt.IntParam("b", 0, 6)])
for H in (3.0, 3.5, 4.0):
    verdicts = []
    for seed in range(6):
        inner = SyntheticEvaluator(bounds={"a": (0, 6), "b": (0, 6)},
                                   bumps=({"center": {"a": 3, "b": 3}, "height": H, "width": 0.35},),
                                   rho=0.5, seed=100 + seed, trades_per_year=100,
                                   regimes=({"from": 0.30, "bumps": (), "base_sharpe": 0.0},))
        ev = ExploitEval(inner)
        res, rep = run_and_gate(ev, space, f"fbs-9999-dead-{H}-{seed}")
        v = rep.values()
        st = rep.statuses()
        fails = [g for g, s in st.items() if s != "PASS"]
        print(f"H={H} seed={seed} verdict={rep.verdict:10s} DSR={v['dsr']:.3f} PBO={v['pbo']:.2f} "
              f"OOS={v['oos_sharpe']:.2f} stress={v['cost_stress_sharpe']:.2f} plateau={v['plateau']:.2f} "
              f"posY={v['positive_years']:.2f} maxY={v['max_year_share']:.2f} trades={st['trade_count']} "
              f"| WFO OOS Sharpe 2019+ = {wfo_sharpe_after(res):.2f} | not-pass={fails}")
        verdicts.append(rep.verdict)
    print(f"  => H={H}: PASS {verdicts.count('PASS')}/{len(verdicts)}")
