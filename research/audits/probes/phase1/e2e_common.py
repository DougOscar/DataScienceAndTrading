"""Shared harness for the end-to-end exploit probes (p05+).

Wraps SyntheticEvaluator so that the full gate set can run:
  * cost=<CostModel>: subtract a Sharpe drag proportional to the stress (spread x, slippage,
    |swap-1|) -- 0.15 annual-Sharpe units per stress unit, so a 1.5x/+1pt stress costs ~0.3;
  * trades carry pnl_ccy / equity_before built from the daily returns (per-trade MinTRL runs).
All ledgers/studies go to a tmp dir; research/ledger is never touched.
"""
from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quantlab import gates as G, opt
from quantlab.contracts import Outcome
from quantlab.costs import CostModel
from quantlab.evaluators import SyntheticEvaluator


@dataclass
class ExploitEval:
    inner: SyntheticEvaluator
    trades_per_year: float = 100.0
    drag_per_unit: float = 0.15
    cost: CostModel = field(default_factory=lambda: CostModel(version_tag="fbs-probe"))
    book: str = "FBS"
    periods_per_year: float = 260.0
    requires_refit = False

    @property
    def cost_version(self):
        return "probe"

    @property
    def dev_window(self):
        return self.inner.dev_window

    def describe(self):
        return {"evaluator": "ExploitEval", **self.inner.describe()}

    def prepare(self):
        pass

    def __call__(self, params, *, cost=None):
        c = self.cost if cost is None or not isinstance(cost, CostModel) else cost
        units = (c.spread_multiplier - 1.0) * 2 + c.slippage_points + abs(c.swap_multiplier - 1.0)
        o = self.inner(params, cost=self.drag_per_unit * units)
        d = o.daily
        r = d["ret"].to_numpy()
        n = r.size
        n_tr = int(round(self.trades_per_year * n / self.periods_per_year))
        rng = np.random.default_rng(abs(hash(tuple(sorted(params.items())))) % 2**32)
        ent = np.sort(rng.choice(n, size=min(n_tr, n), replace=False))
        b = np.r_[ent, n]
        pnl = np.add.reduceat(r, ent) if ent.size else np.zeros(0)
        ts = [datetime.combine(d["date"][int(i)], datetime.min.time()) + timedelta(hours=10) for i in ent]
        xs = [datetime.combine(d["date"][int(min(j - 1, n - 1))], datetime.min.time()) + timedelta(hours=20)
              for j in b[1:]]
        trades = pl.DataFrame({"entry_ts": ts, "exit_ts": xs, "pnl_ccy": pnl * 1e5,
                               "equity_before": [1e5] * len(ent), "skipped": [False] * len(ent)})
        m = dict(o.metrics)
        m["n_trades"] = float(len(ent))
        hold = np.diff(b)
        m["hold_days_max"] = float(min(hold.max(), 10)) if hold.size else 0.0
        return Outcome(daily=d, trades=trades, metrics=m)


def mech_true(study, evaluator):
    return True, "probe: mechanism assumed confirmed"


def run_and_gate(ev, space, sid, *, method="grid", n_trials=None, wfo=opt.WFOConfig(), seed=0, **kw):
    tmp = Path(tempfile.mkdtemp(prefix="rt_phase1_"))
    res = opt.run_study(ev, space, book="FBS", system="probe", issue=9999, attempt=1, study_id=sid,
                        method=method, n_trials=n_trials, n_jobs=1, wfo=wfo, seed=seed,
                        ledger_dir=tmp / "ledger", studies_dir=tmp / "studies", **kw)
    rep = G.evaluate_gates(res, ev, periods_per_year=260.0, mechanism_check=mech_true, n_boot=500)
    return res, rep


def wfo_sharpe_after(res, year=2019):
    w = res.wfo_oos
    if w is None or w.height == 0:
        return float("nan")
    w = w.filter(pl.col("date").dt.year() >= year)["ret"].to_numpy()
    return float(w.mean() / w.std(ddof=1) * math.sqrt(260)) if w.size > 2 and w.std() > 0 else float("nan")


def recent_sharpe(res, frac_from):
    """Annualised Sharpe of the SELECTED trial over rows >= frac_from of the dev window."""
    tid = res.selection["trial_id"]
    r = res.returns[f"t{tid}"].to_numpy()
    r = r[int(frac_from * r.size):]
    return float(r.mean() / r.std(ddof=1) * math.sqrt(260))


def dead_holdout_pass_rate(res, rep, vol=0.005, n_sim=400, seed=0, trades_per_year=100):
    """Pass rate of stats.holdout_check on 1-year holdouts drawn from a ZERO-edge process
    with the dev volatility and the dev trade rate (the edge is dead in the holdout)."""
    from quantlab import stats as st
    band = rep.holdout_band
    if band is None:
        return float("nan")
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n_sim):
        h = rng.standard_normal(260) * vol
        n_tr = rng.poisson(trades_per_year)
        ok += st.holdout_check(band, h, n_tr, check_horizon=False)["criteria_pass"]   # all four criteria
    return ok / n_sim
