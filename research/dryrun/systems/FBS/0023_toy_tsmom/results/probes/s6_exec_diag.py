"""S6 red team: execution-path diagnostics on the selected config (trial 492) -- rollover spread,
spread-triggered short stops on the M1 path, M1 vs no-M1, and a USDJPY side split.
Evaluations (diagnostic): EURUSD 492 base (repeat of a logged trial), EURUSD 492 use_m1=False,
EURUSD 492 with 00:00-bar spread replaced (custom bars), USDJPY 492 base = 4.
"""
import warnings
import numpy as np, polars as pl
warnings.filterwarnings("ignore")
from quantlab import data
from quantlab.evaluators import RuleEvaluator
from quantlab.strategies.toy_tsmom import ToyTsmom
from quantlab.metrics import summary

P = dict(lookback=80, stop_mult=1.0, hold=18)


def main():
    ev = RuleEvaluator(ToyTsmom, "EURUSD", "H4", book="FBS")
    o = ev(P); t = o.trades
    print("trade columns:", t.columns)
    print("SR base", round(o.metrics["sharpe"], 3), "trades", t.height)
    d = ev._data(); b = d.bars_eval
    # rollover / spread by entry & exit hour
    t = t.with_columns(pl.col("entry_ts").dt.hour().alias("eh"), pl.col("exit_ts").dt.hour().alias("xh"))
    print(t.group_by("eh").agg(pl.len(), pl.col("spread_cost_points").mean().round(1)).sort("eh"))
    tot = t["spread_cost_points"].sum()
    roll = t.filter(((pl.col("direction") == 1) & (pl.col("eh") == 0)) | ((pl.col("direction") == -1) & (pl.col("xh") == 0)))
    print(f"spread paid on 00:00-bar legs: {roll['spread_cost_points'].sum()/tot:.1%} of total spread; "
          f"{roll.height} trades, mean {roll['spread_cost_points'].mean():.1f} pts vs others "
          f"{t.join(roll.select('entry_idx'), on='entry_idx', how='anti')['spread_cost_points'].mean():.1f}")
    # M1 spread at 00:00 vs rest of the first hour
    m1 = d.m1.with_columns(pl.col("ts").dt.hour().alias("h"), pl.col("ts").dt.minute().alias("mi"))
    f = m1.filter(pl.col("h") == 0)
    print("M1 spread 00:00 minute median/mean:", f.filter(pl.col("mi") == 0)["spread"].median(), round(f.filter(pl.col("mi") == 0)["spread"].mean(), 1),
          "| 00:05-00:59 median:", f.filter(pl.col("mi") >= 5)["spread"].median(), "| all hours median:", m1["spread"].median())
    # spread-only short stops: stop reached on Ask but not on Bid within the exit bar's M1 path
    ss = t.filter((pl.col("direction") == -1) & pl.col("exit_reason").is_in(["stop", "gap_stop"]))
    pt = d.spec.point
    n_spread_only = 0; hours = []
    for r in ss.iter_rows(named=True):
        lvl = r.get("stop_price") if r.get("stop_price") is not None else None
        if lvl is None:
            continue
        x0 = r["exit_ts"]; w = d.m1.filter((pl.col("ts") >= x0) & (pl.col("ts") < x0 + pl.duration(hours=4)))
        hit = w.filter(pl.col("high") + pl.col("spread") * pt >= lvl)
        if hit.height and hit["high"][0] < lvl:
            n_spread_only += 1; hours.append(hit["ts"][0].hour)
    print(f"short stops: {ss.height}; triggered by the Ask but not the Bid (spread-only): {n_spread_only}; hours {sorted(set(hours))}")
    # no-M1
    ev2 = RuleEvaluator(ToyTsmom, "EURUSD", "H4", book="FBS", use_m1=False)
    o2 = ev2(P)
    print("SR use_m1=False", round(o2.metrics["sharpe"], 3), "trades", o2.trades.height)
    # side split and by year (EURUSD) and USDJPY
    for sym in ("EURUSD", "USDJPY"):
        e = ev if sym == "EURUSD" else RuleEvaluator(ToyTsmom, sym, "H4", book="FBS")
        oo = o if sym == "EURUSD" else e(P)
        tt = oo.trades
        pc = "pnl_money" if "pnl_money" in tt.columns else "pnl_points"
        print(sym, "SR", round(oo.metrics["sharpe"], 3), tt.group_by("direction").agg(pl.len(), pl.col(pc).sum().round(0)).sort("direction").rows())
        print("   by year:", tt.group_by(pl.col("entry_ts").dt.year().alias("y")).agg(pl.col(pc).sum().round(0)).sort("y").rows())


if __name__ == "__main__":
    main()
