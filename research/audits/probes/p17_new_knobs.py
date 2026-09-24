"""Probe 17: new cost knobs. (a) stop_fill='bar_extreme' vs 'level' with/without M1, and whether the
knob is recorded in CostModel.version; (b) spread_max conservatism on no-M1 short checks (D1 and H1);
(c) residual MAE contamination from the exit sub-bar; (d) money-mode swap currency assumption."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from dataclasses import replace
import polars as pl
from quantlab import data
from quantlab.costs import load_instrument, CostModel
from quantlab.engine import run_backtest
from quantlab.sizing import apply_sizing
from sma_strategy import SmaCross, SmaParams

spec = load_instrument("EURUSD")
h1 = data.load_bars("EURUSD", "H1", start="2019-01-01", end="2020-01-01")
m1 = data.load_bars("EURUSD", "M1", start="2019-01-01", end="2020-01-01")
sig = SmaCross(SmaParams()).signals(h1)
for sf in ("level", "bar_extreme"):
    cm = CostModel(stop_fill=sf)
    a = run_backtest(h1, sig, spec, cm).trades
    b = run_backtest(h1, sig, spec, cm, m1=m1, timeframe="H1").trades
    print(f"(a) stop_fill={sf:11s} version={cm.version!r} stressed.version={cm.stressed().version!r}: "
          f"pnl no-M1={a['pnl_points'].sum():.0f}  with-M1={b['pnl_points'].sum():.0f}")

for tf, start in (("D1", "2016-06-01"), ("H1", "2019-01-01")):
    bb = data.load_bars("EURUSD", tf, start=start, end="2020-01-01")
    mm = data.load_bars("EURUSD", "M1", start=start, end="2020-01-01")
    print(f"(b) EURUSD {tf}: median spread={bb['spread'].median()} median spread_max={bb['spread_max'].median()} "
          f"p90 spread_max={bb['spread_max'].quantile(0.9)} max={bb['spread_max'].max()}")
    s = SmaCross(SmaParams(fast=10 if tf == "D1" else 20, slow=30 if tf == "D1" else 50)).signals(bb).with_columns(
        (pl.col("stop_dist") * 0.5).alias("target_dist"))
    r0 = run_backtest(bb, s, spec).trades.filter(pl.col("direction") == -1)
    r1 = run_backtest(bb, s, spec, m1=mm, timeframe=tf).trades.filter(pl.col("direction") == -1)
    print(f"    shorts no-M1: n={r0.height} pnl={r0['pnl_points'].sum():.0f} stops={(r0['exit_reason']=='stop').sum()} "
          f"| with-M1: pnl={r1['pnl_points'].sum():.0f} stops={(r1['exit_reason']=='stop').sum()}")

tr = run_backtest(h1, SmaCross(SmaParams(far_target=True)).signals(h1), spec, m1=m1, timeframe="H1").trades
st = tr.filter(pl.col("exit_reason") == "stop")
ratio = st["mae_points"] / (-st["pnl_points"])
print(f"(c) with M1: stop exits {st.height}; MAE > 1.05x loss in {(ratio > 1.05).sum()}, max ratio {ratio.max():.2f} "
      f"(exit sub-bar's full range is included; the fill happened before that range completed)")
w = st.with_columns(ratio.alias("ratio")).sort("ratio", descending=True).head(1).row(0, named=True)
print("    worst:", {k: w[k] for k in ("entry_ts", "exit_ts", "direction", "entry_price", "exit_price", "pnl_points", "mae_points")})

# (d) money-mode swap for a JPY-quoted pair in a USD account: MT5 CURRENCY_DEPOSIT means the amount is already USD
jspec = replace(load_instrument("USDJPY"), swap_mode="money", swap_long=-5.0, swap_short=-5.0)
jb = data.load_bars("USDJPY", "H1", start="2019-01-01", end="2019-04-01")
js = SmaCross(SmaParams(far_target=True)).signals(jb)
r = run_backtest(jb, js, jspec)
rate_fn = lambda q, a, ts: data.conversion_rate(q, a, ts)
sz = apply_sizing(r.trades, jspec, mode="fixed_lots", lots=1.0, rate_fn=rate_fn, bars=jb)
sw_money = r.trades["swap_money_per_lot"].sum()
sz0 = apply_sizing(r.trades.with_columns(pl.lit(0.0).alias("swap_money_per_lot")), jspec, mode="fixed_lots", lots=1.0, rate_fn=rate_fn, bars=jb)
print(f"(d) USDJPY money-mode swap -5/lot/night: sum swap_money_per_lot={sw_money:.1f}; booked in pnl_ccy (USD) = "
      f"{sz['pnl_ccy'].sum() - sz0['pnl_ccy'].sum():.2f} -> treated as JPY (x 1/USDJPY); if the export says "
      f"CURRENCY_DEPOSIT the correct USD amount is {sw_money:.1f}")
