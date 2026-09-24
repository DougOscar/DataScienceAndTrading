"""Probe 11: sizing / equity. (a) swap_mode='money' swap is dropped from pnl_ccy; (b) lot flooring edge
values; (c) daily_equity marks only with info available at each bar close; (d) risk realised vs target."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from dataclasses import replace
from datetime import datetime
import numpy as np
import polars as pl
from quantlab import data
from quantlab.costs import load_instrument, InstrumentSpec
from quantlab.engine import run_backtest
from quantlab.sizing import apply_sizing, daily_equity, floor_to_step
from sma_strategy import SmaCross, SmaParams

bars = data.load_bars("EURUSD", "H1", start="2019-01-01", end="2020-01-01")
base = load_instrument("EURUSD")
sig = SmaCross(SmaParams(far_target=True)).signals(bars)

money = replace(base, swap_mode="money", swap_long=-7.0, swap_short=2.0)
pts = replace(base, swap_mode="points", swap_long=-7.0, swap_short=2.0)
for lab, sp in (("points", pts), ("money", money)):
    r = run_backtest(bars, sig, sp)
    s = apply_sizing(r.trades, sp, mode="fixed_lots", lots=1.0)
    print(f"(a) swap_mode={lab:6s}: trades' swap_points sum={r.trades['swap_points'].sum():.1f}, swap_money_per_lot sum={r.trades['swap_money_per_lot'].sum():.1f}"
          f" -> total pnl_ccy={s['pnl_ccy'].sum():.2f}")

print("(b) floor_to_step:", {x: floor_to_step(x, 0.01, 0.01, 500) for x in (0.3, 0.29999999, 0.0099999, 0.07, 1.15, 0.57, 0.58, 2.01, 600.0)})

r = run_backtest(bars, sig, base)
s = apply_sizing(r.trades, base, mode="fixed_fraction", risk_fraction=0.01)
print(f"(d) risk realisation error: mean={s['risk_realisation_error'].mean():.4f} min={s['risk_realisation_error'].min():.4f} max={s['risk_realisation_error'].max():.4f} (must be <=0)")
stop_tr = s.filter(pl.col("exit_reason") == "stop")
loss_vs_target = (-stop_tr["pnl_ccy"] / stop_tr["risk_target_ccy"])
print(f"    stop-exit loss / risk target: min={loss_vs_target.min():.4f} max={loss_vs_target.max():.4f}")
eq = daily_equity(bars, r, s)
print(f"(c) final daily equity {eq['equity'][-1]:.2f} vs equity0+sum(pnl_ccy) {100000 + s['pnl_ccy'].sum():.2f}; days={eq.height}")
# truncation test on daily equity: equity up to day D must not change if bars/trades after D are removed
cut = datetime(2019, 7, 1)
bars_t = bars.filter(pl.col("ts") < cut)
r_t = run_backtest(bars_t, sig.head(bars_t.height), base)
s_t = apply_sizing(r_t.trades, base, mode="fixed_fraction", risk_fraction=0.01)
eq_t = daily_equity(bars_t, r_t, s_t)
j = eq_t.join(eq, on="date", suffix="_full")
diff = j.with_columns((pl.col("equity") - pl.col("equity_full")).abs().alias("d")).filter(pl.col("d") > 1e-6)
print(f"    daily equity truncation test at {cut.date()}: differing days = {diff.height} ->", diff.select("date", "equity", "equity_full").to_dicts()[:3])
