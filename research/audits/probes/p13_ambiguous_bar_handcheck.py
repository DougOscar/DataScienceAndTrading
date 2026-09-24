"""Probe 13: SMA 20/50, stop 2xATR, target 1xATR on EURUSD H1 2019; find trades where the M1 path
changes the outcome vs the no-M1 'stop first' rule and verify one against raw M1 rows."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from datetime import timedelta
import polars as pl
from quantlab import data
from quantlab.costs import load_instrument
from quantlab.engine import run_backtest
from sma_strategy import SmaCross, SmaParams

spec = load_instrument("EURUSD")
bars = data.load_bars("EURUSD", "H1", start="2019-01-01", end="2020-01-01")
m1 = data.load_bars("EURUSD", "M1", start="2019-01-01", end="2020-01-01")
sig = SmaCross(SmaParams()).signals(bars).with_columns((pl.col("stop_dist") / 2.0).alias("target_dist"))
r0 = run_backtest(bars, sig, spec).trades
r1 = run_backtest(bars, sig, spec, m1=m1).trades
k = ["entry_ts", "direction", "entry_price", "stop_price", "target_price"]
j = r0.select(*k, pl.col("exit_ts").alias("x0"), pl.col("exit_reason").alias("r0"), pl.col("exit_price").alias("p0"), pl.col("pnl_points").alias("pnl0")).join(
    r1.select("entry_ts", pl.col("exit_ts").alias("x1"), pl.col("exit_reason").alias("r1"), pl.col("exit_price").alias("p1"), pl.col("pnl_points").alias("pnl1")), on="entry_ts", how="full", coalesce=True)
d = j.filter((pl.col("r0") != pl.col("r1")) | pl.col("r0").is_null() | pl.col("r1").is_null())
print(f"trades no-m1={r0.height} m1={r1.height}; differing={d.height}; sum pnl no-m1={r0['pnl_points'].sum():.0f} m1={r1['pnl_points'].sum():.0f}")
print(d.head(6))
if d.height:
    row = d.filter(pl.col("r1").is_not_null() & pl.col("r0").is_not_null()).row(0, named=True)
    x = row["x1"]
    w = m1.filter((pl.col("ts") >= x) & (pl.col("ts") < x + timedelta(hours=1)))
    P = 1e-5
    if row["direction"] == 1:
        first_sl = w.filter(pl.col("low") <= row["stop_price"]).head(1)
        first_tp = w.filter(pl.col("high") >= row["target_price"]).head(1)
    else:
        first_sl = w.filter(pl.col("high") + pl.col("spread") * P >= row["stop_price"]).head(1)
        first_tp = w.filter(pl.col("low") + pl.col("spread") * P <= row["target_price"]).head(1)
    print(f"HAND CHECK trade entered {row['entry_ts']} dir={row['direction']} entry={row['entry_price']:.5f} SL={row['stop_price']:.6f} TP={row['target_price']:.6f}")
    print("  H1 exit bar:", bars.filter(pl.col("ts") == x).select("ts", "open", "high", "low", "close", "spread").to_dicts())
    print("  first M1 through SL:", first_sl.select("ts", "open", "high", "low", "spread").to_dicts())
    print("  first M1 through TP:", first_tp.select("ts", "open", "high", "low", "spread").to_dicts())
    print(f"  engine no-m1: {row['r0']} @ {row['p0']:.6f} pnl {row['pnl0']:.2f} | engine m1: {row['r1']} @ {row['p1']:.6f} pnl {row['pnl1']:.2f}")
