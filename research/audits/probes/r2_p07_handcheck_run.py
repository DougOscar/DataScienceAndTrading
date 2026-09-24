"""Probe 07: run SMA 20/50 + 2xATR stop on EURUSD H1 2019 with and without M1; print trades."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, __import__("os").path.dirname(__file__))
import polars as pl
from quantlab import data
from quantlab.costs import load_instrument
from quantlab.engine import run_backtest
from quantlab.testing import assert_no_lookahead
from sma_strategy import SmaCross, SmaParams
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_cols(20)

spec = load_instrument("EURUSD")
print("spec:", spec)
bars = data.load_bars("EURUSD", "H1", start="2019-01-01", end="2020-01-01")
m1 = data.load_bars("EURUSD", "M1", start="2019-01-01", end="2020-01-01")
for far in (False, True):
    strat = SmaCross(SmaParams(far_target=far))
    assert_no_lookahead(strat, bars, n_checks=25)
    sig = strat.signals(bars)
    r0 = run_backtest(bars, sig, spec).trades
    r1 = run_backtest(bars, sig, spec, m1=m1, timeframe="H1").trades
    print(f"\n=== far_target={far}: trades no-m1={r0.height} m1={r1.height}")
    for lab, r in (("no-m1", r0), ("m1", r1)):
        s = r.group_by("direction", "exit_reason").agg(pl.len(), pl.col("pnl_points").sum().round(1), (pl.col("exit_idx") == pl.col("entry_idx")).sum().alias("same_bar")).sort("direction", "exit_reason")
        print(lab, s.to_dicts())
    if far:
        cols = ["entry_ts", "exit_ts", "direction", "entry_price", "exit_price", "stop_price", "exit_reason", "pnl_points", "spread_cost_points", "mae_points", "mfe_points", "nights"]
        r1.select(cols).write_csv(__import__("os").path.join(__import__("os").path.dirname(__file__), "r2_p07_trades_m1.csv"))
        r0.select(cols).write_csv(__import__("os").path.join(__import__("os").path.dirname(__file__), "r2_p07_trades_nom1.csv"))
        j = r0.select("entry_ts", pl.col("exit_ts").alias("x0"), pl.col("exit_reason").alias("r0"), pl.col("pnl_points").alias("p0")).join(
            r1.select("entry_ts", pl.col("exit_ts").alias("x1"), pl.col("exit_reason").alias("r1"), pl.col("pnl_points").alias("p1")), on="entry_ts", how="full")
        print("trades differing between no-m1 and m1:")
        print(j.filter((pl.col("r0") != pl.col("r1")) | (pl.col("p0") != pl.col("p1")) | pl.col("r0").is_null() | pl.col("r1").is_null()))
        print(r1.select(cols).head(40))
