"""Probe 12: stop fills in a liquidity gap *inside* an M1 bar (USDJPY flash crash 2019-01-03) and across
a weekend gap (real data)."""
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import polars as pl
from quantlab import data
from quantlab.costs import load_instrument
from quantlab.engine import run_backtest

spec = load_instrument("USDJPY")
m1 = data.load_bars("USDJPY", "M1", start="2019-01-02", end="2019-01-04")
worst = m1.sort("low").head(3)
print("USDJPY M1 lowest bars 2019-01-02..03:", worst.select("ts", "open", "high", "low", "close", "spread").to_dicts())
h1 = data.load_bars("USDJPY", "H1", start="2019-01-02", end="2019-01-04")
n = h1.height
i0 = h1.with_row_index().filter(pl.col("ts") == datetime(2019, 1, 2, 23))["index"][0]
sig = pl.DataFrame({"signal": [None] * i0 + [1] + [None] * (n - i0 - 1), "stop_dist": [1.00] * n, "target_dist": [None] * n},
                   schema={"signal": pl.Int8, "stop_dist": pl.Float64, "target_dist": pl.Float64})
for lab, kw in (("no m1", {}), ("m1", {"m1": m1})):
    t = run_backtest(h1, sig, spec, **kw).trades.row(0, named=True)
    print(f"{lab:5s}: long entry {t['entry_ts']} @ {t['entry_price']:.3f}, stop {t['stop_price']:.3f} -> exit {t['exit_ts']} {t['exit_reason']} @ {t['exit_price']:.3f}")
first_hit = m1.filter((pl.col("ts") >= datetime(2019, 1, 3, 0)) & (pl.col("low") <= t["stop_price"])).head(1)
print("first M1 bar through the stop:", first_hit.select("ts", "open", "high", "low", "close").to_dicts())

# weekend gap: find the largest Friday-close -> Monday-open gap for EURUSD/USDJPY/XAUUSD in 2019-2020
for sym in ["EURUSD", "USDJPY", "XAUUSD"]:
    m = data.load_bars(sym, "M1", start="2019-01-01", end="2021-01-01")
    g = m.with_columns(pl.col("close").shift(1).alias("pc"), pl.col("ts").shift(1).alias("pts")).filter(
        (pl.col("ts") - pl.col("pts")) > pl.duration(hours=24)).with_columns((pl.col("open") - pl.col("pc")).abs().alias("gap"))
    r = g.sort("gap", descending=True).head(1).row(0, named=True)
    print(f"{sym}: largest weekend gap {r['pts']} close {r['pc']} -> {r['ts']} open {r['open']} (|gap|={r['gap']:.5f}, spread at Monday open {r['spread']})")
