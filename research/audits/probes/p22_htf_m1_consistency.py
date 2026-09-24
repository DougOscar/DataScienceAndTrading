"""Probe 22 (round 2, N3/N4): can the new HTF-vs-M1 high/low consistency check reject legitimate data,
or miss wrong data? Also crypto artefact-minute dropping vs HTF/M1 consistency."""
import sys, os, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import polars as pl
from quantlab import data
from quantlab.engine import run_backtest
from quantlab.costs import load_instrument

def sig(n):
    return pl.DataFrame({"signal": [None] * n, "stop_dist": [None] * n, "target_dist": [None] * n},
                        schema={"signal": pl.Int8, "stop_dist": pl.Float64, "target_dist": pl.Float64})

def attempt(label, bars, m1, tf, sym="EURUSD"):
    try:
        run_backtest(bars, sig(bars.height), load_instrument(sym), m1=m1, timeframe=tf)
        print(f"[accepted] {label}")
    except ValueError as e:
        print(f"[REJECTED] {label}: {str(e)[:170]}")

h1 = data.load_bars("EURUSD", "H1", start="2019-01-01", end="2020-01-01")
m1 = data.load_bars("EURUSD", "M1", start="2019-01-01", end="2020-01-01")
attempt("EURUSD H1 (resampled) + M1, same window", h1, m1, "H1")
attempt("EURUSD D1 + M1", data.load_bars("EURUSD", "D1", start="2019-01-01", end="2020-01-01"), m1, "D1")
attempt("EURUSD H4 + M1", data.load_bars("EURUSD", "H4", start="2019-01-01", end="2020-01-01"), m1, "H4")
h1file = data.catalog().filter((pl.col("symbol") == "EURUSD") & (pl.col("timeframe") == "H1"))["file"][0]
h1n = data.load_bars("EURUSD", "H1", start="2019-01-01", end="2020-01-01", source_file=h1file)
attempt("EURUSD native-H1 file + M1 (broker's own H1 bars)", h1n, m1, "H1")
j = h1n.join(h1, on="ts", suffix="_m1").filter((pl.col("high") != pl.col("high_m1")) | (pl.col("low") != pl.col("low_m1")))
print(f"    native H1 vs M1-resampled H1: {j.height} of {h1n.height} bars differ in high/low")
attempt("H1 from 2018-12 (warm-up) + M1 from 2019-01-01 (aligned)", data.load_bars("EURUSD", "H1", start="2018-12-01", end="2020-01-01"), m1, "H1")
attempt("H1 + M1 starting mid-bar (2019-01-02 10:30)", h1, m1.filter(pl.col("ts") >= datetime(2019, 1, 2, 10, 30)), "H1")
attempt("D1 to 2019-07-01 + M1 ending mid-day 2019-06-28 12:00", data.load_bars("EURUSD", "D1", start="2019-01-01", end="2019-07-01"),
        m1.filter(pl.col("ts") < datetime(2019, 6, 28, 12)), "D1")
attempt("H1 + M1 with 1 M1 bar missing (the high of 2019-06-06 14:xx)", h1,
        m1.filter(pl.col("ts") != datetime(2019, 6, 6, 14, 45)), "H1")
attempt("D1 bars mis-declared as H1", data.load_bars("EURUSD", "D1", start="2019-01-01", end="2019-07-01"), m1, "H1")
attempt("H1 bars declared as D1", h1, m1, "D1")
attempt("H4 bars declared as H1", data.load_bars("EURUSD", "H4", start="2019-01-01", end="2020-01-01"), m1, "H1")
attempt("H1 of EURUSD + M1 of GBPUSD", h1, data.load_bars("GBPUSD", "M1", start="2019-01-01", end="2020-01-01"), "H1")
# crypto: dropped artefact minutes on the 2017-2020 spring-forward days; fall-back ambiguous hour
for yr, s, e in ((2019, "2019-03-25", "2019-04-05"), (2020, "2020-03-23", "2020-04-03"), (2021, "2021-03-22", "2021-04-02"),
                 (2019, "2019-10-21", "2019-11-01")):
    bm1 = data.load_bars("BTCUSD", "M1", start=s, end=e)
    for tf in ("H1", "H4", "D1"):
        attempt(f"BTCUSD {tf} + M1 {s}..{e}", data.load_bars("BTCUSD", tf, start=s, end=e), bm1, tf, "BTCUSD")
    d = bm1.filter(pl.col("ts").dt.date() == pl.lit(s).str.to_date().dt.offset_by("6d"))
    print(f"    BTC {s}+6d: M1 rows with server hour 03 = {d.filter(pl.col('ts').dt.hour() == 3).height}, total rows that day = {d.height}")
an = data.calendar_anomalies("BTCUSD")
print("calendar_anomalies(BTCUSD) columns:", an.columns, "rows:", an.height)
print(an.head(10))
