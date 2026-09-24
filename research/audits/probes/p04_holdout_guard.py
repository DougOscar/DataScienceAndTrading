"""Probe 04: try every public path to get FBS bars at/after holdout_start (2025-05-15 00:00 server)
without a ledger unlock. PASS = HoldoutLocked/TypeError/ValueError or data strictly before cutoff
(and, for resampled bars, ts+span <= cutoff). NOTE: no request here returns raw holdout rows to
the auditor -- every successful call is immediately reduced to max(ts)."""
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import polars as pl
from quantlab import data, config
from quantlab.contracts import HoldoutLocked

CUT = config.BOOKS["FBS"].holdout_start
SPAN = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}

def attempt(label, fn, tf="M1"):
    try:
        df = fn()
    except HoldoutLocked as e:
        print(f"[blocked HoldoutLocked] {label}"); return
    except Exception as e:
        print(f"[blocked {type(e).__name__}] {label}: {str(e)[:90]}"); return
    if df.height == 0:
        print(f"[ok, empty] {label}"); return
    last = df["ts"].max()
    close_of_last = last + timedelta(minutes=SPAN[tf])
    leak = close_of_last > CUT
    print(f"[{'LEAK' if leak else 'ok'}] {label}: last ts={last}, last bar closes {close_of_last}")

L = data.load_bars
for tf in ["M1", "M5", "H1", "H4", "D1"]:
    attempt(f"default end, {tf}", lambda tf=tf: L("EURUSD", tf, start="2025-05-01"), tf)
    attempt(f"end == cutoff, {tf}", lambda tf=tf: L("EURUSD", tf, start="2025-05-01", end=CUT), tf)
attempt("end = cutoff+1ms", lambda: L("EURUSD", "M1", start="2025-05-10", end=CUT + timedelta(milliseconds=1)))
attempt("end = '2025-05-15 00:00:00.000001'", lambda: L("EURUSD", "M1", start="2025-05-10", end="2025-05-15 00:00:00.000001"))
attempt("start after cutoff, no end", lambda: L("EURUSD", "M1", start="2025-06-01"))
attempt("start after cutoff, end > cutoff", lambda: L("EURUSD", "M1", start="2025-06-01", end="2025-07-01"))
attempt("end tz-aware UTC (= 2025-05-15 00:00 UTC = 03:00 server)", lambda: L("EURUSD", "M1", start="2025-05-10", end=datetime(2025, 5, 15, tzinfo=timezone.utc)))
attempt("end tz-aware string +00:00", lambda: L("EURUSD", "M1", start="2025-05-10", end="2025-05-15T02:00:00+00:00"))
attempt("end tz-aware -05:00 on dev day (2025-05-14 20:00 NY = 05-15 03:00 server)", lambda: L("EURUSD", "M1", start="2025-05-10", end="2025-05-14T20:00:00-04:00"))
attempt("include_holdout=True, no system", lambda: L("EURUSD", "M1", start="2025-05-10", end="2025-06-01", include_holdout=True))
attempt("system given, include_holdout False", lambda: L("EURUSD", "M1", start="2025-05-10", end="2025-06-01", system="x"))
attempt("system not unlocked + include_holdout", lambda: L("EURUSD", "M1", start="2025-05-10", end="2025-06-01", system="nope", include_holdout=True))
attempt("book='fbs' lowercase", lambda: L("EURUSD", "M1", book="fbs", start="2025-05-10", end="2025-06-01"))
attempt("book='B3' for FX symbol", lambda: L("EURUSD", "M1", book="B3", start="2025-05-10"))
h1file = data.catalog().filter((pl.col("symbol") == "EURUSD") & (pl.col("timeframe") == "H1"))["file"][0]
attempt("source_file = native H1 file, H1", lambda: L("EURUSD", "H1", start="2025-05-01", source_file=h1file), "H1")
attempt("source_file = native H1 file, D1", lambda: L("EURUSD", "D1", start="2025-05-01", source_file=h1file), "D1")
attempt("source_file = native H1 file, H4 (relative path)", lambda: L("EURUSD", "H4", start="2025-05-01", source_file=h1file.split("DataScienceAndTrading/")[1]), "H4")
attempt("source_file = XAUUSD file but symbol EURUSD", lambda: L("EURUSD", "D1", start="2025-05-01", source_file=data.catalog().filter(pl.col("symbol")=="XAUUSD")["file"][0]), "D1")
# straddling H4/D1 bars with an intraday end just before cutoff
attempt("H4, end=2025-05-14 22:00", lambda: L("EURUSD", "H4", start="2025-05-12", end="2025-05-14 22:00"), "H4")
attempt("D1, end=2025-05-14 23:59", lambda: L("EURUSD", "D1", start="2025-05-10", end="2025-05-14 23:59"), "D1")
# BTC (24/7) D1 bar ending exactly at cutoff
attempt("BTCUSD D1 default", lambda: L("BTCUSD", "D1", start="2025-05-10"), "D1")
# conversion_rate: timestamps in the holdout
ts_h = pl.Series("t", [datetime(2025, 5, 20, 12)]).dt.replace_time_zone("UTC")
attempt("conversion_rate EUR->USD at 2025-05-20", lambda: data.conversion_rate("EUR", "USD", ts_h).to_frame().with_columns(pl.lit(datetime(2025,5,20,12)).alias("ts")))
ts_edge = pl.Series("t", [datetime(2025, 5, 14, 20, 59, 30)]).dt.replace_time_zone("UTC")  # 23:59:30 server
attempt("conversion_rate at 2025-05-14 23:59:30 server (window end +1min crosses cutoff)", lambda: data.conversion_rate("EUR", "USD", ts_edge).to_frame().with_columns(pl.lit(datetime(2025,5,14,23,59)).alias("ts")))
ts_naive = pl.Series("t", [datetime(2025, 5, 14, 22, 30)])
attempt("conversion_rate with NAIVE series (2025-05-14 22:30)", lambda: data.conversion_rate("EUR", "USD", ts_naive).to_frame().with_columns(pl.lit(datetime(2025,5,14)).alias("ts")))
attempt("conversion_rate via triangulation EUR->JPY at 2025-05-20", lambda: data.conversion_rate("EUR", "JPY", ts_h).to_frame().with_columns(pl.lit(datetime(2025,5,20)).alias("ts")))

# infer_point reads head+tail of the raw file with no guard
import inspect
src = inspect.getsource(data.infer_point)
print("infer_point reads raw file tail without load_bars guard:", ".tail(sample)" in src and "load_bars" not in src)
# cache content
from quantlab.data import _load_resampled
from pathlib import Path
row = data.catalog().filter((pl.col("symbol") == "EURUSD") & (pl.col("timeframe") == "M1")).row(0, named=True)
mx = _load_resampled(Path(row["file"]), "M1", "D1").select(pl.col("ts").max()).collect().item()
print("private _load_resampled returns full series up to", mx, "(private helper; cache parquet on disk also unguarded)")
