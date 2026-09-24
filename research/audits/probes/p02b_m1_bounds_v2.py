"""Probe 02b (re-verification of M1 + attacks on the new span logic). Same scenarios as p02, adapted to
_map_m1_bounds(htf_ts, m1_ts, span) and run_backtest(timeframe=...), plus new attacks:
 D) sparse/irregular HTF frame (one H1 bar per day) with timeframe omitted -> span inferred as 1 day;
 E) timeframe declared SMALLER than the real bars (D1 bars, timeframe='H1') -> window truncated."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from quantlab import data
from quantlab.engine import run_backtest, _map_m1_bounds, _infer_span
from quantlab.costs import load_instrument
from quantlab.testing import assert_engine_causal
from sma_strategy import SmaCross, SmaParams

spec = load_instrument("EURUSD")
def sig_last(n, k=2, sd=0.0030, td=0.0080, d=1):
    return pl.DataFrame({"signal": [None] * (n - k) + [d] + [None] * (k - 1), "stop_dist": [sd] * n, "target_dist": [td] * n},
                        schema={"signal": pl.Int8, "stop_dist": pl.Float64, "target_dist": pl.Float64})

# A) D1 + M1 with the same intraday end
end = datetime(2019, 6, 14, 12, 0)
d1 = data.load_bars("EURUSD", "D1", start="2019-05-01", end=end)
m1 = data.load_bars("EURUSD", "M1", start="2019-05-01", end=end)
s, e = _map_m1_bounds(d1["ts"], m1["ts"], _infer_span(d1["ts"].to_numpy()))
print("A) inferred D1 span:", _infer_span(d1["ts"].to_numpy()), "| M1 rows mapped to last D1 bar", d1["ts"][-1], "=", e[-1] - s[-1],
      "| last mapped M1:", m1["ts"][int(e[-1]) - 1])

# B) H1 bars end 2019-06-01, M1 to 2019-07-01; long opened on the last H1 bar
h1 = data.load_bars("EURUSD", "H1", start="2019-05-01", end="2019-06-01")
m1b = data.load_bars("EURUSD", "M1", start="2019-05-01", end="2019-07-01")
for kw in ({}, {"timeframe": "H1"}):
    t = run_backtest(h1, sig_last(h1.height), spec, m1=m1b, **kw).trades.row(0, named=True)
    print(f"B) timeframe={kw.get('timeframe')}: exit {t['exit_reason']} @ {t['exit_price']:.5f} (p02 pre-fix: target @ 1.12528 from 2019-06-03 21:50)")

# C) session-filtered frame 08-16h
h1s = h1.filter(pl.col("ts").dt.hour().is_between(8, 16))
sc, ec = _map_m1_bounds(h1s["ts"], m1b["ts"], _infer_span(h1s["ts"].to_numpy()))
print("C) session-filtered: inferred span", _infer_span(h1s["ts"].to_numpy()), "| max M1 rows per H1 bar", int((ec - sc).max()))

# D) sparse frame: only the 08:00 H1 bar of each day, timeframe omitted
h1d = h1.filter(pl.col("ts").dt.hour() == 8)
sp = _infer_span(h1d["ts"].to_numpy())
sd_, ed_ = _map_m1_bounds(h1d["ts"], m1b["ts"], sp)
print("D) one-bar-per-day H1 frame: inferred span", sp, "| max M1 rows per 'H1' bar", int((ed_ - sd_).max()),
      "| last bar", h1d["ts"][-1], "maps M1 up to", m1b["ts"][int(ed_[-1]) - 1])
res = {}
for kw in ({}, {"timeframe": "H1"}):
    t = run_backtest(h1d, sig_last(h1d.height, sd=0.0030, td=0.0015), spec, m1=m1b, **kw).trades.row(0, named=True)
    res[kw.get("timeframe")] = t
    print(f"   timeframe={kw.get('timeframe')}: last-bar long exit {t['exit_reason']} @ {t['exit_price']:.5f}")

# E) D1 bars declared as H1
d1b = data.load_bars("EURUSD", "D1", start="2019-01-01", end="2019-07-01")
m1c = data.load_bars("EURUSD", "M1", start="2019-01-01", end="2019-07-01")
sig = SmaCross(SmaParams(fast=5, slow=20, far_target=False)).signals(d1b).with_columns((pl.col("stop_dist") * 0.5).alias("target_dist"))
r_ok = run_backtest(d1b, sig, spec, m1=m1c, timeframe="D1").trades
r_bad = run_backtest(d1b, sig, spec, m1=m1c, timeframe="H1").trades
r_inf = run_backtest(d1b, sig, spec, m1=m1c).trades
print(f"E) D1 SMA5/20 stop 2ATR target 1ATR: timeframe='D1' pnl={r_ok['pnl_points'].sum():.0f} reasons={dict(r_ok.group_by('exit_reason').len().iter_rows())}")
print(f"   timeframe='H1' (mis-declared, no error raised): pnl={r_bad['pnl_points'].sum():.0f} reasons={dict(r_bad.group_by('exit_reason').len().iter_rows())}")
print(f"   timeframe omitted (inferred): pnl={r_inf['pnl_points'].sum():.0f}")

# F) assert_engine_causal with the natural WFO pattern (full-year M1, HTF slices), stop+target SMA
h1y = data.load_bars("EURUSD", "H1", start="2019-01-01", end="2020-01-01")
m1y = data.load_bars("EURUSD", "M1", start="2019-01-01", end="2020-01-01")
class S(SmaCross):
    def signals(self, b):
        return super().signals(b).with_columns((pl.col("stop_dist") * 0.5).alias("target_dist"))
try:
    assert_engine_causal(S(SmaParams()), h1y, spec, m1=m1y, n_checks=40)
    print("F) assert_engine_causal (H1 slices + full-year M1): PASSED")
except AssertionError as ex:
    print("F) assert_engine_causal FAILED:", str(ex)[:200])
