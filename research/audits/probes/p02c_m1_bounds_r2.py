"""Probe 02c (round 2): M1/N3/N4 on the round-2 API (timeframe required with m1; HTF/M1 consistency check)."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from datetime import datetime
import polars as pl
from quantlab import data
from quantlab.engine import run_backtest
from quantlab.costs import load_instrument
from quantlab.testing import assert_engine_causal
from sma_strategy import SmaCross, SmaParams

spec = load_instrument("EURUSD")
def sig_last(n, k=2, sd=0.0030, td=0.0080, d=1):
    return pl.DataFrame({"signal": [None] * (n - k) + [d] + [None] * (k - 1), "stop_dist": [sd] * n, "target_dist": [td] * n},
                        schema={"signal": pl.Int8, "stop_dist": pl.Float64, "target_dist": pl.Float64})
def tryrun(label, *a, **k):
    try:
        t = run_backtest(*a, **k).trades
        print(f"{label}: ran -> {t.select('exit_reason', 'exit_price').row(-1) if t.height else 'no trades'}")
        return t
    except ValueError as e:
        print(f"{label}: REJECTED ({str(e)[:110]})")

h1 = data.load_bars("EURUSD", "H1", start="2019-05-01", end="2019-06-01")
m1b = data.load_bars("EURUSD", "M1", start="2019-05-01", end="2019-07-01")
tryrun("B) last-bar long, H1 + M1 to July, timeframe='H1'", h1, sig_last(h1.height), spec, m1=m1b, timeframe="H1")
tryrun("B') same, timeframe omitted", h1, sig_last(h1.height), spec, m1=m1b)
h1d = h1.filter(pl.col("ts").dt.hour() == 8)
tryrun("D) one-bar-per-day frame, timeframe='H1'", h1d, sig_last(h1d.height, td=0.0015), spec, m1=m1b, timeframe="H1")
tryrun("D') one-bar-per-day frame, timeframe='D1' (overstated)", h1d, sig_last(h1d.height, td=0.0015), spec, m1=m1b, timeframe="D1")
d1b = data.load_bars("EURUSD", "D1", start="2019-01-01", end="2019-07-01")
m1c = data.load_bars("EURUSD", "M1", start="2019-01-01", end="2019-07-01")
s = SmaCross(SmaParams(fast=5, slow=20)).signals(d1b).with_columns((pl.col("stop_dist") * 0.5).alias("target_dist"))
tryrun("E) D1 bars declared H1", d1b, s, spec, m1=m1c, timeframe="H1")
tryrun("E') D1 bars declared D1", d1b, s, spec, m1=m1c, timeframe="D1")
# overstated timeframe on a WFO-style slice with full M1: H1 slice declared D1 -> last-bar window would reach 23h ahead
h1slice = data.load_bars("EURUSD", "H1", start="2019-05-01", end="2019-05-20 13:00")
tryrun("G) H1 slice declared D1 with M1 running past the slice", h1slice, sig_last(h1slice.height), spec, m1=m1b, timeframe="D1")
h1y = data.load_bars("EURUSD", "H1", start="2019-01-01", end="2020-01-01")
m1y = data.load_bars("EURUSD", "M1", start="2019-01-01", end="2020-01-01")
class S(SmaCross):
    def signals(self, b):
        return super().signals(b).with_columns((pl.col("stop_dist") * 0.5).alias("target_dist"))
try:
    assert_engine_causal(S(SmaParams()), h1y, spec, m1=m1y, timeframe="H1", n_checks=40)
    print("F) assert_engine_causal(timeframe='H1', full-year M1, 40 cuts): PASSED")
except AssertionError as ex:
    print("F) FAILED", str(ex)[:200])
