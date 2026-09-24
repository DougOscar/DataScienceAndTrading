"""Probe 02: the M1 path for the LAST HTF bar is m1[searchsorted(ts_last):] -- i.e. every
remaining M1 bar, not just [ts_last, ts_last+span). If the caller's m1 frame extends past
the HTF frame (different end, or D1 bars whose partial last day is dropped by load_bars'
whole-bucket rule while M1 keeps it), stops/targets on the last bar are resolved with
FUTURE prices. Same for any interior HTF gap (bars filtered by session/hour)."""
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np
import polars as pl
from quantlab import data
from quantlab.engine import run_backtest, _map_m1_bounds
from quantlab.costs import load_instrument

spec = load_instrument("EURUSD")
# Case A: same `end` for both loads, D1 frame; end is intraday -> last D1 bar dropped for D1 but M1 keeps the minutes
end = datetime(2019, 6, 14, 12, 0)
d1 = data.load_bars("EURUSD", "D1", start="2019-05-01", end=end)
m1 = data.load_bars("EURUSD", "M1", start="2019-05-01", end=end)
s, e = _map_m1_bounds(d1["ts"], m1["ts"])
last_ts = d1["ts"][-1]
print("A) D1 last bar ts:", last_ts, "| M1 rows mapped to it:", e[-1] - s[-1],
      "| M1 ts range:", m1["ts"][int(s[-1])], "->", m1["ts"][int(e[-1]) - 1])

# Case B: HTF bars end 2019-06-01, m1 loaded to 2019-07-01 (a natural 'load M1 once' pattern)
h1 = data.load_bars("EURUSD", "H1", start="2019-05-01", end="2019-06-01")
m1b = data.load_bars("EURUSD", "M1", start="2019-05-01", end="2019-07-01")
n = h1.height
# go long at the second-to-last bar's signal, tight stop 30 pips; no other signals
sig = pl.DataFrame({"signal": [None] * (n - 2) + [1, None],
                    "stop_dist": [0.0030] * n, "target_dist": [0.0080] * n},
                   schema={"signal": pl.Int8, "stop_dist": pl.Float64, "target_dist": pl.Float64})
r_no = run_backtest(h1, sig, spec).trades
r_m1 = run_backtest(h1, sig, spec, m1=m1b).trades
print("B) HTF last bar:", h1["ts"][-1], "high/low", h1["high"][-1], h1["low"][-1])
print("   without m1:", r_no.select("entry_ts", "exit_ts", "exit_reason", "entry_price", "stop_price", "target_price", "exit_price").to_dicts())
print("   with m1   :", r_m1.select("entry_ts", "exit_ts", "exit_reason", "entry_price", "stop_price", "target_price", "exit_price").to_dicts())
sb, eb = _map_m1_bounds(h1["ts"], m1b["ts"])
print("   M1 bars mapped to last H1 bar:", eb[-1] - sb[-1], "(should be <= 60)")
if r_m1["exit_reason"][0] in ("target", "gap_target"):
    hit = m1b.filter((pl.col("ts") >= h1["ts"][-1]) & (pl.col("high") >= r_m1["target_price"][0])).head(1)
    print("   target was triggered by M1 bar at", hit["ts"][0], "-> outside the last H1 window (future data)")

# Case C: interior gap -- keep only H1 bars 08:00-16:00 (session-filtered frame)
h1s = h1.filter(pl.col("ts").dt.hour().is_between(8, 16))
sc, ec = _map_m1_bounds(h1s["ts"], m1b["ts"])
cnt = ec - sc
print("C) session-filtered H1: max M1 rows mapped to one H1 bar =", int(cnt.max()),
      "at", h1s["ts"][int(np.argmax(cnt))])
