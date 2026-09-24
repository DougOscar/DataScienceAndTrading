"""Probe 14b (re-verification of M3 end-to-end): apply_sizing(bars=...) must feed rate_fn tz-aware UTC at the
entry bar (sizing) and exit bar (P&L); verify every rate against an independent as-of on closed M1 bars,
and check no rate uses a price from after its timestamp."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from datetime import timedelta
import numpy as np
import polars as pl
from quantlab import data
from quantlab.costs import load_instrument
from quantlab.engine import run_backtest
from quantlab.sizing import apply_sizing
from sma_strategy import SmaCross, SmaParams

spec = load_instrument("USDJPY")
bars = data.load_bars("USDJPY", "H1", start="2019-01-01", end="2021-01-01")
tr = run_backtest(bars, SmaCross(SmaParams(far_target=True)).signals(bars), spec).trades
calls = []
def rate_fn(q, a, ts):
    calls.append(ts)
    return data.conversion_rate(q, a, ts)
try:
    apply_sizing(tr, spec, mode="fixed_fraction", risk_fraction=0.01, rate_fn=rate_fn)
    print("apply_sizing without bars= accepted -> STILL OPEN")
except ValueError as e:
    print("apply_sizing without bars= rejected:", str(e)[:70])
s = apply_sizing(tr, spec, mode="fixed_fraction", risk_fraction=0.01, rate_fn=rate_fn, bars=bars)
print("rate_fn received dtypes:", [str(c.dtype) for c in calls])
m1 = data.load_bars("USDJPY", "M1", start="2018-12-31", end="2021-01-01").select("ts", "close")
def indep(ts_naive):   # close of the last M1 bar with ts+1min <= t (closed), independent implementation
    idx = np.searchsorted(m1["ts"].to_numpy(), np.array(ts_naive, dtype="datetime64[ms]") - np.timedelta64(1, "m"), side="right") - 1
    return 1.0 / m1["close"].to_numpy()[idx]
ent = indep(s["entry_ts"].to_list()); ext = indep(s["exit_ts"].to_list())
vpp = spec.value_per_point_per_lot
print(f"entry rates match independent closed-bar as-of: {np.allclose(s['value_per_point_acct'].to_numpy()/vpp, ent)}; "
      f"exit rates match: {np.allclose(s['value_per_point_acct_exit'].to_numpy()/vpp, ext)}")
print(f"risk_realisation_error max={s['risk_realisation_error'].max():.6f} (must be <= 0); final equity={s['equity_after'][-1]:.2f}")
