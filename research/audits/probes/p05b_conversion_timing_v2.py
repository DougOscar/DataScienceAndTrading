"""Probe 05b (re-verification): conversion_rate must (1) reject naive series, (2) return the close of the
last M1 bar whose CLOSE time <= t (i.e. the bar opened at t-1min), never the bar opening at t;
(3) triangulated and inverse paths obey the same rule; (4) crypto path (fixed-offset clock) consistency."""
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
import polars as pl
from quantlab import data

naive = pl.Series("entry_ts", [datetime(2019, 7, 3, 10, 0), datetime(2019, 1, 9, 10, 0), datetime(2019, 1, 9, 10, 0, 30)]).cast(pl.Datetime("ms"))
try:
    data.conversion_rate("JPY", "USD", naive); print("naive accepted -> STILL OPEN")
except ValueError as e:
    print("naive rejected:", str(e)[:80])
aware = naive.dt.replace_time_zone("Europe/Helsinki").dt.convert_time_zone("UTC")
r = data.conversion_rate("JPY", "USD", aware)
m1 = data.load_bars("USDJPY", "M1", start="2019-01-09", end="2019-07-04")
for i, t in enumerate(naive.to_list()):
    tm = t.replace(second=0)
    prev_close = m1.filter(pl.col("ts") == tm - timedelta(minutes=1))["close"][0]
    this_close = m1.filter(pl.col("ts") == tm)["close"][0]
    print(f"t={t} server: 1/rate={1/r[i]:.3f} | close of bar opened {tm - timedelta(minutes=1)} (closed {tm}) = {prev_close}"
          f" | close of bar opened {tm} (closes {tm + timedelta(minutes=1)}) = {this_close} -> "
          f"{'OK (closed bar)' if abs(1/r[i]-prev_close) < 1e-9 else ('LOOKAHEAD' if abs(1/r[i]-this_close) < 1e-9 else '?')}")
# triangulation EUR->JPY = EURUSD * USDJPY both as-of closed bars
r2 = data.conversion_rate("EUR", "JPY", aware[:1])
e = data.load_bars("EURUSD", "M1", start="2019-07-03 09:00", end="2019-07-03 10:00")["close"][-1]
u = m1.filter(pl.col("ts") == datetime(2019, 7, 3, 9, 59))["close"][0]
ej = data.load_bars("EURJPY", "M1", start="2019-07-03 09:00", end="2019-07-03 10:00")["close"][-1]
print(f"EUR->JPY {r2[0]:.5f} (direct EURJPY exists) vs EURJPY 09:59-bar close {ej}")
r3 = data.conversion_rate("NZD", "CHF", aware[:1])
nu = data.load_bars("NZDCHF", "M1", start="2019-07-03 09:00", end="2019-07-03 10:00")["close"][-1]
print(f"NZD->CHF {r3[0]:.5f} vs NZDCHF 09:59-bar close {nu}")
