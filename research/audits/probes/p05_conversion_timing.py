"""Probe 05: quote->account conversion timing.
apply_sizing calls rate_fn(quote, acct, trades['entry_ts']) where entry_ts is NAIVE SERVER time.
data.conversion_rate expects ts_utc; a naive series is silently treated as UTC, so the rate is
read 2h (winter) / 3h (summer) AFTER the actual entry -> look-ahead in sizing & P&L conversion.
Also: as-of backward on M1 'ts' returns the close of the M1 bar that OPENS at t, i.e. a price
that is only known at t+1min."""
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
import polars as pl
from quantlab import data

naive = pl.Series("entry_ts", [datetime(2019, 7, 3, 10, 0), datetime(2019, 1, 9, 10, 0)]).cast(pl.Datetime("ms"))
r_naive = data.conversion_rate("JPY", "USD", naive)
aware = naive.dt.replace_time_zone("Europe/Helsinki").dt.convert_time_zone("UTC")
r_aware = data.conversion_rate("JPY", "USD", aware)
m1 = data.load_bars("USDJPY", "M1", start="2019-01-09", end="2019-07-04")
for i, t in enumerate(naive.to_list()):
    def c_at(x):
        return m1.filter(pl.col("ts") == x)["close"][0]
    def o_at(x):
        return m1.filter(pl.col("ts") == x)["open"][0]
    print(f"entry {t} server: rate(naive series)={r_naive[i]:.8f} -> 1/{1/r_naive[i]:.3f}; "
          f"rate(aware)={r_aware[i]:.8f} -> 1/{1/r_aware[i]:.3f}")
    print(f"   USDJPY M1 close@{t}={c_at(t)} (known only at {t+timedelta(minutes=1)}), open@{t}={o_at(t)}, "
          f"close@{t-timedelta(minutes=1)}={c_at(t-timedelta(minutes=1))}")
    for h in (2, 3):
        tt = t + timedelta(hours=h)
        try:
            print(f"   close@{tt} (= naive treated as UTC, +{h}h) = {c_at(tt)}")
        except Exception:
            pass

# Does apply_sizing pass naive server timestamps? (signature/docs)
from quantlab import sizing
import inspect
print("apply_sizing passes trades['entry_ts'] (naive server time):", 'rate_fn(spec.quote_ccy, account_ccy, trades["entry_ts"])' in inspect.getsource(sizing.apply_sizing))
