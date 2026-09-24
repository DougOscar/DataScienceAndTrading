"""Probe 03: resampled bars vs manual recomputation from raw M1; ts = open label; ts_utc around DST;
D1 boundary vs 17:00 New York."""
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import polars as pl
from quantlab import data

rng = np.random.default_rng(7)
TFM = {"H1": 60, "H4": 240, "D1": 1440}
bad = 0
for sym in ["EURUSD", "XAUUSD", "USDJPY"]:
    m1 = data.load_bars(sym, "M1", start="2019-01-01", end="2021-01-01")
    for tf, mins in TFM.items():
        htf = data.load_bars(sym, tf, start="2019-01-01", end="2021-01-01")
        idx = rng.choice(htf.height, 200, replace=False)
        for i in idx:
            r = htf.row(int(i), named=True)
            w = m1.filter((pl.col("ts") >= r["ts"]) & (pl.col("ts") < r["ts"] + timedelta(minutes=mins)))
            exp = dict(open=w["open"][0], high=w["high"].max(), low=w["low"].min(), close=w["close"][-1],
                       spread=float(w["spread"][0]), spread_max=float(w["spread"].max()), tick_vol=int(w["tick_vol"].sum()))
            for k, v in exp.items():
                if r[k] != v:
                    bad += 1
                    print("MISMATCH", sym, tf, r["ts"], k, r[k], v)
        # label check: every ts aligned to the timeframe grid from server midnight
        mis = htf.filter((pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute().cast(pl.Int32)) % mins != 0).height
        print(f"{sym} {tf}: {htf.height} bars, 200 random bars checked; off-grid labels={mis}")
print("total value mismatches:", bad)

# ts_utc checks around DST (Europe/Helsinki) incl US/EU mismatch weeks
h1 = data.load_bars("EURUSD", "H1", start="2019-03-01", end="2019-11-15")
hel = ZoneInfo("Europe/Helsinki")
chk = h1.with_columns(
    pl.col("ts").map_elements(lambda t: t.replace(tzinfo=hel).astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
                              return_dtype=pl.Datetime("us")).alias("exp_utc"))
chk = chk.with_columns(pl.col("ts_utc").dt.replace_time_zone(None).cast(pl.Datetime("us")).alias("got"))
print("ts_utc mismatches vs zoneinfo (2019 Mar-Nov):", chk.filter(pl.col("got") != pl.col("exp_utc")).height)
for d in ["2019-03-08", "2019-03-11", "2019-03-29", "2019-04-01", "2019-10-25", "2019-10-28", "2019-11-01", "2019-11-04"]:
    day = h1.filter(pl.col("ts").dt.date() == datetime.fromisoformat(d).date())
    if day.height:
        print(d, "first bar", day["ts"][0], "utc", day["ts_utc"][0], "| last bar", day["ts"][-1], "utc", day["ts_utc"][-1], "| n", day.height)

# D1 boundary in New York time for each week of 2019
d1 = data.load_bars("EURUSD", "D1", start="2019-01-01", end="2020-01-01")
ny = d1.with_columns(pl.col("ts_utc").dt.convert_time_zone("America/New_York").dt.hour().alias("ny_hour"))
print("D1 open hour in New York (count):", ny.group_by("ny_hour").len().sort("ny_hour").to_dicts())
print("D1 bars opening at NY hour != 17:", ny.filter(pl.col("ny_hour") != 17).select("ts", "ts_utc").to_dicts()[:12])
# Sunday/Saturday D1 bars?
print("D1 weekday counts:", d1.group_by(pl.col("ts").dt.weekday()).len().sort("ts").to_dicts())

# crypto around autumn fall-back (ambiguous hour) -- BTCUSD 2019-10-27
b = data.load_bars("BTCUSD", "M1", start="2019-10-27 02:50", end="2019-10-27 04:10")
u = b["ts_utc"]
print("BTC M1 around fall-back: n", b.height, "dup ts_utc", u.is_duplicated().sum(), "non-increasing", (u.diff().dt.total_seconds() <= 0).sum())
bb = b.filter(pl.col("ts").dt.hour() == 3)
print("  03:xx server rows:", bb.height, "(a 24/7 feed would have 120 real minutes in the repeated hour if server clock repeats)")
b = data.load_bars("BTCUSD", "M1", start="2019-03-31 02:50", end="2019-03-31 04:10")
print("BTC M1 around spring-forward: hours present", b["ts"].dt.hour().unique().to_list(), "n", b.height)
