"""Probe 15: crypto (24/7) bars exist at server hour 03:xx on EU spring-forward days 2017-2020, an hour that
does not exist in Europe/Helsinki. _localize_expr shifts them +1h, so ts_utc goes BACKWARDS and duplicates."""
import warnings
warnings.filterwarnings("ignore")
import polars as pl
from quantlab import data
b = data.load_bars("BTCUSD", "M1", start="2016-06-01", end="2025-05-01")
u = b["ts_utc"]
print("BTCUSD M1 rows", b.height, "| duplicated ts_utc", int(u.is_duplicated().sum()), "| backward steps", int((u.diff().dt.total_minutes() <= 0).sum()))
print(b.with_columns(u.diff().dt.total_minutes().alias("dm")).filter(pl.col("dm") <= 0).select("ts", "ts_utc", "dm"))
for day in ["2019-03-31", "2020-03-29", "2021-03-28"]:
    s = b.filter(pl.col("ts").dt.date() == pl.lit(day).str.to_date())
    print(day, "rows at non-existent server hour 03:xx:", s.filter(pl.col("ts").dt.hour() == 3).height)
