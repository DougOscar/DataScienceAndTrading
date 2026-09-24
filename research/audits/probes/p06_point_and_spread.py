"""Probe 06: infer_point per symbol (drives spread->price scaling) and spread by server hour
(rollover spike). Which spread does each timeframe's bar carry?"""
import warnings
warnings.filterwarnings("ignore")
import polars as pl
from quantlab import data
pl.Config.set_tbl_rows(40)
cat = data.catalog("FBS").filter(pl.col("timeframe") == "M1")
out = []
for sym in cat["symbol"].to_list():
    p = data.infer_point(sym)
    m1 = data.load_bars(sym, "M1", start="2019-06-01", end="2019-07-01")
    dec = m1.select(pl.col("close").map_elements(lambda x: len(f"{x:.8f}".rstrip("0").split(".")[1]), return_dtype=pl.Int64).max()).item() if m1.height else None
    med = m1["spread"].median() if m1.height else None
    out.append({"symbol": sym, "infer_point": p, "max_decimals_2019_06": dec, "median_spread_pts": med,
                "median_spread_price": (med * p) if med is not None else None, "median_close": m1["close"].median() if m1.height else None})
print(pl.DataFrame(out))

for sym in ["EURUSD", "XAUUSD", "USDJPY"]:
    m1 = data.load_bars(sym, "M1", start="2019-01-01", end="2021-01-01")
    by = m1.group_by(pl.col("ts").dt.hour().alias("h")).agg(pl.col("spread").median().alias("med"), pl.col("spread").mean().alias("mean")).sort("h")
    first = m1.filter(pl.col("ts").dt.minute() == 0).group_by(pl.col("ts").dt.hour().alias("h")).agg(pl.col("spread").median().alias("med_at_:00")).sort("h")
    j = by.join(first, on="h")
    print(sym, "median M1 spread by server hour:", dict(zip(j["h"].to_list(), j["med"].to_list())))
    print(sym, "median spread of the :00 M1 bar (= H1 bar spread):", dict(zip(j["h"].to_list(), j["med_at_:00"].to_list())))
    d1 = data.load_bars(sym, "D1", start="2019-01-01", end="2021-01-01")
    print(sym, "D1 bar spread (first M1 of day) median:", d1["spread"].median(), "vs all-M1 median", m1["spread"].median())
