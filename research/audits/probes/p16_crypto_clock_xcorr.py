"""Probe 16: does the crypto feed's server clock observe EU DST (like FX) or run on a fixed offset (as the
new data.py assumes)? Cross-correlate |1-min returns| of BTCUSD vs XAUUSD/EURUSD keyed by naive server ts,
summer vs winter, per year. Same clock => peak at lag 0 in both seasons. Fixed-offset crypto clock =>
summer peak at a 60-minute lag. Then compare the ts_utc that load_bars assigns."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import polars as pl
from quantlab import data

def absret(sym, s, e):
    b = data.load_bars(sym, "M1", start=s, end=e).select("ts", "ts_utc", "close")
    return b.with_columns(pl.col("close").log().diff().abs().alias("r")).drop_nulls()

def best_lag(a, b, key):
    j = a.select(key, pl.col("r").alias("a")).join(b.select(key, pl.col("r").alias("b")), on=key, how="inner")
    # build dense minute grid on key for lagging
    j = j.sort(key)
    out = {}
    base = a.select(key, pl.col("r").alias("a"))
    for lag in (-120, -60, 0, 60, 120):
        bb = b.select((pl.col(key) + pl.duration(minutes=lag)).alias(key), pl.col("r").alias("b"))
        jj = base.join(bb, on=key, how="inner")
        out[lag] = float(np.corrcoef(jj["a"].to_numpy(), jj["b"].to_numpy())[0, 1]) if jj.height > 100 else float("nan")
    return out

for yr in (2019, 2022, 2024):
    for season, (s, e) in {"summer": (f"{yr}-06-03", f"{yr}-08-30"), "winter": (f"{yr}-01-07", f"{yr}-03-01")}.items():
        btc = absret("BTCUSD", s, e)
        for ref in ("XAUUSD", "EURUSD"):
            x = absret(ref, s, e)
            by_server = best_lag(btc, x, "ts")
            by_utc = best_lag(btc, x, "ts_utc")
            print(f"{yr} {season} BTC vs {ref}: corr by naive server ts at lag(min) {by_server} | by library ts_utc {by_utc}")
